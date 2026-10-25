"""The committed result records, as tables the app can chart.

Everything here reads ``results/manifests/*.json``, which are in git, so the
results pages work on any machine -- no datasets, feature banks or benchmark
CSVs needed. Each loader returns a pandas DataFrame plus nothing else; the app
decides how to draw it. A missing record returns an empty frame rather than
raising, so a partial checkout still opens.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.utils.paths import repo_root

DATASETS = {"mvtec_ad": "MVTec AD", "visa": "VisA", "neu_det": "NEU-DET"}

BACKBONE_LABELS = {
    "ViT-B-32-quickgelu": "ViT-B/32",
    "ViT-B-16-quickgelu": "ViT-B/16",
    "ViT-L-14-quickgelu": "ViT-L/14",
    "ViT-B-32-quickgelu-int8": "ViT-B/32 INT8",
    "ViT-B-16-quickgelu-int8": "ViT-B/16 INT8",
    "ViT-B-16-quickgelu-ml6-9-12": "ViT-B/16 multi-layer",
}


def manifests_dir() -> Path:
    return repo_root() / "results" / "manifests"


@lru_cache(maxsize=None)
def record(name: str) -> dict:
    path = manifests_dir() / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def label(backbone: str) -> str:
    return BACKBONE_LABELS.get(backbone, backbone)


def _figures(dataset: str) -> dict:
    name = "figures.json" if dataset == "mvtec_ad" else f"figures_{dataset}.json"
    return record(name).get("figures", {})


def shots_curves(dataset: str) -> pd.DataFrame:
    """Mean image AUROC per shot count, one row per (model, shots)."""
    rows = []
    for d in _figures(dataset).get("shots_vs_accuracy", {}).get("data", []):
        for shots, auroc in zip(d["shots"], d["mean_image_auroc"]):
            rows.append({"model": label(d["backbone"]), "shots": shots, "image AUROC": auroc})
    return pd.DataFrame(rows)


def frontier(dataset: str) -> pd.DataFrame:
    rows = [{"model": label(d["backbone"]), "shots": d["shots"],
             "ms per part": d["total_ms_per_image"], "image AUROC": d["image_auroc"]}
            for d in _figures(dataset).get("accuracy_latency_frontier", {}).get("data", [])]
    return pd.DataFrame(rows)


def _paired(name: str, metric: str = "image_auroc") -> dict:
    return record(name).get("per_metric", {}).get(metric, {})


def versus_patchcore(dataset: str) -> pd.DataFrame:
    """Ours against few-shot PatchCore given the same photos."""
    configs = [("b_16", "ViT-B/16"), ("b_16_windows", "ViT-B/16 + windows"), ("l_14", "ViT-L/14"),
               ("ml", "ViT-B/16 multi-layer + windows")]
    rows = []
    for metric in ("image_auroc", "pixel_auroc"):
        pc_done = set()
        for tag, name in configs:
            for shots, r in _paired(f"patchcore_vs_{tag}_{dataset}.json", metric).items():
                rows.append({"model": name, "metric": metric.split("_")[0], "shots": int(shots),
                             "AUROC": r["mean_a"], "delta vs PatchCore": r["paired_mean_delta"],
                             "x SE": r["effect_over_se"]})
                if shots not in pc_done:
                    pc_done.add(shots)
                    rows.append({"model": "PatchCore (same photos)", "metric": metric.split("_")[0],
                                 "shots": int(shots), "AUROC": r["mean_b"],
                                 "delta vs PatchCore": 0.0, "x SE": None})
    return pd.DataFrame(rows)


def ablation(dataset: str) -> pd.DataFrame:
    """What each scoring idea adds, image AUROC by shot count (ViT-B/16)."""
    rows = []

    def add(step, shots, value):
        rows.append({"scoring": step, "shots": int(shots), "image AUROC": value})

    if dataset in ("mvtec_ad", "visa"):
        plain = record("text_fusion.json").get("datasets", {}).get(dataset, {}).get("per_shot", {})
        win = record("text_fusion_win123.json").get("datasets", {}).get(dataset, {}).get("per_shot", {})
        for shots, r in plain.items():
            add("memory only", shots, r["0.0"]["mean_image_auroc"])
            add("+ text", shots, r["0.5"]["mean_image_auroc"])
            add("text only (zero-shot)", shots, r["1.0"]["mean_image_auroc"])
        for shots, r in win.items():
            add("+ windows", shots, r["0.0"]["mean_image_auroc"])
            add("+ windows + text", shots, r["0.5"]["mean_image_auroc"])
    else:
        for shots, r in _paired("neu_det_windowed.json").items():
            add("memory only", shots, r["mean_b"])
            add("+ windows", shots, r["mean_a"])
        for shots, r in _paired("neu_det_text.json").items():
            add("+ windows + text", shots, r["mean_a"])
    for shots, r in _paired(f"ml_vs_b16_{dataset}.json").items():
        add("multi-layer", shots, r["mean_a"])
    for shots, r in _paired(f"ml_windows_vs_b16_{dataset}.json").items():
        add("multi-layer + windows", shots, r["mean_a"])
    return pd.DataFrame(rows)


def sustained() -> pd.DataFrame:
    rows = []
    for backbone, rec in record("sustained_torch.json").get("backbones", {}).items():
        for w in rec["windows"]:
            rows.append({"model": label(backbone), "minute": w["t_min"], "ms per part": w["median_ms"]})
    return pd.DataFrame(rows)


def cost_table() -> pd.DataFrame:
    rows = [{
        "configuration": r["config"].replace("ViT-B-", "ViT-B/").replace("ViT-L-", "ViT-L/"),
        "ms per part (cool)": round(r["ms_per_image"]),
        "ms per part (after 5 min)": round(r["sustained_ms_per_image"]) if r["sustained_ms_per_image"] else None,
        "RAM MB": round(r["ram_mb"]) if r["ram_mb"] else None,
        "Rs per year": round(r["inr_per_year"], 2),
        "busy hours / day": round(r["hours_per_day_busy_worst"], 2),
    } for r in record("cost_feasibility.json").get("per_config", [])]
    return pd.DataFrame(rows)


def int8_table() -> pd.DataFrame:
    rows = []
    for ds, files in (("MVTec AD", ("int8_vit_b32.json", "int8_vit_b16.json")),
                      ("VisA", ("int8_vit_b32_visa.json", "int8_vit_b16_visa.json"))):
        for f, model in zip(files, ("ViT-B/32", "ViT-B/16")):
            r = _paired(f).get("10")
            if r:
                rows.append({"dataset": ds, "model": model, "FP32": r["mean_a"], "INT8": r["mean_b"],
                             "AUROC lost to INT8": r["paired_mean_delta"]})
    lat = record("int8_latency.json").get("backbones", {})
    speed = {label(b): (max(v["fp32_ms_bs1_runs"]), max(v["int8_ms_bs1_runs"]), v["fp32_mb"], v["int8_mb"])
             for b, v in lat.items()}
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ms FP32"] = df["model"].map(lambda m: speed.get(m, (None,) * 4)[0])
        df["ms INT8"] = df["model"].map(lambda m: speed.get(m, (None,) * 4)[1])
    return df


def headline() -> dict:
    """The handful of numbers the overview page leads with."""
    def mean_at(dataset, model, shots=10):
        df = shots_curves(dataset)
        hit = df[(df["model"] == model) & (df["shots"] == shots)]
        return float(hit["image AUROC"].iloc[0]) if len(hit) else None

    win = record("text_fusion_win123.json").get("datasets", {})
    proposed = {ds: win.get(ds, {}).get("per_shot", {}).get("10", {}).get("0.5", {}).get("mean_image_auroc")
                for ds in ("mvtec_ad", "visa")}
    holdout = record("incremental_holdout_10shot.json")
    cost = {r["config"]: r for r in record("cost_feasibility.json").get("per_config", [])}
    neu_win = _paired("neu_det_windowed.json").get("10", {}).get("mean_a")
    return {
        "mvtec_best": mean_at("mvtec_ad", "ViT-L/14"),
        "mvtec_proposed": proposed["mvtec_ad"],
        "visa_proposed": proposed["visa"],
        "neu_windows": neu_win,
        "fastest_ms": cost.get("ViT-B-32 ONNX INT8", {}).get("ms_per_image"),
        "rs_per_year_b16": cost.get("ViT-B-16 torch FP32", {}).get("inr_per_year"),
        "append_bit_identical": holdout.get("no_regression_all_seeds") and holdout.get(
            "appended_equals_scratch_all_seeds"),
        "append_seconds": max((v["median"] for v in holdout.get("append_seconds_end_to_end", {}).values()),
                              default=None),
    }
