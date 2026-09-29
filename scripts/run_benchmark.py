"""Run the few-shot benchmark across categories, shot counts and seeds.

    python scripts/run_benchmark.py                          # everything, ViT-B/16
    python scripts/run_benchmark.py --categories bottle cable
    python scripts/run_benchmark.py --backbone ViT-B-32-quickgelu
    python scripts/run_benchmark.py --precision int8         # needs export_onnx.py first

Features are extracted once per (backbone, category) and cached, so the 0/1/5/10
shot runs and the three seeds all reuse a single pass over the images. Re-running
with a warm cache costs seconds.

Writes a tidy CSV: one row per (backbone, category, shots, seed).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mvtec import CATEGORIES, complete_categories  # noqa: E402
from src.data.splits import categories as manifest_categories  # noqa: E402
from src.data.splits import category_root  # noqa: E402
from src.data.splits import (  # noqa: E402
    load_manifest,
    manifest_path,
    manifest_paths,
    shot_counts,
    split_for,
)
from src.eval.features import FeatureBank, bank_path  # noqa: E402
from src.eval.pipeline import evaluate_split  # noqa: E402
from src.model.encoder import FrozenCLIPEncoder  # noqa: E402
from src.model.export import OnnxEncoder, onnx_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import dataset_dir, repo_root, results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def splits_for(
    manifests: list[dict],
    category: str,
    root: Path,
    shots: list[int],
) -> list[dict]:
    """Every (seed, shot count) split for one category, across all seeds."""
    found = []
    for manifest in manifests:
        for k in shots:
            try:
                found.append(split_for(manifest, category, k, root))
            except KeyError:
                continue
    return found


def get_bank(encoder, root, category, manifests, cache_dir, batch_size, workers, refresh):
    path = bank_path(cache_dir, encoder.name, category, encoder.embed_dim)
    needed = {p for m in manifests for p in m["support"]}
    needed |= {t["image"] for m in manifests for t in m["test"]}

    if path.exists() and not refresh:
        bank = FeatureBank.load(path)
        if all(p in bank for p in needed):
            return bank, "cached"

    bank = FeatureBank.build(
        encoder, root, category, sorted(needed), batch_size=batch_size, num_workers=workers
    )
    bank.save(path)
    return bank, "extracted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="prototype.yaml")
    parser.add_argument("--backbone", default=None, help="overrides the config")
    parser.add_argument("--dataset", default=None, help="overrides the config")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=["memory"],
                        choices=["memory", "adapter"],
                        help="scoring heads to compare; features are shared, so "
                             "running both costs almost nothing extra")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None,
                        help="pin torch threads so timings are comparable")
    parser.add_argument("--project-patches", dest="project_patches",
                        action="store_true", default=None,
                        help="512-dim joint image-text space; enables prompting")
    parser.add_argument("--no-project-patches", dest="project_patches",
                        action="store_false",
                        help="768-dim raw transformer width (Member B's convention)")
    parser.add_argument("--precision", default=None, choices=["fp32", "int8"],
                        help="overrides the config; int8 runs the exported ONNX graph")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch

    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    if args.threads:
        torch.set_num_threads(args.threads)

    backbone = args.backbone or config["backbone"]["name"]
    pretrained = config["backbone"]["pretrained"]
    dataset = args.dataset or config["dataset"]["name"]
    precision = args.precision or config["backbone"].get("precision", "fp32")
    # directory that directly contains the category folders
    root = category_root(dataset, dataset_dir(dataset).parent)
    splits_dir = repo_root() / "results" / "manifests" / "splits"
    # fp32 keeps its original file name so earlier results stay where they were
    tag = backbone if precision == "fp32" else f"{backbone}-{precision}"
    out_path = (Path(args.out) if args.out
                else results_root() / "benchmark" / f"{dataset}_{tag}.csv")

    # Member B's manifest is the single definition of a split
    paths = manifest_paths(splits_dir, dataset)
    if not paths:
        print(f"no split manifest for {dataset} in {splits_dir}", file=sys.stderr)
        return 1
    manifests = [load_manifest(p) for p in paths]
    manifest = manifests[0]                     # for category / shot discovery
    data_root_for_splits = dataset_dir(dataset).parent   # her paths include the dataset dir

    project_patches = (
        args.project_patches
        if args.project_patches is not None
        else config["backbone"].get("project_patches", True)
    )
    wanted_shots = shot_counts(manifest)
    if not project_patches:
        # 768-dim tokens are not in the joint image-text space, so a prompt
        # cannot score a patch; zero-shot is unavailable rather than wrong
        wanted_shots = [k for k in wanted_shots if k > 0]

    # complete_categories() verifies against MVTec's published per-category
    # counts; for any other dataset fall back to "the directory exists"
    if dataset == "mvtec_ad":
        present = set(complete_categories(root))
    else:
        present = {c for c in manifest_categories(manifest) if (root / c).is_dir()}
    categories = args.categories or [
        c for c in manifest_categories(manifest) if c in present
    ]
    if not categories:
        print(f"no complete categories under {root}", file=sys.stderr)
        return 1

    print(f"backbone   {backbone} / {pretrained}")
    print(f"dataset    {root}")
    seeds = [m["seed"] for m in manifests]
    print(f"splits     {len(manifests)} manifest(s), seeds {seeds}")
    print(f"categories {len(categories)}   shots {wanted_shots}")
    print(f"patch dim  {'768 unprojected' if not project_patches else '512 projected'}")
    print(f"precision  {precision}")
    print(f"threads    {torch.get_num_threads()}\n")

    if precision == "int8":
        # the graph cannot tell us its dim until loaded, so ask the torch model
        # config instead of building the whole encoder just to read one number
        import open_clip
        cfg = open_clip.get_model_config(backbone)
        dim = cfg["embed_dim"] if project_patches else cfg["vision_cfg"]["width"]
        graph = onnx_path(results_root() / "onnx", backbone, dim, "int8")
        if not graph.exists():
            print(f"no int8 graph at {graph}; run scripts/export_onnx.py first",
                  file=sys.stderr)
            return 1
        encoder = OnnxEncoder(graph, threads=args.threads)
    else:
        encoder = FrozenCLIPEncoder(
            backbone, pretrained, device="cpu", project_patches=project_patches
        )
    print(f"{encoder}\n")

    rows = []
    started = time.perf_counter()
    for i, category in enumerate(categories, 1):
        splits = splits_for(manifests, category, data_root_for_splits, wanted_shots)
        if not splits:
            print(f"[{i}/{len(categories)}] {category}: no splits, skipping")
            continue

        t0 = time.perf_counter()
        bank, how = get_bank(
            encoder, root, category, splits,
            results_root(), args.batch_size, args.workers, args.refresh_cache,
        )
        extract_seconds = time.perf_counter() - t0

        produced = 0
        for method in args.methods:
            # `split`, not `manifest`: the dataset manifest is needed on the
            # next category and must not be shadowed here
            for split in splits:
                # 0-shot is text-only, identical under both heads: score it once
                if split["shots"] == 0 and method != args.methods[0]:
                    continue
                result = evaluate_split(
                    split, root, bank,
                    encoder=encoder,
                    top_k=config["model"]["scoring"]["top_k"],
                    method=method,
                    adapter_config=config["model"].get("adapter"),
                    temperature=config["model"]["scoring"]["temperature"],
                    image_size=tuple(config["dataset"]["image_size"]),
                )
                rows.append(result.as_row())
                produced += 1

        best = {}
        for r in rows[-produced:]:
            best.setdefault((r["method"], r["shots"]), []).append(r["image_auroc"])
        summary = "  ".join(
            f"{m[:3]}{k}={sum(v) / len(v):.3f}" for (m, k), v in sorted(best.items())
        )
        print(
            f"[{i}/{len(categories)}] {category:<12} {how:<10} "
            f"{len(bank)} imgs {extract_seconds:5.1f}s   {summary}"
        )

    total = time.perf_counter() - started

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    meta = out_path.with_suffix(".meta.json")
    meta.write_text(json.dumps({
        "dataset": dataset,
        "backbone": backbone,
        "pretrained": pretrained,
        "precision": precision,
        "patch_dim": encoder.embed_dim,
        "seeds": seeds,
        "top_k": config["model"]["scoring"]["top_k"],
        "categories": categories,
        "methods": args.methods,
        "rows": len(rows),
        "threads": torch.get_num_threads(),
        "total_seconds": round(total, 1),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(rows)} rows -> {out_path}")
    print(f"total {total / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
