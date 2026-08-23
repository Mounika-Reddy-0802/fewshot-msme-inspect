"""Run the few-shot benchmark across categories, shot counts and seeds.

    python scripts/run_benchmark.py                          # everything, ViT-B/16
    python scripts/run_benchmark.py --categories bottle cable
    python scripts/run_benchmark.py --backbone ViT-B-32-quickgelu

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
from src.data.splits import load_split  # noqa: E402
from src.eval.features import FeatureBank, bank_path  # noqa: E402
from src.eval.pipeline import evaluate_split  # noqa: E402
from src.model.encoder import FrozenCLIPEncoder  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import dataset_dir, repo_root, results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def manifests_for(splits_dir: Path, category: str) -> list[dict]:
    """Every manifest for one category, ordered by shots then seed."""
    found = []
    for path in sorted(splits_dir.glob(f"*_{category}_*shot_seed*.json")):
        manifest = load_split(path)
        if manifest["category"] == category:      # guards metal_nut vs metal
            found.append(manifest)
    return sorted(found, key=lambda m: (m["shots"], m["seed"]))


def get_bank(encoder, root, category, manifests, cache_dir, batch_size, workers, refresh):
    path = bank_path(cache_dir, encoder.name, category)
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
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=["memory"],
                        choices=["memory", "adapter"],
                        help="scoring heads to compare; features are shared, so "
                             "running both costs almost nothing extra")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None,
                        help="pin torch threads so timings are comparable")
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
    root = dataset_dir(config["dataset"]["name"])
    splits_dir = repo_root() / "results" / "manifests" / "splits"
    out_path = Path(args.out) if args.out else results_root() / "benchmark" / f"{backbone}.csv"

    categories = args.categories or complete_categories(root)
    if not categories:
        print(f"no complete categories under {root}", file=sys.stderr)
        return 1

    print(f"backbone   {backbone} / {pretrained}")
    print(f"dataset    {root}")
    print(f"categories {len(categories)}")
    print(f"threads    {torch.get_num_threads()}\n")

    encoder = FrozenCLIPEncoder(backbone, pretrained, device="cpu")
    print(f"{encoder}\n")

    rows = []
    started = time.perf_counter()
    for i, category in enumerate(categories, 1):
        manifests = manifests_for(splits_dir, category)
        if not manifests:
            print(f"[{i}/{len(categories)}] {category}: no manifests, skipping")
            continue

        t0 = time.perf_counter()
        bank, how = get_bank(
            encoder, root, category, manifests,
            results_root(), args.batch_size, args.workers, args.refresh_cache,
        )
        extract_seconds = time.perf_counter() - t0

        produced = 0
        for method in args.methods:
            for manifest in manifests:
                # 0-shot is text-only, identical under both heads: score it once
                if manifest["shots"] == 0 and method != args.methods[0]:
                    continue
                result = evaluate_split(
                    manifest, root, bank,
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
        "backbone": backbone,
        "pretrained": pretrained,
        "seed": config["experiment"]["seed"],
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
