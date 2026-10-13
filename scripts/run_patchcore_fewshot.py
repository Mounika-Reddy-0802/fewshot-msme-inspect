"""Few-shot PatchCore on our committed splits: the baseline comparison.

    python scripts/run_patchcore_fewshot.py --dataset mvtec_ad
    python scripts/run_patchcore_fewshot.py --dataset visa --categories candle pcb1

Same manifests, seeds (42, 1, 2), shot counts (1/5/10) and test images as
run_benchmark.py, scored with the same metric code, written in the same CSV
format as ``<dataset>_patchcore-wideresnet502.csv``, so compare_paired.py can
pair it against any of our runs. See src/baselines/patchcore_fewshot.py for the
PatchCore settings.

Test images are embedded once per category and reused for every (seed, shots)
split; only the support set changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.baselines.patchcore_fewshot import FewShotPatchCore  # noqa: E402
from src.data.splits import (  # noqa: E402
    categories as manifest_categories,
    category_root,
    load_manifest,
    manifest_paths,
    split_for,
)
from src.eval.pipeline import load_masks  # noqa: E402
from src.eval.scoring import compute_metrics  # noqa: E402
from src.utils.paths import dataset_dir, repo_root, results_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mvtec_ad")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data_root = dataset_dir(args.dataset).parent
    root = category_root(args.dataset, data_root)
    manifests = [load_manifest(p) for p in
                 manifest_paths(repo_root() / "results" / "manifests" / "splits", args.dataset)]
    categories = args.categories or manifest_categories(manifests[0])

    pc = FewShotPatchCore()
    out_path = Path(args.out) if args.out else (
        results_root() / "benchmark" / f"{args.dataset}_{pc.name}.csv")
    print(f"{pc.name} on {args.dataset}: {len(categories)} categories, "
          f"{len(manifests)} seeds, shots {args.shots}\n", flush=True)

    rows, started = [], time.perf_counter()
    for i, category in enumerate(categories, 1):
        t0 = time.perf_counter()
        base = root / category
        first = split_for(manifests[0], category, args.shots[0], data_root)
        test = first["test"]
        test_emb, grid = pc.embed([base / t["image"] for t in test])
        labels = torch.tensor([t["label"] for t in test])
        masks = load_masks([base / t["mask"] if t["mask"] else None for t in test], (pc.size, pc.size))

        support_cache: dict[str, torch.Tensor] = {}
        summary = []
        for manifest in manifests:
            for k in args.shots:
                split = split_for(manifest, category, k, data_root)
                assert [t["image"] for t in split["test"]] == [t["image"] for t in test], \
                    "test set must not change with seed or shots"
                missing = [p for p in split["support"] if p not in support_cache]
                if missing:
                    emb, _ = pc.embed([base / p for p in missing])
                    support_cache.update(zip(missing, emb))
                support = torch.stack([support_cache[p] for p in split["support"]])
                image_scores, maps = pc.score(support, test_emb, grid)
                metrics = compute_metrics(image_scores=image_scores, image_labels=labels,
                                          pixel_scores=maps, pixel_labels=masks)
                row = {
                    "backbone": pc.name, "method": "patchcore", "category": category,
                    "shots": k, "seed": manifest["seed"], **metrics.as_dict(),
                    "support_size": k, "test_size": len(test),
                    "prototypes": int(support.shape[0] * support.shape[1]),
                    "memory_bytes": int(support.numel() * 4),
                }
                rows.append(row)
                summary.append((k, row["image_auroc"]))
        means = {k: sum(a for s, a in summary if s == k) / len(manifests) for k in args.shots}
        print(f"[{i}/{len(categories)}] {category:<16} {time.perf_counter() - t0:6.1f}s   "
              + "  ".join(f"pc{k}={v:.3f}" for k, v in means.items()), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out_path.with_suffix(".meta.json").write_text(json.dumps({
        "dataset": args.dataset, "backbone": pc.name, "method": "patchcore few-shot",
        "settings": {"layers": ["layer2", "layer3"], "num_neighbors": 9,
                     "coreset": "none", "image_size": pc.size},
        "seeds": [m["seed"] for m in manifests], "categories": categories,
        "rows": len(rows), "threads": args.threads,
        "total_seconds": round(time.perf_counter() - started, 1),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
