"""Paired comparison of two benchmark runs, written as a committed manifest.

    python scripts/compare_paired.py visa_ViT-B-16-quickgelu visa_ViT-B-32-quickgelu \
        --question "does B/32 match B/16 on VisA" --out visa_backbones.json
    python scripts/compare_paired.py mvtec_ad_ViT-B-16-quickgelu mvtec_ad_ViT-B-16-quickgelu-int8 \
        --question "does accuracy survive int8" --out int8_b16.json

Both runs must score identical (category, shots, seed) splits, so split luck
cancels and the right spread is the paired standard error, not the seed-to-seed
spread. An effect under 2x its paired SE is reported as noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import repo_root, results_root  # noqa: E402

METRICS = ("image_auroc", "pixel_auroc")


def load(name: str) -> dict[tuple, dict]:
    path = results_root() / "benchmark" / f"{name}.csv"
    with path.open(encoding="utf-8") as handle:
        return {
            (r["category"], int(r["shots"]), int(r["seed"])): r
            for r in csv.DictReader(handle)
            if r.get("method", "memory") == "memory"
        }


def compare(a: dict, b: dict, metric: str) -> dict:
    """Per shot count: means, paired delta (a - b), its SE, and category votes."""
    keys = sorted(set(a) & set(b))
    by_shot = defaultdict(list)
    for key in keys:
        by_shot[key[1]].append(key)

    def value(row):
        try:
            v = float(row[metric])
        except (TypeError, ValueError):
            return None
        return None if v != v else v                  # NaN: undefined for this split

    out = {}
    for shots, group in sorted(by_shot.items()):
        # a metric that is undefined for a split (pixel AUROC on NEU-DET, which
        # has no masks) is left out rather than poisoning the mean
        group = [k for k in group if value(a[k]) is not None and value(b[k]) is not None]
        if len(group) < 2:
            continue
        va = [value(a[k]) for k in group]
        vb = [value(b[k]) for k in group]
        deltas = [x - y for x, y in zip(va, vb)]
        se = statistics.stdev(deltas) / math.sqrt(len(deltas))
        delta = statistics.fmean(deltas)

        per_cat = defaultdict(list)
        for k, d in zip(group, deltas):
            per_cat[k[0]].append(d)
        favouring_a = sum(statistics.fmean(v) > 0 for v in per_cat.values())

        out[str(shots)] = {
            "n_pairs": len(group),
            "mean_a": round(statistics.fmean(va), 4),
            "mean_b": round(statistics.fmean(vb), 4),
            "paired_mean_delta": round(delta, 4),
            "paired_standard_error": round(se, 4),
            "effect_over_se": round(abs(delta) / se, 1) if se else None,
            "categories_favouring_a": f"{favouring_a}/{len(per_cat)}",
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", help="benchmark CSV stem, e.g. visa_ViT-B-16-quickgelu")
    parser.add_argument("b")
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", required=True, help="file name under results/manifests/")
    args = parser.parse_args()

    a, b = load(args.a), load(args.b)
    shared = set(a) & set(b)
    if not shared:
        print("the two runs share no (category, shots, seed) splits", file=sys.stderr)
        return 1

    manifest = {
        "question": args.question,
        "design": "paired: identical (category, shots, seed) splits; delta is a - b",
        "a": args.a,
        "b": args.b,
        "seeds": sorted({k[2] for k in shared}),
        "categories": len({k[0] for k in shared}),
        "per_metric": {m: rows for m in METRICS if (rows := compare(a, b, m))},
    }

    for metric, rows in manifest["per_metric"].items():
        for shots, r in rows.items():
            print(f"{metric:<12} {shots:>2}-shot  a {r['mean_a']:.4f}  b {r['mean_b']:.4f}  "
                  f"delta {r['paired_mean_delta']:+.4f} ({r['effect_over_se']}x SE)  "
                  f"favouring a {r['categories_favouring_a']}")

    path = repo_root() / "results" / "manifests" / args.out
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
