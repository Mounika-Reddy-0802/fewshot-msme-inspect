"""Turn the benchmark CSV into a table you can paste into a report.

    python scripts/summarise_results.py                     # markdown to stdout
    python scripts/summarise_results.py --metric pixel_auroc
    python scripts/summarise_results.py --dataset visa --backbone ViT-B-32-quickgelu
    python scripts/summarise_results.py --out docs/results_table.md

Aggregates over seeds (mean +- spread), one row per category, one column per
shot count. Cells where a metric is undefined -- pixel metrics for a category
with no masks, AUROC with only one class present -- print as "n/a" rather than
silently becoming 0, which would drag an average down without anyone noticing.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import results_root  # noqa: E402

METRICS = ("image_auroc", "pixel_auroc", "image_f1_max", "pixel_f1_max")
DATASET_NAMES = {"mvtec_ad": "MVTec AD", "visa": "VisA", "neu_det": "NEU-DET"}


def load_rows(benchmark_dir: Path, dataset: str) -> list[dict]:
    """Rows for one dataset; run_benchmark.py writes <dataset>_<backbone>.csv."""
    rows = []
    for path in sorted(benchmark_dir.glob(f"{dataset}_*.csv")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def as_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number      # drop NaN


def cell(values: list[float | None], show_spread: bool) -> str:
    clean = [v for v in values if v is not None]
    if not clean:
        return "n/a"
    mean = statistics.fmean(clean)
    if not show_spread or len(clean) < 2:
        return f"{mean:.3f}"
    return f"{mean:.3f} +/-{statistics.pstdev(clean):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="image_auroc", choices=METRICS)
    parser.add_argument("--dataset", default="mvtec_ad")
    parser.add_argument("--backbone", default=None, help="filter to one backbone")
    parser.add_argument("--method", default=None, help="filter to one scoring head")
    parser.add_argument("--spread", action="store_true", help="show +/- std over seeds")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load_rows(results_root() / "benchmark", args.dataset)
    if not rows:
        print(f"no benchmark CSV for {args.dataset}; run scripts/run_benchmark.py first",
              file=sys.stderr)
        return 1
    if args.backbone:
        rows = [r for r in rows if r.get("backbone") == args.backbone]
    if args.method:
        rows = [r for r in rows if r.get("method") == args.method]

    shots = sorted({int(r["shots"]) for r in rows})
    categories = sorted({r["category"] for r in rows})
    backbones = sorted({r["backbone"] for r in rows})

    grid = defaultdict(list)
    for row in rows:
        grid[(row["category"], int(row["shots"]))].append(as_float(row[args.metric]))

    lines = [
        f"### {args.metric.replace('_', ' ')} on {DATASET_NAMES.get(args.dataset, args.dataset)}",
        "",
        f"Backbone: {', '.join(backbones)}. "
        f"Mean over {len({r['seed'] for r in rows})} seeds"
        + (", +/- population std." if args.spread else ".")
        + f" {len(categories)} categories.",
        "",
        "| category | " + " | ".join(f"{k}-shot" for k in shots) + " |",
        "|" + "---|" * (len(shots) + 1),
    ]

    for category in categories:
        cells = [cell(grid[(category, k)], args.spread) for k in shots]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    means = []
    for k in shots:
        pooled = [v for c in categories for v in grid[(c, k)]]
        means.append(cell(pooled, args.spread))
    lines.append("| **mean** | " + " | ".join(f"**{m}**" for m in means) + " |")

    table = "\n".join(lines)
    print(table)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")
        print(f"\nwritten to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
