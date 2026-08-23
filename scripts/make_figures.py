"""Produce the result figures (T3.3).

    python scripts/make_figures.py

Reads the benchmark and profiling CSVs and writes:

* ``shots_vs_accuracy`` -- image AUROC against the number of labelled shots,
  averaged over categories, with a band across seeds. Shows what few-shot buys.
* ``accuracy_latency_frontier`` -- the headline figure. Image AUROC against CPU
  milliseconds per image. A point is only interesting if nothing sits above and
  to its left, so the Pareto front is drawn explicitly.

Every figure gets a manifest recording exactly which rows produced it. Figure
manifests are committed even though the images are not, so a figure in the report
can always be traced back to the run that made it.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import repo_root, results_root  # noqa: E402

# colourblind-safe, and distinguishable in greyscale print
PALETTE = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_benchmark(benchmark_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(benchmark_dir.glob("*.csv")):
        if path.name.endswith(".meta.json"):
            continue
        for row in read_csv(path):
            row["shots"] = int(row["shots"])
            row["seed"] = int(row["seed"])
            for key in ("image_auroc", "pixel_auroc", "image_f1_max", "pixel_f1_max"):
                row[key] = float(row[key]) if row.get(key) not in (None, "", "None") else None
            rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else float("nan")


def pareto_front(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Points as (cost, accuracy, label). Keeps those nothing dominates: no other
    point is both cheaper and more accurate."""
    front = []
    for cost, accuracy, label in sorted(points):
        if not front or accuracy > front[-1][1]:
            front.append((cost, accuracy, label))
    return front


def figure_shots_vs_accuracy(rows, out_dir, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_backbone = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_backbone[row["backbone"]][row["shots"]].append(row["image_auroc"])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    used = []
    for i, (backbone, by_shot) in enumerate(sorted(by_backbone.items())):
        shots = sorted(by_shot)
        # each (category, seed) is one sample; band is the spread across them
        means = [mean(by_shot[s]) for s in shots]
        los = [min(by_shot[s]) for s in shots]
        his = [max(by_shot[s]) for s in shots]
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(shots, means, "o-", color=colour, label=backbone, linewidth=2, markersize=6)
        ax.fill_between(shots, los, his, color=colour, alpha=0.12)
        used.append({"backbone": backbone, "shots": shots, "mean_image_auroc": means})

    ax.set_xlabel("labelled normal images (shots)")
    ax.set_ylabel("image AUROC")
    ax.set_title("Few-shot accuracy on MVTec AD\n(mean over 15 categories, band = min-max across categories x seeds)",
                 fontsize=10)
    ax.set_xticks(sorted({r["shots"] for r in rows}))
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    paths = _save(fig, out_dir, "shots_vs_accuracy", dpi)
    plt.close(fig)
    return paths, used


def figure_frontier(rows, profiles, out_dir, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latency = {
        p["backbone"]: float(p["latency_bs1_per_image_ms"])
        for p in profiles
        if p.get("latency_bs1_per_image_ms")
    }
    if not latency:
        return None, []

    grouped = defaultdict(list)
    for row in rows:
        if row["backbone"] in latency:
            grouped[(row["backbone"], row["shots"])].append(row["image_auroc"])

    points, used = [], []
    for (backbone, shots), values in grouped.items():
        cost, accuracy = latency[backbone], mean(values)
        points.append((cost, accuracy, f"{backbone} {shots}-shot"))
        used.append({"backbone": backbone, "shots": shots,
                     "ms_per_image": cost, "image_auroc": accuracy})

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    backbones = sorted({b for b, _ in grouped})
    for i, backbone in enumerate(backbones):
        subset = [(c, a, l) for c, a, l in points if l.startswith(backbone)]
        subset.sort(key=lambda t: t[1])
        ax.scatter([c for c, _, _ in subset], [a for _, a, _ in subset],
                   s=70, color=PALETTE[i % len(PALETTE)], label=backbone,
                   zorder=3, edgecolor="white", linewidth=1)
        for cost, accuracy, label in subset:
            ax.annotate(label.split()[-1], (cost, accuracy), fontsize=7.5,
                        xytext=(5, -3), textcoords="offset points", color="#444")

    front = pareto_front(points)
    if len(front) > 1:
        ax.plot([c for c, _, _ in front], [a for _, a, _ in front],
                "--", color="#666", linewidth=1.2, zorder=2,
                label="Pareto front")

    ax.set_xlabel("CPU latency (ms per image, batch size 1)")
    ax.set_ylabel("image AUROC")
    ax.set_title("Accuracy against CPU cost\n(MVTec AD, frozen backbone + prototype memory, no GPU)",
                 fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()

    paths = _save(fig, out_dir, "accuracy_latency_frontier", dpi)
    plt.close(fig)
    return paths, used


def _save(fig, out_dir: Path, name: str, dpi: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(str(path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load_benchmark(results_root() / "benchmark")
    if not rows:
        print("no benchmark rows; run scripts/run_benchmark.py first", file=sys.stderr)
        return 1
    profiles = read_csv(results_root() / "profiling" / "backbones.csv")

    out_dir = Path(args.out) if args.out else results_root() / "figures"
    manifest = {
        "source_rows": len(rows),
        "backbones": sorted({r["backbone"] for r in rows}),
        "categories": sorted({r["category"] for r in rows}),
        "shots": sorted({r["shots"] for r in rows}),
        "seeds": sorted({r["seed"] for r in rows}),
        "figures": {},
    }

    paths, used = figure_shots_vs_accuracy(rows, out_dir, args.dpi)
    manifest["figures"]["shots_vs_accuracy"] = {"files": paths, "data": used}
    print(f"shots_vs_accuracy       -> {paths[0]}")

    paths, used = figure_frontier(rows, profiles, out_dir, args.dpi)
    if paths:
        manifest["figures"]["accuracy_latency_frontier"] = {"files": paths, "data": used}
        print(f"accuracy_latency_frontier -> {paths[0]}")
    else:
        print("accuracy_latency_frontier: skipped, no profiling data "
              "(run scripts/run_profiling.py)")

    # manifests are committed even though the images are not
    manifest_path = repo_root() / "results" / "manifests" / "figures.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest                -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
