"""Generate the seeded few-shot split manifests.

    python scripts/make_splits.py                      # all downloaded categories
    python scripts/make_splits.py --categories bottle cable
    python scripts/make_splits.py --shots 0 1 5 10 --seeds 0 1 2

Manifests go to results/manifests/splits/ inside the repo -- not to
FEWSHOT_RESULTS_ROOT -- because they are the reproducibility record and have to
be committed. Everything else under results/ is gitignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mvtec import CATEGORIES, available_categories  # noqa: E402
from src.data.splits import SHOT_COUNTS, make_split, write_split  # noqa: E402
from src.utils.paths import dataset_dir, repo_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mvtec_ad")
    parser.add_argument("--root", default=None, help="defaults to <data root>/<dataset>")
    parser.add_argument("--categories", nargs="+", default=None, choices=[*CATEGORIES, "all"])
    parser.add_argument("--shots", nargs="+", type=int, default=list(SHOT_COUNTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root) if args.root else dataset_dir(args.dataset)
    out_dir = Path(args.out) if args.out else repo_root() / "results" / "manifests" / "splits"

    categories = args.categories
    if not categories or categories == ["all"]:
        categories = available_categories(root)
    if not categories:
        print(f"no categories found under {root}", file=sys.stderr)
        return 1

    written = 0
    for category in categories:
        for shots in args.shots:
            for seed in args.seeds:
                manifest = make_split(root, category, shots, seed, dataset=args.dataset)
                write_split(manifest, out_dir)
                written += 1
        counts = manifest["counts"]
        print(
            f"{category:<12} normals={counts['train_normals_available']:<4} "
            f"test={counts['test_total']:<4} "
            f"(normal {counts['test_normal']}, anomalous {counts['test_anomalous']}, "
            f"masks {counts['test_with_mask']})"
        )

    print(f"\n{written} manifests -> {out_dir}")
    print(f"categories: {len(categories)}  shots: {args.shots}  seeds: {args.seeds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
