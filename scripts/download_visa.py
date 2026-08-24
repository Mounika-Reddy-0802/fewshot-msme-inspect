"""Fetch and prepare the full VisA dataset (all twelve categories).

Unlike MVTec AD, VisA ships as a single archive organised by CSV split
files rather than folders -- reorganising that by hand is easy to get
subtly wrong. Instead this uses anomalib's own Visa datamodule, which
downloads the official archive from its AWS host, extracts it, and
reorganises each category into visa_pytorch/<category>/ with a
train/good, test/good, test/bad, ground_truth/bad layout -- the same
shape as MVTec AD's train/good + test/<split> layout, so
generate_splits.py and build_feature_bank.py need no rework beyond
pointing at the new dataset.

    python scripts/download_visa.py
    python scripts/download_visa.py --categories candle capsules

The ~2 GB archive downloads once; every category after the first just
reuses the cached download and only redoes the reorganisation step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import dataset_dir  # noqa: E402

CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES,
        help="defaults to all twelve categories",
    )
    parser.add_argument("--root", default=None, help="defaults to <data root>/visa")
    args = parser.parse_args()

    from anomalib.data import Visa

    root = Path(args.root) if args.root else dataset_dir("visa")
    root.mkdir(parents=True, exist_ok=True)

    print(f"target categories: {len(args.categories)} -> {root}\n")
    for category in args.categories:
        processed = root / "visa_pytorch" / category / "train" / "good"
        if processed.is_dir():
            print(f"{category:<12} already present, skipping")
            continue

        print(f"{category:<12} downloading / preparing ...")
        datamodule = Visa(root=str(root), category=category)
        datamodule.prepare_data()

        for split in ("train/good", "test/good", "test/bad"):
            path = root / "visa_pytorch" / category / split
            count = len(list(path.rglob("*.JPG"))) + len(list(path.rglob("*.png"))) if path.exists() else 0
            print(f"  {split:<14} {count} images")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
