"""Fetch the full MVTec AD dataset (all fifteen categories).

Week 1 validated the environment against a single category (`bottle`); this
script performs the actual Workstream 2 corpus-acquisition task (T2.1) —
downloading every category. Categories already present locally are skipped,
so re-running this script is safe and resumable.

    python scripts/download_mvtec_full.py
    python scripts/download_mvtec_full.py --categories bottle cable

Uses the same mirror as scripts/download_mvtec_category.py
(foersben/mvtec-ad), which keeps the original folder layout
(train/good, test/<defect>, ground_truth/<defect>).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import dataset_dir  # noqa: E402

MIRROR = "foersben/mvtec-ad"
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)


def download_category(category: str, root: Path) -> None:
    from huggingface_hub import snapshot_download

    if (root / category / "train" / "good").is_dir():
        print(f"{category:<12} already present, skipping")
        return

    print(f"{category:<12} downloading from {MIRROR} ...")
    snapshot_download(
        repo_id=MIRROR,
        repo_type="dataset",
        allow_patterns=[f"{category}/**", "license.txt", "readme.txt"],
        local_dir=str(root),
    )

    category_dir = root / category
    for split in ("train/good", "test", "ground_truth"):
        path = category_dir / split
        count = len(list(path.rglob("*.png"))) if path.exists() else 0
        print(f"  {split:<14} {count} png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES,
        help="defaults to all fifteen categories",
    )
    parser.add_argument("--root", default=None, help="defaults to <data root>/mvtec_ad")
    args = parser.parse_args()

    root = Path(args.root) if args.root else dataset_dir("mvtec_ad")
    root.mkdir(parents=True, exist_ok=True)

    print(f"target categories: {len(args.categories)} -> {root}\n")
    for category in args.categories:
        download_category(category, root)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
