"""Fetch a single MVTec AD category.

Week 1 only needs one category to validate the environment, and the official
archive is a 5 GB tarball of all fifteen. This pulls just the category we ask
for from a mirror that keeps the original folder layout
(train/good, test/<defect>, ground_truth/<defect>).

    python scripts/download_mvtec_category.py --category bottle

Full-dataset downloads and the split generators are Member B's; this script
exists so the PatchCore validation in docs/ can be reproduced.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle", choices=CATEGORIES)
    parser.add_argument("--root", default=None, help="defaults to <data root>/mvtec_ad")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    root = Path(args.root) if args.root else dataset_dir("mvtec_ad")
    root.mkdir(parents=True, exist_ok=True)

    if (root / args.category / "train" / "good").is_dir():
        print(f"{args.category} already present at {root / args.category}")
        return 0

    print(f"downloading {args.category} from {MIRROR} into {root}")
    snapshot_download(
        repo_id=MIRROR,
        repo_type="dataset",
        allow_patterns=[f"{args.category}/**", "license.txt", "readme.txt"],
        local_dir=str(root),
    )

    category_dir = root / args.category
    for split in ("train/good", "test", "ground_truth"):
        path = category_dir / split
        count = len(list(path.rglob("*.png"))) if path.exists() else 0
        print(f"  {split:<14} {count} png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
