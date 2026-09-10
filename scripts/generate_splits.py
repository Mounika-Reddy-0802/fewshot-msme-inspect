"""Generate seeded 0/1/5/10-shot splits for MVTec AD or VisA (T2.2).

Reads whatever categories are already downloaded for the chosen dataset
and writes one manifest recording, for every category and shot count,
exactly which images are the "shot" references and which images make up
the fixed test set. The seed comes from configs/base.yaml, so this is
safe to re-run at any point and always reproduces identical splits for
categories it has already covered.

    python scripts/generate_splits.py --dataset mvtec_ad
    python scripts/generate_splits.py --dataset visa
    python scripts/generate_splits.py --dataset visa --categories candle cashew

Manifest is written to ${results_root}/manifests/splits/<dataset>.json.
This file is tracked in Git (see GIT_RULES.md) even though the rest of
results/ is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import data_root, dataset_dir, results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

SHOT_COUNTS = (0, 1, 5, 10)
IMAGE_EXTS = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")

# VisA's categories live one level deeper (visa_pytorch/<category>) than
# MVTec's (<category> directly under the dataset root).
CATEGORY_SUBDIR = {
    "mvtec_ad": "",
    "visa": "visa_pytorch",
}


def category_root(dataset: str) -> Path:
    root = dataset_dir(dataset)
    subdir = CATEGORY_SUBDIR.get(dataset, "")
    return (root / subdir) if subdir else root


def _dedupe_case_insensitive(paths: list[Path]) -> list[Path]:
    """Windows filesystems are case-insensitive, so globbing both '*.png'
    and '*.PNG' matches the same files twice. Dedupe by resolved, lowered
    path before returning."""
    seen: dict[str, Path] = {}
    for p in paths:
        seen[str(p.resolve()).lower()] = p
    return sorted(seen.values())


def glob_images(folder: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in IMAGE_EXTS:
        found.extend(folder.glob(pattern))
    return _dedupe_case_insensitive(found)


def rglob_images(folder: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in IMAGE_EXTS:
        found.extend(folder.rglob(pattern))
    return _dedupe_case_insensitive(found)


def discover_categories(root: Path) -> list[str]:
    """Any category folder with a populated train/good is considered present."""
    if not root.is_dir():
        return []
    return [
        p.name for p in sorted(root.iterdir())
        if p.is_dir() and (p / "train" / "good").is_dir()
    ]


def build_category_split(category_dir: Path, relpath_root: Path, rng) -> dict:
    good_images = glob_images(category_dir / "train" / "good")
    good_images = [p.relative_to(relpath_root).as_posix() for p in good_images]

    test_images = rglob_images(category_dir / "test")
    test_images = [p.relative_to(relpath_root).as_posix() for p in test_images]

    rng.shuffle(good_images)

    shots = {str(k): good_images[:k] for k in SHOT_COUNTS}

    return {
        "test": test_images,
        "shots": shots,
        "num_train_good_available": len(good_images),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mvtec_ad", choices=list(CATEGORY_SUBDIR))
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="defaults to every category currently present for this dataset",
    )
    args = parser.parse_args()

    config = load_config("base.yaml")
    seed = config["experiment"]["seed"]
    seed_everything(seed)

    import random
    rng = random.Random(seed)

    root = category_root(args.dataset)
    categories = args.categories or discover_categories(root)

    if not categories:
        print(f"no categories found under {root} -- run the download script first")
        return 1

    print(f"seed {seed} -- generating splits for {len(categories)} {args.dataset} categories: {categories}")

    manifest = {
        "dataset": args.dataset,
        "seed": seed,
        "shot_counts": list(SHOT_COUNTS),
        "categories": {},
    }

    relpath_root = data_root()  # every path in the manifest is relative to this

    for category in categories:
        category_dir = root / category
        manifest["categories"][category] = build_category_split(category_dir, relpath_root, rng)
        n_test = len(manifest["categories"][category]["test"])
        n_good = manifest["categories"][category]["num_train_good_available"]
        print(f"  {category:<12} {n_good} train/good available, {n_test} test images")

    out_dir = results_root() / "manifests" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
