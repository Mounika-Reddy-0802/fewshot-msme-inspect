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

Manifest is written to ${results_root}/manifests/splits/<dataset>.json,
with every path stored as posix-style (forward slashes) regardless of OS,
so the manifest is identical whether generated on Windows, Mac or Linux.
This file is tracked in Git (see GIT_RULES.md) even though the rest of
results/ is not.
"""

from __future__ import annotations

# Import the command-line parsing tools used to define and read script options.
import argparse
# Import JSON serialization so generated manifests can be written as structured files.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import data_root, dataset_dir, repo_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

SHOT_COUNTS = (0, 1, 5, 10)
IMAGE_EXTS = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")

# VisA's categories live one level deeper (visa_pytorch/<category>) than
# MVTec's (<category> directly under the dataset root). 
CATEGORY_SUBDIR = {
    "mvtec_ad": "",
    "visa": "visa_pytorch",
    "neu_det": "",
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
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="one manifest per seed; defaults to the seed in base.yaml. "
             "Extra seeds are what give a result error bars -- with a single "
             "draw, a category that dips at 5 shots cannot be told apart from "
             "an unlucky support image.",
    )
    args = parser.parse_args()

    config = load_config("base.yaml")
    default_seed = config["experiment"]["seed"]
    seeds = args.seeds or [default_seed]
    seed_everything(default_seed)

    root = category_root(args.dataset)
    categories = args.categories or discover_categories(root)

    if not categories:
        print(f"no categories found under {root} -- run the download script first")
        return 1

    print(f"seeds {seeds} -- generating splits for {len(categories)} "
          f"{args.dataset} categories: {categories}")

    relpath_root = data_root()  # every path in the manifest is relative to this

    # Manifests are the reproducibility record and must be committed, so they go
    # to the repository rather than to FEWSHOT_RESULTS_ROOT, which usually points
    # outside it.
    out_dir = repo_root() / "results" / "manifests" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    import random

    for seed in seeds:
        rng = random.Random(seed)      # fresh per seed, so the draws are independent
        manifest = {
            "dataset": args.dataset,
            "seed": seed,
            "shot_counts": list(SHOT_COUNTS),
            "categories": {},
        }
        for category in categories:
            manifest["categories"][category] = build_category_split(
                root / category, relpath_root, rng
            )

        # the default seed keeps the original filename, so existing references
        # and Member B's workflow are unaffected
        name = (
            f"{args.dataset}.json" if seed == default_seed
            else f"{args.dataset}_seed{seed}.json"
        )
        (out_dir / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        first = categories[0]
        print(f"  seed {seed:<4} -> {name:<28}"
              f"{manifest['categories'][first]['num_train_good_available']} "
              f"train/good in {first}, "
              f"{len(manifest['categories'][first]['test'])} test")

    print(f"\n{len(seeds)} manifest(s) written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
