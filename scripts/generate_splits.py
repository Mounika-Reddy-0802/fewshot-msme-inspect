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
# Import sys so the repository root can be added to Python's import search path.
import sys
# Import Path for platform-independent filesystem path operations.
from pathlib import Path

# Resolve the repository root and place it first so local packages are importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import configuration loading after the path adjustment above.
from src.utils.config import load_config  # noqa: E402
# Import helpers that locate datasets, data-relative paths, and the repository.
from src.utils.paths import data_root, dataset_dir, repo_root  # noqa: E402
# Import the shared seeding helper used to establish reproducible behavior.
from src.utils.seed import seed_everything  # noqa: E402

# Define the supported numbers of training reference images per category.
SHOT_COUNTS = (0, 1, 5, 10)
# List supported image suffix patterns in both common case variants.
IMAGE_EXTS = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")

# VisA's categories live one level deeper (visa_pytorch/<category>) than
# MVTec's (<category> directly under the dataset root). 
CATEGORY_SUBDIR = {
    # Keep MVTec categories directly below the dataset root.
    "mvtec_ad": "",
    # Account for VisA's additional archive directory.
    "visa": "visa_pytorch",
    # Keep NEU-DET categories directly below the dataset root.
    "neu_det": "",
}

def category_root(dataset: str) -> Path:
    # Resolve the configured filesystem root for the selected dataset.
    root = dataset_dir(dataset)
    # Look up the optional nested directory for this dataset name.
    subdir = CATEGORY_SUBDIR.get(dataset, "")
    # Return the nested directory when present, otherwise return the dataset root.
    return (root / subdir) if subdir else root


def _dedupe_case_insensitive(paths: list[Path]) -> list[Path]:
    """Windows filesystems are case-insensitive, so globbing both '*.png'
    and '*.PNG' matches the same files twice. Dedupe by resolved, lowered
    path before returning."""
    # Store one path for each normalized path key.
    seen: dict[str, Path] = {}
    # Process every discovered path so duplicate spellings collapse together.
    for p in paths:
        # Resolve and lowercase each path to make comparison case-insensitive.
        seen[str(p.resolve()).lower()] = p
    # Sort the retained original paths for deterministic manifest output.
    return sorted(seen.values())


def glob_images(folder: Path) -> list[Path]:
    # Start an accumulator for image files found directly in one folder.
    found: list[Path] = []
    # Search the folder once for each supported filename pattern.
    for pattern in IMAGE_EXTS:
        # Append matching files without descending into subdirectories.
        found.extend(folder.glob(pattern))
    # Remove case-variant duplicates before returning the direct matches.
    return _dedupe_case_insensitive(found)


def rglob_images(folder: Path) -> list[Path]:
    # Start an accumulator for image files found throughout a folder tree.
    found: list[Path] = []
    # Search every descendant directory for each supported image pattern.
    for pattern in IMAGE_EXTS:
        # Append matching files from all nested directories.
        found.extend(folder.rglob(pattern))
    # Normalize and sort recursive matches for stable output.
    return _dedupe_case_insensitive(found)


def discover_categories(root: Path) -> list[str]:
    """Any category folder with a populated train/good is considered present."""
    # Return no categories when the expected dataset root is absent.
    if not root.is_dir():
        return []
    # Inspect each direct child in name order to make discovery deterministic.
    return [
        # Keep only directories that contain the expected train/good image area.
        p.name for p in sorted(root.iterdir())
        # Require both a directory entry and its train/good subdirectory.
        if p.is_dir() and (p / "train" / "good").is_dir()
    ]


def build_category_split(category_dir: Path, relpath_root: Path, rng) -> dict:
    # Collect training-good images directly from the category's train folder.
    good_images = glob_images(category_dir / "train" / "good")
    # Convert training paths to data-root-relative POSIX strings.
    good_images = [p.relative_to(relpath_root).as_posix() for p in good_images]
    # Recursively collect every test image under the category's test directory.
    test_images = rglob_images(category_dir / "test")
    # Convert test paths to the same stable data-root-relative format.
    test_images = [p.relative_to(relpath_root).as_posix() for p in test_images]

    # Shuffle the available good images using the caller's seeded generator.
    rng.shuffle(good_images)

    # Build cumulative shot lists from the shuffled training images.
    shots = {str(k): good_images[:k] for k in SHOT_COUNTS}

    # Return the test paths, nested shot lists, and available training count.
    return {
        "test": test_images,
        "shots": shots,
        "num_train_good_available": len(good_images),
    }


def main() -> int:
    # Create the command-line parser using this module's documentation string.
    parser = argparse.ArgumentParser(description=__doc__)
    # Define the dataset option and restrict it to the supported layouts.
    parser.add_argument("--dataset", default="mvtec_ad", choices=list(CATEGORY_SUBDIR))
    # Define an optional list that limits generation to named categories.
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="defaults to every category currently present for this dataset",
    )
    # Define optional integer seeds so each seed can produce its own manifest.
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="one manifest per seed; defaults to the seed in base.yaml. "
             "Extra seeds are what give a result error bars -- with a single "
             "draw, a category that dips at 5 shots cannot be told apart from "
             "an unlucky support image.",
    )
    # Parse command-line values into a namespace for the remainder of main.
    args = parser.parse_args()

    # Load the shared base configuration used by the experiment scripts.
    config = load_config("base.yaml")
    # Read the configured default experiment seed.
    default_seed = config["experiment"]["seed"]
    # Use requested seeds when supplied, otherwise use the configured seed.
    seeds = args.seeds or [default_seed]
    # Initialize shared randomness with the configured experiment seed.
    seed_everything(default_seed)

    # Resolve the category-bearing root for the selected dataset.
    root = category_root(args.dataset)
    # Prefer explicitly requested categories and otherwise discover available ones.
    categories = args.categories or discover_categories(root)

    # Stop with a nonzero status when the selected dataset has no categories.
    if not categories:
        print(f"no categories found under {root} -- run the download script first")
        return 1

    # Report the selected seeds, dataset, and category count before generation.
    print(f"seeds {seeds} -- generating splits for {len(categories)} "
          f"{args.dataset} categories: {categories}")

    # Use the configured data root as the base for all manifest paths.
    relpath_root = data_root()  # every path in the manifest is relative to this

    # Locate the repository-controlled directory for generated split manifests.
    # Manifests are the reproducibility record and must be committed, so they go
    # to the repository rather than to FEWSHOT_RESULTS_ROOT, which usually points
    # outside it.
    out_dir = repo_root() / "results" / "manifests" / "splits"
    # Create the output directory and any missing parents before writing files.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import the standard random module only when generation is needed.
    import random

    # Generate one independent manifest for each requested seed.
    for seed in seeds:
        # Create a fresh deterministic generator for this seed.
        rng = random.Random(seed)      # fresh per seed, so the draws are independent
        # Initialize the manifest metadata shared by every category in this seed.
        manifest = {
            "dataset": args.dataset,
            "seed": seed,
            "shot_counts": list(SHOT_COUNTS),
            "categories": {},
        }
        # Add a split block for each selected category.
        for category in categories:
            # Build and store the category split using the current seeded generator.
            manifest["categories"][category] = build_category_split(
                root / category, relpath_root, rng
            )

        # the default seed keeps the original filename, so existing references
        # and Member B's workflow are unaffected
        name = (
            f"{args.dataset}.json" if seed == default_seed
            else f"{args.dataset}_seed{seed}.json"
        )
        # Serialize this seed's manifest to its selected output filename.
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
