"""Generate seeded 0/1/5/10-shot splits for MVTec AD (T2.2).

Reads whatever categories are already downloaded under the dataset root and
writes one manifest recording, for every category and shot count, exactly
which images are the "shot" references and which images make up the fixed
test set. The seed comes from configs/base.yaml, so this is safe to re-run
at any point (e.g. after more categories finish downloading) and always
reproduces identical splits for categories it has already covered.

    python scripts/generate_splits.py
    python scripts/generate_splits.py --categories bottle cable

Manifest is written to ${results_root}/manifests/splits/mvtec_ad.json, the
exact path base.yaml's dataset.split_manifest points to. This file is
tracked in Git (see GIT_RULES.md) even though the rest of results/ is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import dataset_dir, results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

SHOT_COUNTS = (0, 1, 5, 10)


def discover_categories(root: Path) -> list[str]:
    """Any category folder with a populated train/good is considered present."""
    if not root.is_dir():
        return []
    found = [
        p.name for p in sorted(root.iterdir())
        if p.is_dir() and (p / "train" / "good").is_dir()
    ]
    return found


def build_category_split(category_dir: Path, rng) -> dict:
    good_images = sorted((category_dir / "train" / "good").glob("*.png"))
    good_images = [str(p.relative_to(category_dir.parents[1])) for p in good_images]

    test_images = sorted((category_dir / "test").rglob("*.png"))
    test_images = [str(p.relative_to(category_dir.parents[1])) for p in test_images]

    rng.shuffle(good_images)

    shots = {}
    for k in SHOT_COUNTS:
        shots[str(k)] = good_images[:k]

    return {
        "test": test_images,
        "shots": shots,
        "num_train_good_available": len(good_images),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="defaults to every category currently present under the data root",
    )
    args = parser.parse_args()

    config = load_config("base.yaml")
    seed = config["experiment"]["seed"]
    seed_everything(seed)

    import random
    rng = random.Random(seed)

    root = dataset_dir("mvtec_ad")
    categories = args.categories or discover_categories(root)

    if not categories:
        print(f"no categories found under {root} -- run the download script first")
        return 1

    print(f"seed {seed} -- generating splits for {len(categories)} categories: {categories}")

    manifest = {
        "dataset": "mvtec_ad",
        "seed": seed,
        "shot_counts": list(SHOT_COUNTS),
        "categories": {},
    }

    for category in categories:
        category_dir = root / category
        manifest["categories"][category] = build_category_split(category_dir, rng)
        n_test = len(manifest["categories"][category]["test"])
        n_good = manifest["categories"][category]["num_train_good_available"]
        print(f"  {category:<12} {n_good} train/good available, {n_test} test images")

    out_dir = results_root() / "manifests" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mvtec_ad.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
