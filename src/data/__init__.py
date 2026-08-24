"""Dataset loading and split reading.

Member B owns dataset acquisition, split generation and feature banks
(`scripts/download_*.py`, `scripts/generate_splits.py`,
`scripts/build_feature_bank.py`). This package holds the MVTec directory
conventions and a reader for her manifests, so the evaluation pipeline consumes
her splits rather than defining its own.
"""

from src.data.mvtec import (
    CATEGORIES,
    OFFICIAL_COUNTS,
    available_categories,
    complete_categories,
    list_test_samples,
    list_train_normals,
    verify_category,
)
from src.data.splits import (
    Sample,
    categories,
    load_manifest,
    manifest_path,
    resolve_split,
    shot_counts,
    split_for,
)

__all__ = [
    "CATEGORIES",
    "OFFICIAL_COUNTS",
    "available_categories",
    "complete_categories",
    "list_test_samples",
    "list_train_normals",
    "verify_category",
    "Sample",
    "categories",
    "load_manifest",
    "manifest_path",
    "resolve_split",
    "shot_counts",
    "split_for",
]
