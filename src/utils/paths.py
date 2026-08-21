"""Path resolution.

Nothing in this project hardcodes a machine-specific path. Datasets and result
artifacts live outside the repo (they are large and gitignored), so their roots
come from environment variables with sensible in-repo defaults:

    FEWSHOT_DATA_ROOT      where MVTec AD / VisA / NEU-DET are unpacked
    FEWSHOT_RESULTS_ROOT   where runs, logs and figures are written
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKERS = (".git", "requirements.txt")


def repo_root() -> Path:
    """Walk up from this file until a repository marker shows up."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if any((parent / marker).exists() for marker in _MARKERS):
            return parent
    return here.parents[2]


def data_root() -> Path:
    return Path(os.environ.get("FEWSHOT_DATA_ROOT", repo_root() / "data")).expanduser()


def results_root() -> Path:
    return Path(
        os.environ.get("FEWSHOT_RESULTS_ROOT", repo_root() / "results")
    ).expanduser()


def dataset_dir(name: str) -> Path:
    """Directory for one dataset, e.g. ``dataset_dir("mvtec_ad")``."""
    return data_root() / name


MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)


def find_category_root(
    base: str | Path,
    categories: tuple[str, ...] = MVTEC_CATEGORIES,
    min_matches: int = 3,
    max_depth: int = 4,
) -> Path | None:
    """Locate a dataset root by its category folders rather than by its name.

    Hosted notebooks mount datasets wherever they like -- on Kaggle an attached
    dataset lands under /kaggle/input/<whatever-the-uploader-called-it>. Looking
    for a directory that has several known category folders side by side finds it
    without pinning us to one dataset slug.

    Returns the directory holding the categories, or None if nothing matches.
    """
    base = Path(base)
    if not base.is_dir():
        return None

    wanted = set(categories)
    base_depth = len(base.resolve().parts)

    for root, dirs, _ in os.walk(base):
        if len(Path(root).resolve().parts) - base_depth >= max_depth:
            dirs[:] = []
            continue
        if len(wanted & set(dirs)) >= min_matches:
            return Path(root)
    return None
