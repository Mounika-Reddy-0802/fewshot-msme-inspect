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
