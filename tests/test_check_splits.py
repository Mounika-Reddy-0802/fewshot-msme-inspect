"""Test that scripts/check_splits.py passes against the real committed manifests.

This is a regression guard: if anyone's generator change breaks nesting,
introduces a support/test overlap, or (for multi-seed datasets) causes two
seeds to draw the same images, this test catches it on the next pytest run
rather than silently shipping bad splits.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_check_splits_passes_on_committed_manifests():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "check_splits.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"check_splits.py failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_check_splits_ignores_the_results_root(tmp_path):
    """The committed manifests live in the repo. Pointing FEWSHOT_RESULTS_ROOT
    elsewhere -- the documented setup -- must not make the check find nothing."""
    import os

    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "FEWSHOT_RESULTS_ROOT": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "check_splits.py")],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "neu_det" in result.stdout and "mvtec_ad" in result.stdout
