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
