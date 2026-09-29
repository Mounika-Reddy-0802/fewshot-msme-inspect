"""Architecture freeze, made checkable.

After the freeze the files below must not change: the model is what was
measured. Their hashes are recorded in results/manifests/architecture_freeze.json
and tests/test_freeze.py fails if any differs, so a change cannot slip in
unnoticed -- it has to update the record, which shows up in review.

Line endings are normalised before hashing: the same file checked out on Windows
(CRLF) and on Colab (LF) is the same model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

FROZEN = (
    "src/model/encoder.py",
    "src/model/memory.py",
    "src/model/windows.py",
    "src/model/adapter.py",
    "src/model/export.py",
    "src/eval/scoring.py",
    "configs/base.yaml",
    "configs/prototype.yaml",
    "configs/proposed.yaml",
)


def digest(path: Path) -> str:
    text = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(text).hexdigest()


def digests(repo: Path) -> dict[str, str]:
    return {rel: digest(repo / rel) for rel in FROZEN}
