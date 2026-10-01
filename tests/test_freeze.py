"""The frozen architecture files still match the freeze record."""

from __future__ import annotations

import json

from src.utils.freeze import FROZEN, digests
from src.utils.paths import repo_root

RECORD = repo_root() / "results" / "manifests" / "architecture_freeze.json"


def test_record_covers_every_frozen_file():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert set(record["files"]) == set(FROZEN)


def test_frozen_files_are_unchanged():
    record = json.loads(RECORD.read_text(encoding="utf-8"))["files"]
    changed = [f for f, h in digests(repo_root()).items() if record[f] != h]
    assert not changed, (
        f"architecture files changed after the freeze: {changed}. If this is a "
        "deliberate, agreed change, regenerate the record and say why in the commit."
    )
