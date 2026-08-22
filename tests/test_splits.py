"""The few-shot splits are the reproducibility claim, so they get real tests.

Everything here runs on a synthetic MVTec tree so the suite stays fast and does
not depend on anything being downloaded.
"""

import json

import pytest

from src.data.mvtec import available_categories, list_test_samples, list_train_normals
from src.data.splits import (
    SCHEMA_VERSION,
    load_split,
    make_split,
    resolve_split,
    write_split,
)


@pytest.fixture
def fake_root(tmp_path):
    """One category laid out exactly like MVTec AD: 20 train normals, 5 normal
    test images, 4 defective ones with masks."""
    base = tmp_path / "bottle"
    for i in range(20):
        p = base / "train" / "good" / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    for i in range(5):
        p = base / "test" / "good" / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    for i in range(4):
        img = base / "test" / "broken" / f"{i:03d}.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"")
        mask = base / "ground_truth" / "broken" / f"{i:03d}_mask.png"
        mask.parent.mkdir(parents=True, exist_ok=True)
        mask.write_bytes(b"")
    return tmp_path


def test_loader_finds_the_layout(fake_root):
    assert len(list_train_normals(fake_root, "bottle")) == 20
    samples = list_test_samples(fake_root, "bottle")
    assert len(samples) == 9
    assert sum(s.label for s in samples) == 4
    assert all(s.mask is not None for s in samples if s.label == 1)
    assert all(s.mask is None for s in samples if s.label == 0)


def test_available_categories_only_lists_downloaded(fake_root):
    assert available_categories(fake_root) == ["bottle"]


def test_shots_are_nested(fake_root):
    """1-shot must be a subset of 5-shot, which must be a subset of 10-shot --
    otherwise a gain from adding shots is confounded with a luckier draw."""
    one, five, ten = (
        set(make_split(fake_root, "bottle", k, seed=0)["support"]) for k in (1, 5, 10)
    )
    assert one < five < ten
    assert len(one) == 1 and len(five) == 5 and len(ten) == 10


def test_same_seed_same_split(fake_root):
    a = make_split(fake_root, "bottle", 5, seed=7)
    b = make_split(fake_root, "bottle", 5, seed=7)
    assert a["support"] == b["support"]


def test_different_seed_different_split(fake_root):
    a = make_split(fake_root, "bottle", 5, seed=0)
    b = make_split(fake_root, "bottle", 5, seed=1)
    assert a["support"] != b["support"]


def test_test_split_does_not_depend_on_shots(fake_root):
    """Accuracy at 1 and 10 shots has to be measured on identical queries."""
    a = make_split(fake_root, "bottle", 1, seed=0)["test"]
    b = make_split(fake_root, "bottle", 10, seed=0)["test"]
    assert a == b


def test_zero_shot_has_no_support(fake_root):
    manifest = make_split(fake_root, "bottle", 0, seed=0)
    assert manifest["support"] == []
    assert manifest["counts"]["test_total"] == 9


def test_asking_for_too_many_shots_is_an_error(fake_root):
    with pytest.raises(ValueError, match="only has 20"):
        make_split(fake_root, "bottle", 50, seed=0)


def test_paths_are_relative_so_manifests_are_portable(fake_root):
    manifest = make_split(fake_root, "bottle", 5, seed=0)
    assert all(not p.startswith("/") and ":" not in p for p in manifest["support"])
    assert all(t["image"].startswith("test/") for t in manifest["test"])


def test_round_trip_through_disk(fake_root, tmp_path):
    manifest = make_split(fake_root, "bottle", 5, seed=3)
    path = write_split(manifest, tmp_path / "splits")
    assert path.name == "mvtec_ad_bottle_5shot_seed3.json"

    loaded = load_split(path)
    assert loaded == manifest

    support, test = resolve_split(loaded, fake_root)
    assert len(support) == 5 and all(p.exists() for p in support)
    assert len(test) == 9
    assert all(s.mask.exists() for s in test if s.mask)


def test_schema_mismatch_is_refused(fake_root, tmp_path):
    manifest = make_split(fake_root, "bottle", 1, seed=0)
    manifest["schema_version"] = SCHEMA_VERSION + 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_split(path)
