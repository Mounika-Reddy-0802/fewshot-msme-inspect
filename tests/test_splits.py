"""Tests for reading Member B's split manifests.

She owns generation (`scripts/generate_splits.py`); these cover the reader that
turns her format into what the evaluation pipeline consumes, plus the two
properties her splits must have for the results to mean anything.
"""

import json

import pytest

from src.data.splits import (
    categories,
    load_manifest,
    resolve_split,
    shot_counts,
    split_for,
)


@pytest.fixture
def fake_root(tmp_path):
    """A miniature MVTec tree plus a manifest in her format."""
    ds = tmp_path / "mvtec_ad" / "bottle"
    for i in range(20):
        p = ds / "train" / "good" / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    for i in range(5):
        p = ds / "test" / "good" / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    for i in range(4):
        (ds / "test" / "broken").mkdir(parents=True, exist_ok=True)
        (ds / "test" / "broken" / f"{i:03d}.png").write_bytes(b"")
        (ds / "ground_truth" / "broken").mkdir(parents=True, exist_ok=True)
        (ds / "ground_truth" / "broken" / f"{i:03d}_mask.png").write_bytes(b"")

    order = [f"{i:03d}.png" for i in (7, 3, 11, 0, 15, 2, 9, 18, 4, 12)]
    manifest = {
        "dataset": "mvtec_ad",
        "seed": 42,
        "shot_counts": [0, 1, 5, 10],
        "categories": {
            "bottle": {
                "test": (
                    [f"mvtec_ad/bottle/test/good/{i:03d}.png" for i in range(5)]
                    + [f"mvtec_ad/bottle/test/broken/{i:03d}.png" for i in range(4)]
                ),
                "shots": {
                    "0": [],
                    "1": [f"mvtec_ad/bottle/train/good/{order[0]}"],
                    "5": [f"mvtec_ad/bottle/train/good/{n}" for n in order[:5]],
                    "10": [f"mvtec_ad/bottle/train/good/{n}" for n in order],
                },
                "num_train_good_available": 20,
            }
        },
    }
    path = tmp_path / "mvtec_ad.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, path


def test_loads_her_format(fake_root):
    _, path = fake_root
    manifest = load_manifest(path)
    assert manifest["seed"] == 42
    assert categories(manifest) == ["bottle"]
    assert shot_counts(manifest) == [0, 1, 5, 10]


def test_rejects_a_manifest_missing_keys(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"dataset": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_manifest(bad)


def test_labels_are_derived_from_the_directory(fake_root):
    root, path = fake_root
    split = split_for(load_manifest(path), "bottle", 5, root)
    assert split["counts"]["test_total"] == 9
    assert split["counts"]["test_normal"] == 5
    assert split["counts"]["test_anomalous"] == 4


def test_masks_are_found_for_anomalies_only(fake_root):
    root, path = fake_root
    split = split_for(load_manifest(path), "bottle", 5, root)
    assert split["counts"]["test_with_mask"] == 4
    assert all(t["mask"] is None for t in split["test"] if t["label"] == 0)
    assert all(t["mask"] for t in split["test"] if t["label"] == 1)


def test_her_shots_are_nested(fake_root):
    """1-shot must be a subset of 5-shot, which must be a subset of 10-shot, or
    a gain from more shots is confounded with a luckier draw."""
    root, path = fake_root
    manifest = load_manifest(path)
    one, five, ten = (
        set(split_for(manifest, "bottle", k, root)["support"]) for k in (1, 5, 10)
    )
    assert one < five < ten


def test_test_split_does_not_depend_on_shots(fake_root):
    root, path = fake_root
    manifest = load_manifest(path)
    a = split_for(manifest, "bottle", 1, root)["test"]
    b = split_for(manifest, "bottle", 10, root)["test"]
    assert a == b


def test_paths_are_category_relative(fake_root):
    """Her manifest stores data-root-relative paths; the pipeline resolves
    against the category directory, so the prefix must be stripped."""
    root, path = fake_root
    split = split_for(load_manifest(path), "bottle", 5, root)
    assert all(p.startswith("train/") for p in split["support"])
    assert all(t["image"].startswith("test/") for t in split["test"])


def test_zero_shot_has_no_support(fake_root):
    root, path = fake_root
    assert split_for(load_manifest(path), "bottle", 0, root)["support"] == []


def test_unknown_category_is_an_error(fake_root):
    root, path = fake_root
    with pytest.raises(KeyError, match="cable"):
        split_for(load_manifest(path), "cable", 5, root)


def test_unknown_shot_count_is_an_error(fake_root):
    root, path = fake_root
    with pytest.raises(KeyError, match="3-shot"):
        split_for(load_manifest(path), "bottle", 3, root)


def test_resolve_gives_existing_files(fake_root):
    root, path = fake_root
    split = split_for(load_manifest(path), "bottle", 5, root)
    support, test = resolve_split(split, root)
    assert len(support) == 5 and all(p.exists() for p in support)
    assert len(test) == 9
    assert all(s.mask.exists() for s in test if s.mask)
