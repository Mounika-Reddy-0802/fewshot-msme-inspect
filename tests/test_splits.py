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


# --- multiple seeds ---------------------------------------------------------

def test_manifest_paths_finds_every_seed(tmp_path):
    """The default seed keeps the plain name; extra seeds are siblings."""
    from src.data.splits import manifest_paths

    (tmp_path / "mvtec_ad.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mvtec_ad_seed1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mvtec_ad_seed2.json").write_text("{}", encoding="utf-8")
    (tmp_path / "visa.json").write_text("{}", encoding="utf-8")

    found = [p.name for p in manifest_paths(tmp_path, "mvtec_ad")]
    assert found == ["mvtec_ad.json", "mvtec_ad_seed1.json", "mvtec_ad_seed2.json"]
    assert "visa.json" not in found


def test_manifest_paths_puts_the_default_seed_first(tmp_path):
    from src.data.splits import manifest_paths

    (tmp_path / "mvtec_ad_seed1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mvtec_ad.json").write_text("{}", encoding="utf-8")
    assert manifest_paths(tmp_path, "mvtec_ad")[0].name == "mvtec_ad.json"


def test_manifest_paths_of_nothing(tmp_path):
    from src.data.splits import manifest_paths

    assert manifest_paths(tmp_path, "mvtec_ad") == []


def test_committed_manifests_cover_several_seeds():
    """A single seed cannot distinguish a real dip from an unlucky draw, so the
    repository must carry more than one."""
    from src.data.splits import load_manifest, manifest_paths
    from src.utils.paths import repo_root

    paths = manifest_paths(repo_root() / "results" / "manifests" / "splits", "mvtec_ad")
    seeds = sorted(load_manifest(p)["seed"] for p in paths)
    assert len(seeds) >= 3, f"only seeds {seeds} committed"
    assert len(set(seeds)) == len(seeds), "duplicate seeds"


def test_each_committed_seed_is_nested_and_distinct():
    from src.data.splits import load_manifest, manifest_paths
    from src.utils.paths import repo_root

    paths = manifest_paths(repo_root() / "results" / "manifests" / "splits", "mvtec_ad")
    draws = set()
    for p in paths:
        manifest = load_manifest(p)
        for block in manifest["categories"].values():
            one, five, ten = (set(block["shots"][k]) for k in ("1", "5", "10"))
            assert one < five < ten, f"{p.name} is not nested"
        draws.add(tuple(manifest["categories"]["bottle"]["shots"]["5"]))
    assert len(draws) == len(paths), "seeds produced identical draws"


# --- mask naming conventions ------------------------------------------------

def test_finds_mvtec_style_masks(tmp_path):
    """MVTec: test/broken/000.png -> ground_truth/broken/000_mask.png"""
    from src.data.splits import _mask_for

    gt = tmp_path / "ds" / "cat" / "ground_truth" / "broken"
    gt.mkdir(parents=True)
    (gt / "000_mask.png").write_bytes(b"")
    assert _mask_for("ds/cat/test/broken/000.png", tmp_path) is not None


def test_finds_visa_style_masks(tmp_path):
    """VisA: test/bad/000.JPG -> ground_truth/bad/000.png -- a different
    convention entirely. Guessing wrong is silent: the image gets an all-zero
    mask and its pixel metrics become meaningless without any error."""
    from src.data.splits import _mask_for

    gt = tmp_path / "ds" / "cat" / "ground_truth" / "bad"
    gt.mkdir(parents=True)
    (gt / "000.png").write_bytes(b"")
    assert _mask_for("ds/cat/test/bad/000.JPG", tmp_path) is not None


def test_missing_mask_returns_none(tmp_path):
    from src.data.splits import _mask_for

    (tmp_path / "ds" / "cat" / "ground_truth" / "bad").mkdir(parents=True)
    assert _mask_for("ds/cat/test/bad/000.JPG", tmp_path) is None
