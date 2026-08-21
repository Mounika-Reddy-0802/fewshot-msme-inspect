"""Roots come from the environment, never from a hardcoded path."""

from pathlib import Path

from src.utils.paths import (
    data_root,
    dataset_dir,
    find_category_root,
    repo_root,
    results_root,
)


def test_repo_root_holds_the_configs():
    assert (repo_root() / "configs" / "base.yaml").exists()


def test_data_root_follows_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("FEWSHOT_DATA_ROOT", str(tmp_path))
    assert data_root() == Path(tmp_path)
    assert dataset_dir("mvtec_ad") == Path(tmp_path) / "mvtec_ad"


def test_results_root_defaults_into_the_repo(monkeypatch):
    monkeypatch.delenv("FEWSHOT_RESULTS_ROOT", raising=False)
    assert results_root() == repo_root() / "results"


def _make_mvtec(base, categories):
    for category in categories:
        (base / category / "train" / "good").mkdir(parents=True)
        (base / category / "test" / "good").mkdir(parents=True)
    return base


def test_finds_a_dataset_mounted_under_any_name(tmp_path):
    """Kaggle mounts an attached dataset under the uploader's slug, so the root
    has to be found by its category folders, not by its name."""
    mounted = tmp_path / "input" / "someones-mvtec-ad-upload" / "mvtec_anomaly_detection"
    _make_mvtec(mounted, ["bottle", "cable", "capsule"])
    assert find_category_root(tmp_path / "input") == mounted


def test_returns_none_when_nothing_is_attached(tmp_path):
    (tmp_path / "input" / "some-other-dataset").mkdir(parents=True)
    assert find_category_root(tmp_path / "input") is None


def test_ignores_a_lone_category(tmp_path):
    """One category is our own single-category download, not a full mount."""
    _make_mvtec(tmp_path / "input" / "mvtec_ad", ["bottle"])
    assert find_category_root(tmp_path / "input") is None


def test_missing_base_is_not_an_error(tmp_path):
    assert find_category_root(tmp_path / "does_not_exist") is None
