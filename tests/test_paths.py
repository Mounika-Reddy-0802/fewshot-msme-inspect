"""Roots come from the environment, never from a hardcoded path."""

from pathlib import Path

from src.utils.paths import data_root, dataset_dir, repo_root, results_root


def test_repo_root_holds_the_configs():
    assert (repo_root() / "configs" / "base.yaml").exists()


def test_data_root_follows_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("FEWSHOT_DATA_ROOT", str(tmp_path))
    assert data_root() == Path(tmp_path)
    assert dataset_dir("mvtec_ad") == Path(tmp_path) / "mvtec_ad"


def test_results_root_defaults_into_the_repo(monkeypatch):
    monkeypatch.delenv("FEWSHOT_RESULTS_ROOT", raising=False)
    assert results_root() == repo_root() / "results"
