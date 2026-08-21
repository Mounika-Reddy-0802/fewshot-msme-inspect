"""The configs have to load, inherit and resolve before anything else works."""

import pytest
import yaml

from src.utils.config import CONFIG_DIR, load_config

EXPERIMENT_CONFIGS = ["baseline.yaml", "prototype.yaml", "adapter.yaml"]


@pytest.mark.parametrize("name", EXPERIMENT_CONFIGS)
def test_experiment_config_loads(name):
    config = load_config(name)
    assert config["experiment"]["seed"] == 42
    assert config["dataset"]["name"] == "mvtec_ad"
    assert config["backbone"]["frozen"] is True


@pytest.mark.parametrize("name", EXPERIMENT_CONFIGS)
def test_no_placeholder_survives(name):
    text = yaml.safe_dump(load_config(name))
    assert "${" not in text


def test_extends_merges_instead_of_replacing():
    config = load_config("baseline.yaml")
    # `shots` is overridden by the child, the rest of `dataset` comes from base
    assert config["dataset"]["shots"] == 0
    assert config["dataset"]["image_size"] == [224, 224]


def test_adapter_inherits_through_two_levels():
    config = load_config("adapter.yaml")
    assert config["model"]["memory"]["append_only"] is True   # from prototype
    assert config["model"]["kind"] == "adapter"               # its own
    assert config["profiling"]["device"] == "cpu"             # from base


def test_dotted_overrides():
    config = load_config("prototype.yaml", overrides=["dataset.shots=10", "backbone.name=ViT-L-14"])
    assert config["dataset"]["shots"] == 10
    assert config["backbone"]["name"] == "ViT-L-14"


def test_missing_config_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_config("nope.yaml")


def test_unknown_placeholder_is_an_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("dataset:\n  root: ${not_a_root}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(bad)


def test_every_shipped_config_is_covered():
    shipped = {p.name for p in CONFIG_DIR.glob("*.yaml")} - {"base.yaml"}
    assert shipped == set(EXPERIMENT_CONFIGS)


@pytest.mark.parametrize("name", EXPERIMENT_CONFIGS)
def test_openai_weights_use_a_quickgelu_backbone(name):
    """OpenAI CLIP weights need QuickGELU; open_clip only warns if they do not
    match, and the resulting features are silently different."""
    backbone = load_config(name)["backbone"]
    if backbone["pretrained"] == "openai":
        assert backbone["name"].endswith("-quickgelu"), backbone["name"]
