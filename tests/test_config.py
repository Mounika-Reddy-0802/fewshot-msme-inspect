"""The configs have to load, inherit and resolve before anything else works."""

import pytest
import yaml

from src.utils.config import CONFIG_DIR, load_config

EXPERIMENT_CONFIGS = ["baseline.yaml", "prototype.yaml", "adapter.yaml", "proposed.yaml"]


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


def test_scoring_temperature_is_a_real_knob():
    """Hardcoding this while the config advertised it would silently ignore the
    setting -- the same failure as the adapter's alpha."""
    import inspect

    from src.eval.pipeline import evaluate_split, zero_shot_scores

    assert "temperature" in inspect.signature(evaluate_split).parameters
    assert "temperature" in inspect.signature(zero_shot_scores).parameters
    assert load_config("prototype.yaml")["model"]["scoring"]["temperature"] > 0


def test_image_size_is_a_real_knob():
    import inspect

    from src.eval.pipeline import evaluate_split

    assert "image_size" in inspect.signature(evaluate_split).parameters
    assert load_config("base.yaml")["dataset"]["image_size"] == [224, 224]


def test_unread_config_keys_say_so():
    """Every key nothing consumes carries a NOT YET READ marker, so no one sets
    one and assumes it took effect."""
    unread = {
        "configs/base.yaml": ["split_manifest", "output_dir", "track", "enabled"],
        "configs/prototype.yaml": ["append_only", "normalise", "reduction",
                                   "patch_level", "max_prototypes_per_class"],
        "configs/adapter.yaml": ["init_from_memory"],
    }
    for path, keys in unread.items():
        text = (CONFIG_DIR.parent / path).read_text(encoding="utf-8")
        for key in keys:
            line = next(l for l in text.splitlines() if l.strip().startswith(f"{key}:"))
            assert "NOT YET READ" in line, f"{path}:{key} is unread but not marked"


def test_proposed_config_turns_on_windows_and_text():
    scoring = load_config("proposed.yaml")["model"]["scoring"]
    assert scoring["scales"] == [1, 2, 3] and scoring["text_weight"] == 0.5
    # the plain prototype config stays single-pass, so earlier results reproduce
    plain = load_config("prototype.yaml")["model"]["scoring"]
    assert plain["scales"] == [1] and plain["text_weight"] == 0.0
