"""Feature bank and pipeline tests, on synthetic features so they stay fast."""

import pytest
import torch

from src.eval.features import FeatureBank
from src.eval.pipeline import load_masks

GRID, D = (4, 4), 8
P = GRID[0] * GRID[1]


@pytest.fixture
def bank():
    b = FeatureBank("ViT-test", "bottle", GRID, D)
    paths = ["train/good/000.png", "test/good/000.png", "test/broken/000.png"]
    b._index = {p: i for i, p in enumerate(paths)}
    b._images = torch.randn(3, D).half()
    b._patches = torch.randn(3, P, D).half()
    return b


def test_lookup_preserves_requested_order(bank):
    a = bank.patches(["test/broken/000.png", "train/good/000.png"])
    b = bank.patches(["train/good/000.png", "test/broken/000.png"])
    assert torch.equal(a[0], b[1]) and torch.equal(a[1], b[0])


def test_features_widen_to_float32(bank):
    """Stored as fp16 to halve the bank; similarities must be computed in fp32."""
    assert bank._patches.dtype == torch.float16
    assert bank.patches(["test/good/000.png"]).dtype == torch.float32


def test_missing_path_names_itself(bank):
    with pytest.raises(KeyError, match="nope.png"):
        bank.patches(["nope.png"])


def test_membership(bank):
    assert "test/good/000.png" in bank
    assert "absent.png" not in bank


def test_round_trip_through_disk(bank, tmp_path):
    path = bank.save(tmp_path / "bottle.pt")
    loaded = FeatureBank.load(path)
    assert loaded.backbone == bank.backbone and loaded.grid == bank.grid
    assert torch.equal(
        loaded.patches(["test/good/000.png"]), bank.patches(["test/good/000.png"])
    )


def test_bank_path_separates_backbones(tmp_path):
    from src.eval.features import bank_path

    a = bank_path(tmp_path, "ViT-B-16-quickgelu", "bottle")
    b = bank_path(tmp_path, "ViT-B-32-quickgelu", "bottle")
    assert a != b, "mixing features from two backbones would be silently wrong"


# --- masks ------------------------------------------------------------------

def _write_mask(path, size=(64, 64), box=None):
    from PIL import Image

    img = Image.new("L", size, 0)
    if box:
        for x in range(box[0], box[2]):
            for y in range(box[1], box[3]):
                img.putpixel((x, y), 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def test_normal_images_get_an_empty_mask():
    masks = load_masks([None, None], (32, 32))
    assert masks.shape == (2, 32, 32)
    assert masks.sum() == 0


def test_mask_is_binary_after_resizing(tmp_path):
    path = _write_mask(tmp_path / "m.png", box=(10, 10, 40, 40))
    masks = load_masks([path], (32, 32))
    assert set(masks.unique().tolist()) <= {0, 1}
    assert masks.sum() > 0


def test_mask_keeps_its_position(tmp_path):
    """A defect in the top-left must not migrate. If resize and centre-crop did
    not mirror the image preprocessing, the mask would not line up with the map."""
    path = _write_mask(tmp_path / "m.png", size=(64, 64), box=(0, 0, 32, 32))
    mask = load_masks([path], (32, 32))[0]
    assert mask[:16, :16].float().mean() > 0.9    # defect stayed top-left
    assert mask[16:, 16:].float().mean() < 0.1    # and did not appear bottom-right


def test_mixed_masks_and_normals(tmp_path):
    path = _write_mask(tmp_path / "m.png", box=(10, 10, 40, 40))
    masks = load_masks([None, path, None], (32, 32))
    assert masks.shape == (3, 32, 32)
    assert masks[0].sum() == 0 and masks[1].sum() > 0 and masks[2].sum() == 0


# --- caching ----------------------------------------------------------------

def test_mask_cache_returns_identical_tensors(tmp_path):
    """Same category, same test split -> the masks are loaded once and reused."""
    from src.eval import pipeline

    pipeline.clear_caches()
    path = _write_mask(tmp_path / "bottle" / "gt.png", box=(4, 4, 20, 20))
    key = (str(tmp_path / "bottle"), ("gt.png",), (32, 32))

    masks = load_masks([path], (32, 32))
    pipeline._MASK_CACHE[key] = masks
    assert pipeline._MASK_CACHE[key] is masks


def test_clear_caches_empties_both():
    from src.eval import pipeline

    pipeline._MASK_CACHE["x"] = torch.zeros(1)
    pipeline._TEXT_CACHE["y"] = torch.zeros(1)
    pipeline.clear_caches()
    assert not pipeline._MASK_CACHE and not pipeline._TEXT_CACHE


def test_text_cache_keys_on_the_backbone():
    """Two backbones live in different embedding spaces; sharing a cached text
    embedding between them would be silently wrong."""
    from src.eval import pipeline

    pipeline.clear_caches()

    class FakeEncoder:
        def __init__(self, name):
            self.name, self.pretrained = name, "openai"
            self.calls = 0

        def encode_text(self, prompts):
            self.calls += 1
            return torch.ones(len(prompts), 4)

    a, b = FakeEncoder("ViT-B-16"), FakeEncoder("ViT-L-14")
    pipeline.category_text_embeddings(a, "bottle")
    pipeline.category_text_embeddings(a, "bottle")      # cached, no new calls
    pipeline.category_text_embeddings(b, "bottle")      # different backbone

    assert a.calls == 2, "two ensembles on first call, then cached"
    assert b.calls == 2, "must not reuse the other backbone's embedding"
    pipeline.clear_caches()
