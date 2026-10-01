"""The demo engine, on a fake encoder so no weights are needed."""

from __future__ import annotations

import pytest
import torch

from src.demo.inspector import Inspector
from src.model.encoder import EncodedBatch, l2_normalise


class FakeEncoder:
    """Each 'image' is a tensor already; its patches are fixed random vectors
    seeded by the image's value, so identical images encode identically."""

    embed_dim = 8
    grid_size = (2, 2)
    image_size = (16, 16)

    def preprocess(self, image):
        return image

    def encode_images(self, batch):
        patches = []
        for img in batch:
            g = torch.Generator().manual_seed(int(img.sum().item() * 1000) % (2**31))
            patches.append(torch.randn(4, 8, generator=g))
        patches = l2_normalise(torch.stack(patches))
        return EncodedBatch(patches.mean(1), patches, self.grid_size)


def image(value: float) -> torch.Tensor:
    return torch.full((3, 16, 16), value)


@pytest.fixture
def inspector():
    return Inspector(FakeEncoder())


def test_a_good_photo_scores_zero_against_itself(inspector):
    inspector.add_product("a", [image(0.1), image(0.2)])
    assert inspector.inspect("a", image(0.1)).score == pytest.approx(0.0, abs=1e-6)


def test_an_unseen_part_scores_higher(inspector):
    inspector.add_product("a", [image(0.1), image(0.2)])
    assert inspector.inspect("a", image(0.9)).score > 0.05


def test_heatmap_is_at_input_resolution(inspector):
    inspector.add_product("a", [image(0.1)])
    assert inspector.inspect("a", image(0.3)).heatmap.shape == (16, 16)


def test_adding_a_product_leaves_others_bit_identical(inspector):
    inspector.add_product("a", [image(0.1), image(0.2)])
    before = inspector.inspect("a", image(0.7)).score
    inspector.add_product("b", [image(0.5)])
    assert inspector.inspect("a", image(0.7)).score == before
    assert inspector.products == ["a", "b"]


def test_adding_photos_appends(inspector):
    inspector.add_product("a", [image(0.1)])
    result = inspector.add_product("a", [image(0.2)])
    assert result.images == 2 and result.prototypes == 8


def test_threshold_needs_two_photos(inspector):
    inspector.add_product("a", [image(0.1)])
    assert inspector.suggested_threshold("a") is None
    inspector.add_product("a", [image(0.2)])
    assert inspector.suggested_threshold("a") > 0


def test_threshold_is_the_worst_held_out_good_score(inspector):
    photos = [image(0.1), image(0.2), image(0.3)]
    inspector.add_product("a", photos)
    threshold = inspector.suggested_threshold("a")
    held_out = []
    for i in range(3):
        other = Inspector(FakeEncoder())
        other.add_product("x", photos[:i] + photos[i + 1:])
        held_out.append(other.inspect("x", photos[i]).score)
    assert threshold == pytest.approx(max(held_out), abs=1e-6)


def test_no_photos_is_an_error(inspector):
    with pytest.raises(ValueError):
        inspector.add_product("a", [])


class TextEncoder(FakeEncoder):
    """Adds prompts: every text encodes to a fixed vector per wording."""

    def encode_text(self, prompts):
        out = []
        for p in prompts:
            g = torch.Generator().manual_seed(sum(map(ord, p)))
            out.append(torch.randn(8, generator=g))
        return l2_normalise(torch.stack(out))


def test_description_fuses_text_into_the_image_score():
    inspector = Inspector(TextEncoder())
    inspector.add_product("bracket", [image(0.1), image(0.2)], description="a cracked rim")
    result = inspector.inspect("bracket", image(0.9))
    assert result.text_score is not None
    assert result.score == pytest.approx(0.5 * result.memory_score + 0.5 * result.text_score)
    assert inspector.descriptions["bracket"] == "a cracked rim"


def test_without_a_description_the_score_is_the_memory_score():
    inspector = Inspector(TextEncoder())
    inspector.add_product("bracket", [image(0.1)])
    result = inspector.inspect("bracket", image(0.9))
    assert result.text_score is None and result.score == result.memory_score
