"""Encoder tests.

The ones that load real weights are marked `slow` and skipped unless the weights
are already cached, so the suite stays fast on a machine that has not downloaded
them yet.
"""

import pytest
import torch

from src.model.encoder import EncodedBatch, check_backbone_pair, l2_normalise


def test_quickgelu_mismatch_is_fatal():
    """A warning is not enough -- this pairing silently changes every feature."""
    with pytest.raises(ValueError, match="quickgelu"):
        check_backbone_pair("ViT-B-16", "openai")


def test_correct_pairings_are_accepted():
    check_backbone_pair("ViT-B-16-quickgelu", "openai")
    check_backbone_pair("ViT-B-16", "laion2b_s34b_b88k")  # laion weights use plain GELU


def test_l2_normalise_gives_unit_vectors():
    x = torch.randn(4, 8)
    n = l2_normalise(x)
    assert torch.allclose(n.norm(dim=-1), torch.ones(4), atol=1e-6)


def test_l2_normalise_survives_a_zero_vector():
    assert torch.isfinite(l2_normalise(torch.zeros(1, 8))).all()


def test_patch_map_reshapes_to_the_grid():
    batch = EncodedBatch(
        image=torch.randn(2, 512),
        patches=torch.randn(2, 196, 512),
        grid=(14, 14),
    )
    assert batch.patch_map().shape == (2, 14, 14, 512)


def test_patch_map_preserves_order():
    patches = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    batch = EncodedBatch(image=torch.zeros(2, 3), patches=patches, grid=(2, 2))
    assert torch.equal(batch.patch_map().reshape(2, 4, 3), patches)


# --- tests that need the real weights ---------------------------------------

def _encoder():
    from src.model.encoder import FrozenCLIPEncoder

    return FrozenCLIPEncoder("ViT-B-16-quickgelu", "openai", device="cpu")


@pytest.fixture(scope="module")
def encoder():
    try:
        return _encoder()
    except Exception as exc:  # weights not cached / offline
        pytest.skip(f"CLIP weights unavailable: {type(exc).__name__}")


@pytest.mark.slow
def test_grid_and_dims_match_vit_b16(encoder):
    assert encoder.grid_size == (14, 14)
    assert encoder.num_patches == 196
    assert encoder.embed_dim == 512
    assert encoder.image_size == (224, 224)


@pytest.mark.slow
def test_encode_shapes(encoder):
    out = encoder.encode_images(torch.zeros(2, 3, 224, 224))
    assert out.image.shape == (2, 512)
    assert out.patches.shape == (2, 196, 512)
    assert out.patch_map().shape == (2, 14, 14, 512)


@pytest.mark.slow
def test_outputs_are_normalised(encoder):
    out = encoder.encode_images(torch.rand(2, 3, 224, 224))
    assert torch.allclose(out.image.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(out.patches.norm(dim=-1), torch.ones(2, 196), atol=1e-5)


@pytest.mark.slow
def test_patches_and_text_share_one_space(encoder):
    """Open-vocabulary scoring depends on this: a patch has to be comparable to
    a prompt, which it only is if the projection was applied."""
    out = encoder.encode_images(torch.rand(1, 3, 224, 224))
    text = encoder.encode_text(["a photo of a bottle", "a photo of a broken bottle"])
    assert text.shape == (2, 512)
    similarity = out.patches[0] @ text.T
    assert similarity.shape == (196, 2)
    assert torch.isfinite(similarity).all()
    assert (similarity.abs() <= 1.0 + 1e-4).all()   # cosine, both sides normalised


@pytest.mark.slow
def test_backbone_is_frozen(encoder):
    assert not any(p.requires_grad for p in encoder.model.parameters())


@pytest.mark.slow
def test_encoding_is_deterministic(encoder):
    x = torch.rand(1, 3, 224, 224)
    assert torch.allclose(encoder.encode_images(x).image, encoder.encode_images(x).image)


def test_multi_layer_last_block_equals_default(encoder):
    """layers=(12,) must reproduce the v1.0 features exactly, so release 1.1 cannot
    have changed any earlier result."""
    from src.model.encoder import FrozenCLIPEncoder

    last = FrozenCLIPEncoder(encoder.name, encoder.pretrained, project_patches=False, layers=(12,))
    base = FrozenCLIPEncoder(encoder.name, encoder.pretrained, project_patches=False)
    x = torch.randn(2, 3, *base.image_size, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(last.encode_images(x).patches, base.encode_images(x).patches, atol=1e-5)


def test_multi_layer_concatenates_unit_blocks(encoder):
    from src.model.encoder import FrozenCLIPEncoder

    ml = FrozenCLIPEncoder(encoder.name, encoder.pretrained, project_patches=False, layers=(6, 9, 12))
    out = ml.encode_images(torch.zeros(1, 3, *ml.image_size))
    assert out.patches.shape[-1] == ml.embed_dim == 3 * 768
    assert torch.allclose(out.patches.norm(dim=-1), torch.ones(1, ml.num_patches), atol=1e-4)
    assert ml.name.endswith("-ml6-9-12")
