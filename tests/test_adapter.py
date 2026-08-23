"""Tip-Adapter tests.

The interesting properties are about the kernel's limits: beta should interpolate
between spreading credit across the cache and plain nearest-neighbour, and alpha
should interpolate between the cache and the text branch. Both are asserted
rather than assumed, because a wrong sign or a swapped index would still produce
plausible-looking numbers.
"""

import pytest
import torch

from src.model.adapter import AdapterConfig, TipAdapter
from src.model.encoder import l2_normalise
from src.model.memory import PrototypeMemory

GRID, D = (4, 4), 8
P = GRID[0] * GRID[1]


def _keys(n=20, seed=0):
    return l2_normalise(torch.randn(n, D, generator=torch.Generator().manual_seed(seed)))


def _patches(n=3, seed=1):
    return l2_normalise(torch.randn(n, P, D, generator=torch.Generator().manual_seed(seed)))


def _text(seed=2):
    return l2_normalise(torch.randn(2, D, generator=torch.Generator().manual_seed(seed)))


# --- config validation ------------------------------------------------------

@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_alpha_must_be_a_mixing_weight(alpha):
    with pytest.raises(ValueError, match="alpha"):
        AdapterConfig(alpha=alpha)


def test_beta_must_be_positive():
    with pytest.raises(ValueError, match="beta"):
        AdapterConfig(beta=0)


def test_top_k_must_be_positive():
    with pytest.raises(ValueError, match="top_k"):
        AdapterConfig(top_k=0)


# --- shapes -----------------------------------------------------------------

def test_scores_have_the_right_shapes():
    adapter = TipAdapter(_keys(), GRID)
    scores = adapter(_patches(5))
    assert scores.image.shape == (5,)
    assert scores.patches.shape == (5, *GRID)


def test_rejects_wrong_key_rank():
    with pytest.raises(ValueError, match="cache keys"):
        TipAdapter(torch.randn(2, P, D), GRID)


def test_rejects_wrong_query_rank():
    with pytest.raises(ValueError, match="expected"):
        TipAdapter(_keys(), GRID)(torch.randn(P, D))


def test_text_must_be_a_normal_anomalous_pair():
    with pytest.raises(ValueError, match="2 text embeddings"):
        TipAdapter(_keys(), GRID, text_embeddings=torch.randn(3, D))


# --- the affinity kernel ----------------------------------------------------

def test_a_memorised_patch_scores_as_normal():
    """exp(-beta(1-s)) equals 1 at s=1, so anomaly is 0 for an exact match."""
    keys = _keys(20)                                    # must hold at least P patches
    query = keys[:P].unsqueeze(0)                       # patches drawn from the cache
    adapter = TipAdapter(keys, GRID, AdapterConfig(alpha=0.0))
    assert torch.allclose(adapter.cache_anomaly(query), torch.zeros(1, P), atol=1e-5)


def test_anomaly_stays_in_the_unit_interval():
    adapter = TipAdapter(_keys(), GRID, AdapterConfig(alpha=0.0))
    anomaly = adapter.cache_anomaly(_patches(4))
    assert (anomaly >= 0).all() and (anomaly <= 1).all()


def test_larger_beta_sharpens_the_kernel():
    """A sharper kernel is less forgiving of an imperfect match, so unfamiliar
    patches score higher."""
    keys, query = _keys(), _patches(4)
    soft = TipAdapter(keys, GRID, AdapterConfig(beta=1.0, alpha=0.0)).cache_anomaly(query)
    sharp = TipAdapter(keys, GRID, AdapterConfig(beta=20.0, alpha=0.0)).cache_anomaly(query)
    assert (sharp >= soft - 1e-6).all()
    assert sharp.mean() > soft.mean()


def test_unfamiliar_patches_score_above_memorised_ones():
    keys = _keys(30, seed=5)
    familiar = keys[:P].unsqueeze(0)
    novel = _patches(1, seed=99)
    adapter = TipAdapter(keys, GRID, AdapterConfig(alpha=0.0))
    assert adapter.cache_anomaly(novel).mean() > adapter.cache_anomaly(familiar).mean()


def test_top_k_averages_more_of_the_cache():
    keys, query = _keys(), _patches(3)
    one = TipAdapter(keys, GRID, AdapterConfig(top_k=1, alpha=0.0)).cache_anomaly(query)
    five = TipAdapter(keys, GRID, AdapterConfig(top_k=5, alpha=0.0)).cache_anomaly(query)
    assert (five >= one - 1e-6).all()


# --- blending ---------------------------------------------------------------

def test_alpha_zero_is_pure_cache():
    keys, query = _keys(), _patches(3)
    adapter = TipAdapter(keys, GRID, AdapterConfig(alpha=0.0), _text())
    assert torch.allclose(
        adapter(query).patches.reshape(3, P), adapter.cache_anomaly(query), atol=1e-6
    )


def test_alpha_one_is_pure_text():
    keys, query = _keys(), _patches(3)
    adapter = TipAdapter(keys, GRID, AdapterConfig(alpha=1.0), _text())
    assert torch.allclose(
        adapter(query).patches.reshape(3, P), adapter.text_anomaly(query), atol=1e-6
    )


def test_blend_lies_between_its_two_branches():
    keys, query = _keys(), _patches(3)
    adapter = TipAdapter(keys, GRID, AdapterConfig(alpha=0.5), _text())
    cache = adapter.cache_anomaly(query)
    text = adapter.text_anomaly(query)
    blended = adapter(query).patches.reshape(3, P)
    assert (blended >= torch.minimum(cache, text) - 1e-6).all()
    assert (blended <= torch.maximum(cache, text) + 1e-6).all()


def test_text_branch_needs_text():
    with pytest.raises(RuntimeError, match="no text embeddings"):
        TipAdapter(_keys(), GRID).text_anomaly(_patches(1))


# --- interop with the memory ------------------------------------------------

def test_builds_from_an_existing_memory():
    """A class already in memory should gain an adapter without re-extracting
    anything."""
    memory = PrototypeMemory(dim=D, grid=GRID)
    memory.add_class("bottle", _patches(3, seed=7))

    adapter = TipAdapter.from_memory(memory, "bottle")
    assert adapter.num_keys == memory.size("bottle")
    assert adapter.grid == memory.grid


def test_from_memory_rejects_an_unknown_class():
    memory = PrototypeMemory(dim=D, grid=GRID)
    memory.add_class("bottle", _patches(1))
    with pytest.raises(KeyError, match="cable"):
        TipAdapter.from_memory(memory, "cable")


def test_keys_are_frozen_unless_asked():
    assert not TipAdapter(_keys(), GRID).keys.requires_grad
    assert TipAdapter(_keys(), GRID, AdapterConfig(trainable=True)).keys.requires_grad


def test_trainable_keys_carry_gradient():
    """Tip-Adapter-F unfreezes the cache; scoring must stay differentiable."""
    adapter = TipAdapter(_keys(), GRID, AdapterConfig(trainable=True, alpha=0.0))
    adapter(_patches(2)).image.sum().backward()
    assert adapter.keys.grad is not None
    assert torch.isfinite(adapter.keys.grad).all()
