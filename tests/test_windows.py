"""Window scoring and text fusion."""

from __future__ import annotations

import torch

from src.model.encoder import l2_normalise
from src.model.memory import PrototypeMemory
from src.model.windows import _spread, text_image_scores, window_pool, windowed_scores

GRID = (4, 4)


def tokens(n: int, seed: int) -> torch.Tensor:
    return l2_normalise(torch.randn(n, 16, 8, generator=torch.Generator().manual_seed(seed)))


def test_scale_one_is_exactly_the_memory_head():
    support, query = tokens(3, 0), tokens(2, 1)
    memory = PrototypeMemory(8, GRID)
    memory.add_class("c", support)
    reference = memory.score(query, "c")
    windowed = windowed_scores(support, query, GRID, scales=(1,))
    assert torch.equal(windowed.patches, reference.patches)
    assert torch.equal(windowed.image, reference.image)


def test_window_pool_shapes():
    x = tokens(2, 0)
    assert window_pool(x, GRID, 1).shape == (2, 16, 8)
    assert window_pool(x, GRID, 2).shape == (2, 9, 8)
    assert window_pool(x, GRID, 3).shape == (2, 4, 8)


def test_spread_returns_the_full_grid_and_averages_coverage():
    window_map = torch.zeros(1, 3, 3)
    window_map[0, 0, 0] = 1.0                 # only the top-left 2x2 window scores
    spread = _spread(window_map, 2)
    assert spread.shape == (1, 4, 4)
    assert spread[0, 0, 0] == 1.0             # covered by that window alone
    assert spread[0, 1, 1] == 0.25            # covered by four windows, one scoring
    assert spread[0, 3, 3] == 0.0


def test_multiscale_map_is_patch_grid_sized():
    out = windowed_scores(tokens(3, 0), tokens(2, 1), GRID, scales=(1, 2, 3))
    assert out.patches.shape == (2, 4, 4)
    assert out.image.shape == (2,)


def test_query_identical_to_support_scores_zero_at_every_scale():
    support = tokens(2, 0)
    out = windowed_scores(support, support[:1], GRID, scales=(1, 2, 3))
    assert out.image[0] < 1e-5


def test_text_score_follows_the_closer_prompt():
    text = torch.eye(2, 4)                     # normal = e0, anomalous = e1
    images = torch.tensor([[1.0, 0.0, 0, 0], [0.0, 1.0, 0, 0]])
    scores = text_image_scores(images, text)
    assert scores[0] < 0.5 < scores[1]
