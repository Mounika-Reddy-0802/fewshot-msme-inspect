"""Few-shot PatchCore baseline: shapes and the basic ordering it must get right."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

pytest.importorskip("anomalib")


@pytest.fixture(scope="module")
def patchcore():
    from src.baselines.patchcore_fewshot import FewShotPatchCore
    try:
        return FewShotPatchCore()
    except Exception as exc:                      # backbone weights not downloadable here
        pytest.skip(f"WideResNet-50 weights unavailable: {type(exc).__name__}")


def _image(path, seed):
    g = torch.Generator().manual_seed(seed)
    pixels = (torch.rand(256, 256, 3, generator=g) * 255).to(torch.uint8).numpy()
    Image.fromarray(pixels).save(path)
    return path


def test_embedding_shape(patchcore, tmp_path):
    emb, grid = patchcore.embed([_image(tmp_path / "a.png", 0)])
    assert emb.shape[0] == 1 and emb.shape[1] == grid[0] * grid[1]
    assert emb.shape[2] == 512 + 1024              # layer2 + layer3 of WideResNet-50-2


def test_seen_part_scores_below_an_unseen_one(patchcore, tmp_path):
    support, grid = patchcore.embed([_image(tmp_path / "good.png", 1)])
    same, _ = patchcore.embed([tmp_path / "good.png"])
    other, _ = patchcore.embed([_image(tmp_path / "other.png", 2)])
    scores, maps = patchcore.score(support, torch.cat([same, other]), grid)
    assert scores[0] < scores[1]
    assert maps.shape == (2, patchcore.size, patchcore.size)
