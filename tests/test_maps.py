"""Tests for Member A's map post-processing and metric aggregation.

Member B's metric functions are tested in tests/test_metrics.py; these cover the
layer that feeds them.
"""

import numpy as np
import pytest
import torch

from src.eval.scoring import compute_metrics, gaussian_blur, upsample_maps


def test_upsample_reaches_mask_resolution():
    assert upsample_maps(torch.rand(3, 14, 14), (224, 224), sigma=0).shape == (3, 224, 224)


def test_upsample_rejects_wrong_rank():
    with pytest.raises(ValueError, match="expected"):
        upsample_maps(torch.rand(14, 14), (224, 224))


def test_blur_leaves_a_constant_field_flat():
    """Why reflect padding, not zeros: with zeros a uniform 0.7 field blurs down
    to 0.21 at the border, so every image edge would read as anomalous."""
    const = torch.full((1, 32, 32), 0.7)
    assert torch.allclose(gaussian_blur(const, 4.0), const, atol=1e-6)


def test_blur_preserves_mass_away_from_the_border():
    spike = torch.zeros(1, 64, 64)
    spike[0, 32, 32] = 1.0
    assert gaussian_blur(spike, 2.0).sum() == pytest.approx(1.0, rel=1e-5)


def test_blur_is_a_noop_at_zero_sigma():
    maps = torch.rand(2, 8, 8)
    assert torch.equal(gaussian_blur(maps, 0), maps)


def test_blur_smooths_a_spike():
    maps = torch.zeros(1, 21, 21)
    maps[0, 10, 10] = 1.0
    blurred = gaussian_blur(maps, 2.0)
    assert blurred[0, 10, 10] < 1.0
    assert blurred[0, 10, 11] > 0.0


def test_aggregation_delegates_to_the_harness():
    m = compute_metrics([0.1, 0.9], [0, 1])
    assert m.image_auroc == 1.0
    assert m.pixel_auroc is None and m.pixel_f1_max is None


def test_aggregation_with_masks():
    m = compute_metrics(
        [0.1, 0.9], [0, 1],
        pixel_scores=torch.rand(2, 4, 4),
        pixel_labels=torch.randint(0, 2, (2, 4, 4)),
    )
    assert m.pixel_auroc is not None
    assert set(m.as_dict()) == {"image_auroc", "pixel_auroc", "image_f1_max", "pixel_f1_max"}


def test_degenerate_split_becomes_nan_not_a_crash():
    """Her harness raises when only one class is present -- correct for a
    harness, fatal for a sweep over 15 categories. It must degrade to NaN."""
    m = compute_metrics([0.1, 0.2], [0, 0])
    assert np.isnan(m.image_auroc)


def test_accepts_torch_tensors():
    m = compute_metrics(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
    assert m.image_auroc == 1.0
