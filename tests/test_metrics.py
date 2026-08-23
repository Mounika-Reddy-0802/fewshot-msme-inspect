"""Metric tests, mostly against cases where the answer is known by hand."""

import numpy as np
import pytest
import torch

from src.eval.metrics import (
    auroc,
    compute_metrics,
    f1_max,
    gaussian_blur,
    upsample_maps,
)


def test_perfect_separation_is_one():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_inverted_ranking_is_zero():
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0


def test_uninformative_scores_are_half():
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5


def test_auroc_is_nan_without_both_classes():
    assert np.isnan(auroc([0.1, 0.2], [0, 0]))


def test_f1_max_is_one_when_separable():
    assert f1_max([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_f1_max_beats_a_bad_threshold():
    """The point of F1-max: a poor fixed cut should not be mistaken for a poor
    model."""
    scores = [0.01, 0.02, 0.03, 0.04]
    labels = [0, 0, 1, 1]
    assert f1_max(scores, labels) == pytest.approx(1.0)


def test_f1_max_handles_all_zero_precision_and_recall():
    assert np.isfinite(f1_max([0.0, 0.0, 1.0, 1.0], [1, 1, 0, 0]))


def test_auroc_accepts_torch_tensors():
    assert auroc(torch.tensor([0.1, 0.9]), torch.tensor([0, 1])) == 1.0


def test_upsample_reaches_mask_resolution():
    maps = torch.rand(3, 14, 14)
    assert upsample_maps(maps, (224, 224), sigma=0).shape == (3, 224, 224)


def test_upsample_rejects_wrong_rank():
    with pytest.raises(ValueError, match="expected"):
        upsample_maps(torch.rand(14, 14), (224, 224))


def test_blur_leaves_a_constant_field_flat():
    """The reason for reflect padding rather than zero padding. With zeros, a
    uniform 0.7 field blurs down to 0.21 at the border -- so every image edge
    would read as anomalous and pixel AUROC would be junk."""
    const = torch.full((1, 32, 32), 0.7)
    blurred = gaussian_blur(const, 4.0)
    assert torch.allclose(blurred, const, atol=1e-6)


def test_blur_preserves_mass_away_from_the_border():
    """Mass is conserved for signal the kernel can see all of. Total mass over a
    whole random field is not conserved, because reflection duplicates edge
    values -- that is expected, not a bug."""
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
    assert blurred[0, 10, 10] < 1.0          # peak spread out
    assert blurred[0, 10, 11] > 0.0          # into its neighbour


def test_compute_metrics_without_masks():
    m = compute_metrics([0.1, 0.9], [0, 1])
    assert m.image_auroc == 1.0
    assert m.pixel_auroc is None and m.pixel_f1_max is None


def test_compute_metrics_with_masks():
    m = compute_metrics(
        [0.1, 0.9], [0, 1],
        pixel_scores=torch.rand(2, 4, 4), pixel_labels=torch.randint(0, 2, (2, 4, 4)),
    )
    assert m.pixel_auroc is not None
    assert set(m.as_dict()) == {"image_auroc", "pixel_auroc", "image_f1_max", "pixel_f1_max"}
