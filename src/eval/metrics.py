"""Evaluation metrics.

Four numbers, matching what the anomaly-detection literature reports so our
results are comparable to published ones:

* **image AUROC** -- can it tell a defective part from a good one
* **pixel AUROC** -- does it point at the right region
* **image F1-max** -- best achievable F1 over all thresholds, which is closer to
  what an MSME actually cares about than AUROC
* **pixel F1-max** -- same, per pixel

F1-max is reported instead of F1-at-some-threshold because no threshold has been
chosen yet; picking one is a deployment decision, and quoting F1 at an arbitrary
cut would understate every method equally but arbitrarily.

Anomaly maps come out at the patch grid (14x14 for ViT-B/16) and have to reach
mask resolution before pixel metrics mean anything. :func:`upsample_maps` does
that with bilinear interpolation plus optional Gaussian smoothing -- both are
standard, and the smoothing matters: neighbouring patches overlap in receptive
field, so a raw patch grid is blockier than the underlying evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Metrics:
    image_auroc: float
    pixel_auroc: float | None
    image_f1_max: float
    pixel_f1_max: float | None

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def auroc(scores, labels) -> float:
    """Area under the ROC curve. NaN if only one class is present."""
    from sklearn.metrics import roc_auc_score

    scores, labels = _to_numpy(scores).ravel(), _to_numpy(labels).ravel()
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def f1_max(scores, labels) -> float:
    """Best F1 over every threshold the data actually produces."""
    from sklearn.metrics import precision_recall_curve

    scores, labels = _to_numpy(scores).ravel(), _to_numpy(labels).ravel()
    if len(np.unique(labels)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, scores)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(np.nanmax(f1))


def gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over (B, H, W) maps."""
    if sigma <= 0:
        return maps
    radius = max(1, int(round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=maps.dtype, device=maps.device)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    x = maps.unsqueeze(1)                                   # (B, 1, H, W)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x.squeeze(1)


def upsample_maps(
    maps: torch.Tensor,
    size: tuple[int, int],
    sigma: float = 4.0,
) -> torch.Tensor:
    """Patch-grid anomaly maps -> mask resolution. (B, H, W) -> (B, *size)."""
    if maps.ndim != 3:
        raise ValueError(f"expected (B, H, W), got {tuple(maps.shape)}")
    upsampled = F.interpolate(
        maps.unsqueeze(1).float(), size=size, mode="bilinear", align_corners=False
    ).squeeze(1)
    return gaussian_blur(upsampled, sigma)


def compute_metrics(
    image_scores,
    image_labels,
    pixel_scores=None,
    pixel_labels=None,
) -> Metrics:
    """All four metrics. Pixel metrics are None when no masks were supplied."""
    has_pixels = pixel_scores is not None and pixel_labels is not None
    return Metrics(
        image_auroc=auroc(image_scores, image_labels),
        pixel_auroc=auroc(pixel_scores, pixel_labels) if has_pixels else None,
        image_f1_max=f1_max(image_scores, image_labels),
        pixel_f1_max=f1_max(pixel_scores, pixel_labels) if has_pixels else None,
    )
