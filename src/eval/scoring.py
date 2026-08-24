"""Anomaly-map post-processing and metric aggregation (Member A).

This sits between the model and Member B's metric harness in
:mod:`src.eval.metrics`, which stays the single source of truth for how AUROC
and F1-max are computed. Nothing here reimplements a metric; it prepares the
arrays her functions expect and packages the four results.

Two jobs:

**Get the anomaly map to mask resolution.** Maps come out at the patch grid
(14x14 for ViT-B/16, 7x7 for ViT-B/32), while masks are full images. Bilinear
upsampling plus a Gaussian blur is the standard treatment, and the blur matters:
neighbouring patches share receptive field, so a raw patch grid is blockier than
the evidence behind it.

**Tolerate degenerate splits.** Her metrics raise when only one class is present,
which is the right behaviour for a harness. A benchmark sweeping many categories
and shot counts would abort partway on that, so here it becomes NaN and the
summary reports "n/a" rather than a fabricated number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from src.eval.metrics import evaluate_image_level, evaluate_pixel_level


@dataclass(frozen=True)
class Metrics:
    image_auroc: float
    pixel_auroc: float | None
    image_f1_max: float
    pixel_f1_max: float | None

    def as_dict(self) -> dict:
        return dict(asdict(self))


def _to_numpy(x):
    import numpy as np

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over (B, H, W) maps.

    Reflect padding, not zero padding: with zeros a uniform 0.7 field blurs down
    to 0.21 at the border, so every image edge would read as anomalous.
    """
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


def _safe(fn, y_true, y_score) -> dict:
    """Run one of Member B's evaluators, turning a degenerate split into NaN."""
    try:
        return fn(_to_numpy(y_true), _to_numpy(y_score))
    except ValueError:
        return {}


def compute_metrics(
    image_scores,
    image_labels,
    pixel_scores=None,
    pixel_labels=None,
) -> Metrics:
    """All four metrics, computed by :mod:`src.eval.metrics`.

    Pixel metrics are None when no masks were supplied, NaN when the masks
    contain only one class.
    """
    image = _safe(evaluate_image_level, image_labels, image_scores)

    pixel: dict = {}
    has_pixels = pixel_scores is not None and pixel_labels is not None
    if has_pixels:
        pixel = _safe(evaluate_pixel_level, pixel_labels, pixel_scores)

    nan = float("nan")
    return Metrics(
        image_auroc=image.get("image_auroc", nan),
        image_f1_max=image.get("image_f1_max", nan),
        pixel_auroc=pixel.get("pixel_auroc", nan) if has_pixels else None,
        pixel_f1_max=pixel.get("pixel_f1_max", nan) if has_pixels else None,
    )
