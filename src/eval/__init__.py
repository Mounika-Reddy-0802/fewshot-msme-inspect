"""Evaluation: metrics and the harness that runs them.

Member B owns the evaluation harness. These are stand-ins with the metric
definitions pinned down (image/pixel AUROC, F1-max) so results produced now stay
comparable to whatever they build.
"""

from src.eval.metrics import (
    evaluate_image_level,
    evaluate_pixel_level,
    f1_max,
    image_auroc,
    pixel_auroc,
)
from src.eval.scoring import Metrics, compute_metrics, upsample_maps

__all__ = [
    "Metrics",
    "compute_metrics",
    "upsample_maps",
    "evaluate_image_level",
    "evaluate_pixel_level",
    "f1_max",
    "image_auroc",
    "pixel_auroc",
]
