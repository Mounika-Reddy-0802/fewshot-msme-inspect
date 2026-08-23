"""Evaluation: metrics and the harness that runs them.

Member B owns the evaluation harness. These are stand-ins with the metric
definitions pinned down (image/pixel AUROC, F1-max) so results produced now stay
comparable to whatever they build.
"""

from src.eval.metrics import Metrics, auroc, compute_metrics, f1_max, upsample_maps

__all__ = ["Metrics", "auroc", "compute_metrics", "f1_max", "upsample_maps"]
