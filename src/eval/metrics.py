"""Evaluation metrics for anomaly detection (T3.1).

Deliberately independent of any specific model: every function here takes
plain arrays of scores/labels and returns a metric. Whatever Member A's
model produces -- prototype-memory distances, adapter logits, whatever --
just needs to be turned into these two shapes before calling in:

    image-level:  y_true   int array [N]        0 = normal, 1 = anomalous
                  y_score  float array [N]       higher = more anomalous

    pixel-level:  y_true   int array [N, H, W]   0/1 ground-truth masks
                  y_score  float array [N, H, W] higher = more anomalous

This keeps the harness usable the moment any scores exist -- including
placeholder / random scores today, for testing -- without waiting on the
model.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


def image_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Image-level AUROC: does the anomaly score rank anomalous images above normal ones."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(np.unique(y_true)) < 2:
        raise ValueError("image_auroc needs both normal and anomalous images in y_true")
    return float(roc_auc_score(y_true, y_score))


def pixel_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pixel-level AUROC over every pixel of every image, flattened together."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(np.unique(y_true)) < 2:
        raise ValueError("pixel_auroc needs both normal and anomalous pixels in y_true")
    return float(roc_auc_score(y_true, y_score))


def f1_max(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Best F1 achievable across all thresholds (image- or pixel-level, same shapes as above)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(np.max(f1_scores))


def evaluate_image_level(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """One-call summary of every image-level metric base.yaml expects."""
    return {
        "image_auroc": image_auroc(y_true, y_score),
        "image_f1_max": f1_max(y_true, y_score),
    }


def evaluate_pixel_level(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """One-call summary of every pixel-level metric base.yaml expects."""
    return {
        "pixel_auroc": pixel_auroc(y_true, y_score),
        "pixel_f1_max": f1_max(y_true, y_score),
    }
