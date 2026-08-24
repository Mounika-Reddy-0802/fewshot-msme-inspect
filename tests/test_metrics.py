"""Sanity checks for src/eval/metrics.py using synthetic scores.

These prove the harness itself is correct before any real model exists to
plug into it: perfect scores should give perfect metrics, and scores with
no separating power should give near-chance metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    evaluate_image_level,
    evaluate_pixel_level,
    f1_max,
    image_auroc,
    pixel_auroc,
)


def test_image_auroc_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.15, 0.9, 0.95, 0.8])
    assert image_auroc(y_true, y_score) == pytest.approx(1.0)


def test_image_auroc_near_chance():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)  # random scores carry no real signal
    auroc = image_auroc(y_true, y_score)
    assert 0.35 < auroc < 0.65  # loose band -- randomness, not a tight bound


def test_image_auroc_requires_both_classes():
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        image_auroc(y_true, y_score)


def test_f1_max_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert f1_max(y_true, y_score) == pytest.approx(1.0)


def test_pixel_auroc_perfect_separation():
    y_true = np.zeros((2, 4, 4), dtype=int)
    y_true[0, 0, 0] = 1
    y_true[1, 1, 1] = 1
    y_score = np.zeros((2, 4, 4))
    y_score[0, 0, 0] = 1.0
    y_score[1, 1, 1] = 1.0
    assert pixel_auroc(y_true, y_score) == pytest.approx(1.0)


def test_evaluate_image_level_returns_expected_keys():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.3, 0.7, 0.9])
    result = evaluate_image_level(y_true, y_score)
    assert set(result.keys()) == {"image_auroc", "image_f1_max"}
    assert result["image_auroc"] == pytest.approx(1.0)


def test_evaluate_pixel_level_returns_expected_keys():
    y_true = np.array([[0, 1], [0, 0]])
    y_score = np.array([[0.1, 0.9], [0.2, 0.3]])
    result = evaluate_pixel_level(y_true, y_score)
    assert set(result.keys()) == {"pixel_auroc", "pixel_f1_max"}
