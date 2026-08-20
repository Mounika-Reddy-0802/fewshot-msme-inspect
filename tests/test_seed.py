"""Seeding has to be reproducible or none of the few-shot numbers mean anything."""

import random

from src.utils.seed import DEFAULT_SEED, seed_everything


def _draw():
    return [random.random() for _ in range(5)]


def test_same_seed_same_draw():
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    assert _draw() == first


def test_different_seed_different_draw():
    seed_everything(1)
    first = _draw()
    seed_everything(2)
    assert _draw() != first


def test_returns_the_seed_it_used():
    assert seed_everything() == DEFAULT_SEED
