"""Tests for the figure helpers -- the Pareto logic in particular, since a wrong
front would misrepresent the headline result."""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "make_figures", Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"
)
make_figures = importlib.util.module_from_spec(spec)
sys.modules["make_figures"] = make_figures
spec.loader.exec_module(make_figures)

pareto_front = make_figures.pareto_front
mean = make_figures.mean


def test_front_drops_dominated_points():
    """(20, 0.80) is both slower and worse than (10, 0.90), so it is dominated."""
    points = [(10, 0.90, "a"), (20, 0.80, "b"), (30, 0.95, "c")]
    assert [p[2] for p in pareto_front(points)] == ["a", "c"]


def test_front_keeps_a_cheaper_and_better_point():
    points = [(5, 0.99, "cheap-good"), (50, 0.90, "slow-worse")]
    assert [p[2] for p in pareto_front(points)] == ["cheap-good"]


def test_front_is_monotonic_in_accuracy():
    points = [(1, 0.5, "a"), (2, 0.7, "b"), (3, 0.6, "c"), (4, 0.9, "d")]
    front = pareto_front(points)
    accuracies = [a for _, a, _ in front]
    assert accuracies == sorted(accuracies)
    assert "c" not in [p[2] for p in front]


def test_front_of_one_point():
    assert pareto_front([(1, 0.5, "only")]) == [(1, 0.5, "only")]


def test_front_of_nothing():
    assert pareto_front([]) == []


def test_mean_ignores_missing_values():
    """Pixel metrics are None for categories without masks; they must not drag
    an average to NaN."""
    assert mean([1.0, None, 3.0]) == 2.0


def test_mean_of_all_missing_is_nan():
    assert mean([None, None]) != mean([None, None])   # NaN
