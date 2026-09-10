"""Prototype memory tests.

The append-only claim is a headline contribution, so it is asserted directly:
adding a class must leave every existing class's scores bit-identical.
"""

import pytest
import torch

from src.model.encoder import l2_normalise
from src.model.memory import PrototypeMemory

GRID = (4, 4)
P = GRID[0] * GRID[1]
D = 8


def _patches(n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return l2_normalise(torch.randn(n, P, D, generator=g))


@pytest.fixture
def memory():
    m = PrototypeMemory(dim=D, grid=GRID)
    m.add_class("bottle", _patches(3, seed=1))
    return m


def test_add_class_stores_every_patch(memory):
    assert memory.classes == ["bottle"]
    assert memory.size("bottle") == 3 * P


def test_adding_shots_later_appends(memory):
    memory.add_class("bottle", _patches(2, seed=2))
    assert memory.size("bottle") == 5 * P
    assert memory.classes == ["bottle"]


def test_accepts_already_flat_prototypes():
    m = PrototypeMemory(dim=D, grid=GRID)
    m.add_class("x", torch.randn(10, D))
    assert m.size("x") == 10


def test_rejects_wrong_dimension():
    m = PrototypeMemory(dim=D, grid=GRID)
    with pytest.raises(ValueError, match="expected dim"):
        m.add_class("x", torch.randn(2, P, D + 1))


def test_scores_have_the_right_shapes(memory):
    scores = memory.score(_patches(5, seed=9), "bottle")
    assert scores.image.shape == (5,)
    assert scores.patches.shape == (5, *GRID)


def test_memorised_patches_score_as_normal():
    """A query identical to something in memory is a perfect match, so its
    anomaly score is ~0. This is what makes 1 - similarity meaningful."""
    support = _patches(2, seed=3)
    m = PrototypeMemory(dim=D, grid=GRID)
    m.add_class("bottle", support)
    scores = m.score(support, "bottle")
    assert torch.allclose(scores.image, torch.zeros(2), atol=1e-5)


def test_unseen_patterns_score_higher_than_memorised_ones():
    m = PrototypeMemory(dim=D, grid=GRID)
    support = _patches(4, seed=4)
    m.add_class("bottle", support)
    novel = l2_normalise(torch.randn(4, P, D, generator=torch.Generator().manual_seed(99)))
    assert m.score(novel, "bottle").image.mean() > m.score(support, "bottle").image.mean()


def test_unknown_class_is_an_error(memory):
    with pytest.raises(KeyError, match="cable"):
        memory.score(_patches(1), "cable")


def test_top_k_averages_more_neighbours():
    support = _patches(4, seed=5)
    query = _patches(2, seed=6)
    one = PrototypeMemory(dim=D, grid=GRID, top_k=1)
    five = PrototypeMemory(dim=D, grid=GRID, top_k=5)
    for m in (one, five):
        m.add_class("bottle", support)
    # averaging over more neighbours can only lower similarity, so raise anomaly
    assert (five.score(query, "bottle").image >= one.score(query, "bottle").image - 1e-6).all()


def test_top_k_must_be_positive():
    with pytest.raises(ValueError, match="top_k"):
        PrototypeMemory(dim=D, grid=GRID, top_k=0)


# --- the append-only claim --------------------------------------------------

def test_adding_a_class_does_not_disturb_existing_scores(memory):
    """The headline claim. Scores for `bottle` must be bit-identical after a
    completely unrelated class is appended -- not merely close."""
    query = _patches(4, seed=7)
    before = memory.score(query, "bottle")

    memory.add_class("cable", _patches(6, seed=8))
    after = memory.score(query, "bottle")

    assert torch.equal(before.image, after.image)
    assert torch.equal(before.patches, after.patches)


def test_adding_a_class_does_not_touch_existing_prototypes(memory):
    original = memory._banks["bottle"].clone()
    memory.add_class("cable", _patches(6, seed=8))
    assert torch.equal(memory._banks["bottle"], original)


def test_classes_accumulate_without_bound(memory):
    for i in range(5):
        memory.add_class(f"part_{i}", _patches(1, seed=20 + i))
    assert len(memory) == 6
    assert memory.total_prototypes() == 3 * P + 5 * P


def test_cost_of_a_class_is_reported(memory):
    before = memory.memory_bytes()
    memory.add_class("cable", _patches(3, seed=8))
    # float32 x D dims x (3 images x P patches)
    assert memory.memory_bytes() - before == 3 * P * D * 4


# --- persistence ------------------------------------------------------------

def test_round_trip_through_disk(memory, tmp_path):
    memory.add_class("cable", _patches(2, seed=11))
    query = _patches(3, seed=12)
    before = memory.score(query, "bottle")

    path = memory.save(tmp_path / "memory.pt")
    loaded = PrototypeMemory.load(path)

    assert loaded.classes == memory.classes
    assert loaded.total_prototypes() == memory.total_prototypes()
    assert torch.equal(loaded.score(query, "bottle").image, before.image)


def test_save_writes_a_readable_summary(memory, tmp_path):
    import json

    path = memory.save(tmp_path / "memory.pt")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["classes"] == {"bottle": 3 * P}
    assert meta["total_prototypes"] == 3 * P
