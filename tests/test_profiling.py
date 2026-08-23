"""Profiling tests.

These check the measurement machinery, not the speed of any particular machine.
Asserting on wall-clock would make the suite flaky the moment CI is busy.
"""

import time

import pytest
import torch

from src.eval.profiling import (
    LatencyStats,
    Profile,
    count_flops,
    describe_machine,
    measure_latency,
    measure_peak_ram,
    pin_threads,
)


def test_pin_threads_reports_what_it_set():
    original = torch.get_num_threads()
    try:
        assert pin_threads(2) == 2
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(original)


def test_pin_threads_leaves_the_default_alone():
    assert pin_threads(None) == torch.get_num_threads()


def test_latency_measures_a_known_sleep():
    stats = measure_latency(lambda: time.sleep(0.01), batch_size=1, warmup=1, iters=5)
    assert stats.median_ms == pytest.approx(10, abs=8)
    assert stats.iters == 5


def test_per_image_divides_by_batch_size():
    stats = measure_latency(lambda: time.sleep(0.01), batch_size=4, warmup=0, iters=3)
    assert stats.per_image_ms == pytest.approx(stats.median_ms / 4)


def test_p90_is_at_least_the_median():
    stats = measure_latency(lambda: time.sleep(0.001), batch_size=1, warmup=1, iters=10)
    assert stats.p90_ms >= stats.median_ms


def test_warmup_passes_are_not_timed():
    """The first passes pay for lazy allocation; including them would inflate
    every number."""
    calls = []
    measure_latency(lambda: calls.append(1), batch_size=1, warmup=5, iters=10)
    assert len(calls) == 15


def test_peak_ram_is_a_number():
    value = measure_peak_ram(lambda: torch.zeros(256, 256))
    assert value == value  # not NaN when psutil is installed


def test_flops_counts_a_linear_layer():
    """A Linear(10, 20) does 10*20 multiply-accumulates."""
    model = torch.nn.Linear(10, 20, bias=False)
    gflops = count_flops(model, torch.zeros(1, 10))
    assert gflops == pytest.approx(200 / 1e9, rel=1e-6)


def test_flops_returns_none_rather_than_raising():
    class Odd(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("nope")

    assert count_flops(Odd(), torch.zeros(1, 3)) is None


def test_machine_description_records_the_thread_count():
    info = describe_machine()
    assert info["torch_threads"] == torch.get_num_threads()
    assert "torch" in info and "platform" in info


def test_profile_row_is_flat_and_named_by_batch_size():
    profile = Profile(
        backbone="ViT-test",
        threads=4,
        parameters=1000,
        gflops=1.5,
        peak_ram_mb=12.0,
        latency=[
            LatencyStats(1, 10.0, 12.0, 10.0, 20),
            LatencyStats(8, 40.0, 44.0, 5.0, 20),
        ],
    )
    row = profile.as_row()
    assert row["latency_bs1_per_image_ms"] == 10.0
    assert row["latency_bs8_per_image_ms"] == 5.0
    assert all(not isinstance(v, (list, dict)) for v in row.values())


def test_memory_search_latency_grows_with_the_bank():
    """Inference cost is not the backbone alone -- the search scales with stored
    prototypes, so a 10-shot system is not free relative to a 1-shot one."""
    from src.eval.profiling import measure_memory_latency

    small = measure_memory_latency(dim=64, num_patches=49, prototypes=49, warmup=1, iters=5)
    large = measure_memory_latency(dim=64, num_patches=49, prototypes=4900, warmup=1, iters=5)
    assert small > 0 and large > 0
    assert large > small


def test_memory_search_latency_is_milliseconds():
    from src.eval.profiling import measure_memory_latency

    ms = measure_memory_latency(dim=64, num_patches=49, prototypes=490, warmup=1, iters=5)
    assert 0 < ms < 1000, "a single small search should be well under a second"
