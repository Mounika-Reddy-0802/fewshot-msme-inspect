"""CPU cost measurement.

The accuracy numbers are only half the claim; the other half is what they cost on
hardware an MSME already owns. This measures four things per backbone:

* **latency** -- wall-clock per image, at batch size 1 (a workshop inspects one
  part at a time) and batched (what a lab would quote). Reported as median plus
  p90, because a mean hides the stalls that a shop floor actually notices.
* **peak RAM** -- resident-set growth across the forward pass.
* **FLOPs** -- hardware-independent, so a reader can predict cost on a machine we
  never tested.
* **parameters** -- model size on disk.

Three things that quietly ruin CPU timings, all handled here:

1. **Thread count.** torch picks a default from the machine. Two runs on
   different machines, or with another job running, are not comparable unless it
   is pinned. :func:`pin_threads` records what was used.
2. **Warmup.** The first passes pay for lazy allocation and cache misses. They
   are discarded.
3. **Background load.** Nothing else should be running. This cannot be enforced
   from inside, so :func:`describe_machine` records the load alongside the
   numbers, and the caller is expected to check it.
"""

from __future__ import annotations

import gc
import platform
import statistics
import time
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class LatencyStats:
    batch_size: int
    median_ms: float
    p90_ms: float
    per_image_ms: float
    iters: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Profile:
    backbone: str
    threads: int
    parameters: int
    gflops: float | None
    peak_ram_mb: float
    latency: list[LatencyStats]

    def as_row(self) -> dict:
        row = {
            "backbone": self.backbone,
            "threads": self.threads,
            "parameters": self.parameters,
            "gflops": self.gflops,
            "peak_ram_mb": round(self.peak_ram_mb, 1),
        }
        for stat in self.latency:
            row[f"latency_bs{stat.batch_size}_median_ms"] = round(stat.median_ms, 2)
            row[f"latency_bs{stat.batch_size}_p90_ms"] = round(stat.p90_ms, 2)
            row[f"latency_bs{stat.batch_size}_per_image_ms"] = round(stat.per_image_ms, 2)
        return row


def pin_threads(threads: int | None) -> int:
    """Fix the thread count so timings are comparable between runs."""
    if threads:
        torch.set_num_threads(threads)
    return torch.get_num_threads()


def describe_machine() -> dict:
    """Recorded next to every timing, so a suspicious number can be explained."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        # a busy machine invalidates the latency numbers; record it rather than
        # pretend it did not happen
        info["cpu_percent_before"] = psutil.cpu_percent(interval=0.5)
    except ImportError:
        pass
    return info


def measure_latency(
    forward,
    batch_size: int,
    warmup: int = 5,
    iters: int = 20,
) -> LatencyStats:
    """Time a callable, discarding warmup passes."""
    for _ in range(warmup):
        forward()

    gc.collect()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        forward()
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    median = statistics.median(samples)
    p90 = samples[min(len(samples) - 1, int(0.9 * len(samples)))]
    return LatencyStats(
        batch_size=batch_size,
        median_ms=median,
        p90_ms=p90,
        per_image_ms=median / batch_size,
        iters=iters,
    )


def measure_peak_ram(forward) -> float:
    """Resident-set growth over one forward pass, in MB."""
    try:
        import psutil
    except ImportError:
        return float("nan")

    process = psutil.Process()
    gc.collect()
    before = process.memory_info().rss
    peak = before
    for _ in range(3):
        forward()
        peak = max(peak, process.memory_info().rss)
    return (peak - before) / 1e6


def count_flops(model, example: torch.Tensor) -> float | None:
    """GFLOPs for one forward pass. None if no counter is installed."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None

    import logging

    logging.getLogger("fvcore").setLevel(logging.ERROR)
    try:
        analysis = FlopCountAnalysis(model, example)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        return analysis.total() / 1e9
    except Exception:
        return None


def measure_memory_latency(
    dim: int,
    num_patches: int,
    prototypes: int,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """Median ms to score one image against a memory bank of this size.

    Inference cost is not the backbone alone: every query patch is compared
    against every stored prototype, so the search grows with the number of shots.
    Charging a 10-shot system the same latency as a 1-shot one would understate
    what few-shot actually costs.
    """
    import torch as _torch

    from src.model.memory import PrototypeMemory

    memory = PrototypeMemory(dim=dim, grid=(num_patches, 1))
    memory.add_class("x", _torch.randn(prototypes, dim))
    query = _torch.randn(1, num_patches, dim)

    return measure_latency(
        lambda: memory.score(query, "x"), batch_size=1, warmup=warmup, iters=iters
    ).median_ms


def profile_encoder(
    encoder,
    batch_sizes: tuple[int, ...] = (1, 8),
    warmup: int = 5,
    iters: int = 20,
    threads: int | None = None,
) -> Profile:
    """Full cost profile of one frozen backbone."""
    used_threads = pin_threads(threads)
    height, width = encoder.image_size

    latency = []
    for batch_size in batch_sizes:
        example = torch.zeros(batch_size, 3, height, width)
        latency.append(
            measure_latency(
                lambda: encoder.encode_images(example), batch_size, warmup, iters
            )
        )

    single = torch.zeros(1, 3, height, width)
    return Profile(
        backbone=encoder.name,
        threads=used_threads,
        parameters=encoder.num_parameters(),
        gflops=count_flops(encoder.model.visual, single),
        peak_ram_mb=measure_peak_ram(lambda: encoder.encode_images(single)),
        latency=latency,
    )
