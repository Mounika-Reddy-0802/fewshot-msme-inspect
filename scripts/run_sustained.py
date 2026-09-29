"""Latency under sustained load: does the laptop slow down as it heats up?

    python scripts/run_sustained.py --minutes 10 --threads 6
    python scripts/run_sustained.py --engine onnx-int8 --backbones ViT-B-32-quickgelu

run_profiling.py measures a cool machine for a few seconds -- burst latency. An
inspection station runs for hours. An earlier profiling pass taken right after
an hour of benchmarking came out 3.6x slower than the same pass on a cool
machine, so a single latency number would mislead.

This runs back-to-back batch-size-1 inference for a fixed wall-clock duration
and records the median latency per window, plus the CPU clock psutil reports
(on Windows that is the fixed base clock, so latency is the signal to read).
The summary compares the first window with the last: the ratio is how much a
workshop should add to the burst figure for continuous use.

Run it on an otherwise idle machine, plugged in, with OneDrive sync paused.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.eval.profiling import describe_machine, pin_threads  # noqa: E402
from src.model.encoder import FrozenCLIPEncoder  # noqa: E402
from src.model.export import OnnxEncoder, onnx_path  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import repo_root, results_root  # noqa: E402


def build(engine: str, name: str, project_patches: bool, threads: int):
    if engine == "torch":
        return FrozenCLIPEncoder(name, "openai", device="cpu", project_patches=project_patches)
    import open_clip
    cfg = open_clip.get_model_config(name)
    dim = cfg["embed_dim"] if project_patches else cfg["vision_cfg"]["width"]
    return OnnxEncoder(onnx_path(results_root() / "onnx", name, dim, engine.split("-")[1]), threads)


def cpu_mhz() -> float | None:
    try:
        import psutil
        freq = psutil.cpu_freq()
        return round(freq.current, 0) if freq else None
    except Exception:
        return None


def sustain(encoder, minutes: float, window_s: float, warmup: int) -> list[dict]:
    example = torch.zeros(1, 3, *encoder.image_size)
    for _ in range(warmup):
        encoder.encode_images(example)

    windows, samples = [], []
    start = window_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        encoder.encode_images(example)
        now = time.perf_counter()
        samples.append((now - t0) * 1000)
        if now - window_start >= window_s:
            windows.append({
                "t_min": round((now - start) / 60, 2),
                "median_ms": round(statistics.median(samples), 2),
                "p90_ms": round(sorted(samples)[int(0.9 * (len(samples) - 1))], 2),
                "images": len(samples),
                "cpu_mhz": cpu_mhz(),
            })
            print(f"    {windows[-1]['t_min']:5.1f} min  median {windows[-1]['median_ms']:7.1f} ms  "
                  f"p90 {windows[-1]['p90_ms']:7.1f} ms  {windows[-1]['cpu_mhz']} MHz", flush=True)
            samples, window_start = [], now
        if now - start >= minutes * 60:
            return windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", nargs="+",
                        default=["ViT-B-32-quickgelu", "ViT-B-16-quickgelu"])
    parser.add_argument("--engine", default="torch", choices=["torch", "onnx-fp32", "onnx-int8"])
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--window", type=float, default=30, help="seconds per window")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--manifest", default=None,
                        help="file name under results/manifests/ for the summary")
    args = parser.parse_args()

    config = load_config("prototype.yaml")
    threads = pin_threads(args.threads)
    machine = describe_machine()
    print(f"threads {threads}   cpu load before {machine.get('cpu_percent_before')}%")

    out_dir = results_root() / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"engine": args.engine, "threads": threads, "minutes": args.minutes,
               "window_s": args.window, "machine": machine, "backbones": {}}

    for name in args.backbones:
        print(f"\n{name} ({args.engine}), {args.minutes:g} min ...", flush=True)
        encoder = build(args.engine, name, config["backbone"].get("project_patches", True), threads)
        windows = sustain(encoder, args.minutes, args.window, args.warmup)

        path = out_dir / f"sustained_{args.engine}_{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(windows[0]))
            writer.writeheader()
            writer.writerows(windows)

        first, last = windows[0]["median_ms"], windows[-1]["median_ms"]
        summary["backbones"][name] = {
            "first_window_ms": first,
            "last_window_ms": last,
            "worst_window_ms": max(w["median_ms"] for w in windows),
            "slowdown_last_vs_first": round(last / first, 2),
            "images": sum(w["images"] for w in windows),
            "windows": windows,
        }
        print(f"  first {first:.1f} ms -> last {last:.1f} ms  ({last / first:.2f}x)")
        del encoder

    if args.manifest:
        target = repo_root() / "results" / "manifests" / args.manifest
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nmanifest -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
