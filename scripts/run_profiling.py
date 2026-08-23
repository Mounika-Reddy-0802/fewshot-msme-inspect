"""Measure CPU cost for each backbone.

    python scripts/run_profiling.py --threads 14
    python scripts/run_profiling.py --backbones ViT-B-16-quickgelu ViT-B-32-quickgelu

Run this on an otherwise idle machine. The numbers are the point of the project,
and a background download or a test suite running alongside will inflate them.
The machine's CPU load at start is recorded next to the results so an implausible
row can be traced rather than argued about.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.profiling import describe_machine, pin_threads, profile_encoder  # noqa: E402
from src.model.encoder import FrozenCLIPEncoder  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import results_root  # noqa: E402

DEFAULT_BACKBONES = (
    "ViT-B-32-quickgelu",
    "ViT-B-16-quickgelu",
    "ViT-L-14-quickgelu",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="prototype.yaml")
    parser.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    pretrained = config["backbone"]["pretrained"]
    warmup = args.warmup or config["profiling"]["warmup_iters"]
    iters = args.iters or config["profiling"]["measure_iters"]

    threads = pin_threads(args.threads)
    machine = describe_machine()

    print(f"threads    {threads}")
    print(f"cpu        {machine.get('processor', '?')}")
    load = machine.get("cpu_percent_before")
    if load is not None:
        print(f"cpu load   {load}%" + ("   <-- HIGH, timings will be inflated" if load > 25 else ""))
    print(f"warmup {warmup}, iters {iters}, batch sizes {args.batch_sizes}\n")

    rows, failed = [], []
    for name in args.backbones:
        print(f"{name} ...", flush=True)
        try:
            encoder = FrozenCLIPEncoder(name, pretrained, device="cpu")
        except Exception as exc:
            print(f"  skipped: {type(exc).__name__} (weights not available?)")
            failed.append(name)
            continue

        profile = profile_encoder(
            encoder,
            batch_sizes=tuple(args.batch_sizes),
            warmup=warmup,
            iters=iters,
            threads=threads,
        )
        row = profile.as_row()
        row["grid"] = f"{encoder.grid_size[0]}x{encoder.grid_size[1]}"
        row["embed_dim"] = encoder.embed_dim
        rows.append(row)

        gflops = f"{row['gflops']:.1f}" if row["gflops"] else "n/a"
        print(
            f"  params {row['parameters'] / 1e6:.0f}M  {gflops} GFLOPs  "
            f"bs1 {row['latency_bs1_per_image_ms']:.0f} ms/img  "
            f"RAM {row['peak_ram_mb']:.0f} MB"
        )

    if not rows:
        print("nothing profiled", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else results_root() / "profiling" / "backbones.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({k for r in rows for k in r})
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_path.with_suffix(".meta.json").write_text(
        json.dumps({"machine": machine, "warmup": warmup, "iters": iters,
                    "batch_sizes": args.batch_sizes, "skipped": failed}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n{len(rows)} backbones -> {out_path}")
    if failed:
        print(f"skipped (weights unavailable): {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
