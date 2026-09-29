"""The Rs 0 feasibility table: what running this method actually costs.

    python scripts/cost_table.py
    python scripts/cost_table.py --watts 45 --tariff 8 --parts-per-day 10000 --gpu-hours 0

Measured inputs, read from the committed run records:

* **per-image latency** -- ONNX FP32/INT8 and torch FP32, 6 threads, batch size 1
  (int8_latency.json, torch_latency.json), and torch after 10 minutes of
  continuous load (sustained_torch.json);
* **RAM** -- peak working set of the loaded inspector (peak_ram.json);
* **compute spent on the study** -- the wall-clock of every benchmark run, from
  the .meta.json files next to each benchmark CSV. A lower bound: sweeps,
  ablations and aborted runs are not included.

Stated assumptions, each overridable, each printed next to the result:

* ``--watts`` -- CPU power while inferring. Default 45 W, the i7-13700H's base
  power; the whole laptop draws more, so this understates energy per image.
* ``--tariff`` -- Rs per kWh. Default 8, a typical Indian commercial tariff.
* ``--gpu-hours`` -- free-tier GPU hours used (Kaggle/Colab). Cost Rs 0 either
  way; recorded because the Blueprint asks for it.
* ``--commercial-inr`` -- the price of a commercial machine-vision licence, if a
  cited quote is available. Left out rather than guessed.

Capacity (busy hours per day) is planned on the *sustained* latency where it was
measured: the laptop throttles to 5-6x its burst latency after a few minutes.
Hardware spend is Rs 0 by construction: the laptop already existed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import repo_root, results_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watts", type=float, default=45.0)
    parser.add_argument("--tariff", type=float, default=8.0, help="Rs per kWh")
    parser.add_argument("--parts-per-day", type=int, default=10_000)
    parser.add_argument("--gpu-hours", type=float, default=0.0)
    parser.add_argument("--commercial-inr", type=float, default=None)
    args = parser.parse_args()

    manifests = repo_root() / "results" / "manifests"

    def load(name):
        return json.loads((manifests / name).read_text(encoding="utf-8"))

    onnx = load("int8_latency.json")["backbones"]
    torch_rows = {r["backbone"]: r for r in load("torch_latency.json")["rows"]}
    sustained = load("sustained_torch.json")["backbones"]
    ram = {(r["backbone"], r["engine"]): r["inspector_mb"] for r in load("peak_ram.json")["rows"]}

    runs = {}
    for meta in sorted((results_root() / "benchmark").glob("*.meta.json")):
        runs[meta.name.removesuffix(".meta.json")] = json.loads(
            meta.read_text(encoding="utf-8"))["total_seconds"]
    study_hours = sum(runs.values()) / 3600

    def row(config, ms, ram_mb, sustained_ms=None):
        inr_per_image = args.watts * ms / 1000 / 3.6e6 * args.tariff
        worst = sustained_ms or ms
        return {
            "config": config,
            "ms_per_image": ms,
            "sustained_ms_per_image": sustained_ms,
            "images_per_hour": int(3600_000 / ms),
            "inr_per_image": inr_per_image,
            "inr_per_year": inr_per_image * args.parts_per_day * 300,
            "hours_per_day_busy_worst": args.parts_per_day * worst / 3.6e6,
            "ram_mb": ram_mb,
        }

    rows = []
    for backbone, rec in onnx.items():
        short = backbone.replace("-quickgelu", "")
        for precision, key in (("fp32", "fp32_ms_bs1_runs"), ("int8", "int8_ms_bs1_runs")):
            ms = sorted(rec[key])[len(rec[key]) // 2]            # the higher of two runs
            rows.append(row(f"{short} ONNX {precision.upper()}", ms,
                            ram.get((backbone, f"onnx-{precision}"))))
    for backbone, rec in torch_rows.items():
        rows.append(row(f"{backbone.replace('-quickgelu', '')} torch FP32",
                        rec["latency_bs1_median_ms"], ram.get((backbone, "torch")),
                        sustained.get(backbone, {}).get("last_window_ms")))

    record = {
        "assumptions": {"cpu_watts": args.watts, "tariff_inr_per_kwh": args.tariff,
                        "parts_per_day": args.parts_per_day, "working_days_per_year": 300},
        "hardware_spend_inr": 0,
        "free_tier_gpu_hours": args.gpu_hours,
        "study_benchmark_cpu_hours_lower_bound": round(study_hours, 2),
        "study_benchmark_runs": {k: round(v / 60, 1) for k, v in runs.items()},
        "study_electricity_inr_lower_bound": round(study_hours * args.watts / 1000 * args.tariff, 2),
        "commercial_licence_inr": args.commercial_inr,
        "per_config": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
                       for r in rows],
    }

    print(f"assumptions: {args.watts:g} W CPU, Rs {args.tariff:g}/kWh, "
          f"{args.parts_per_day:,} parts/day, 300 days/yr\n")
    print("| config | ms/image (burst) | ms/image (sustained) | Rs per year | "
          "busy hours/day (worst) | RAM MB |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        sus = f"{r['sustained_ms_per_image']:.0f}" if r["sustained_ms_per_image"] else "-"
        ram_mb = f"{r['ram_mb']:.0f}" if r["ram_mb"] else "-"
        print(f"| {r['config']} | {r['ms_per_image']:.0f} | {sus} | {r['inr_per_year']:.2f} | "
              f"{r['hours_per_day_busy_worst']:.2f} | {ram_mb} |")
    print(f"\nhardware Rs 0 | free-tier GPU hours {args.gpu_hours:g} | "
          f"benchmark CPU time for the whole study >= {study_hours:.1f} h "
          f"(~Rs {record['study_electricity_inr_lower_bound']:.2f} of electricity)")
    if args.commercial_inr is None:
        print("commercial licence: not included -- pass --commercial-inr with a cited quote")

    out = manifests / "cost_feasibility.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
