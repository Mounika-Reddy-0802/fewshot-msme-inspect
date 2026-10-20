"""CPU cost of the PatchCore baseline: the "before" numbers (Blueprint wk 2).

Our method is only worth reporting if it is cheaper than what an MSME would
otherwise reach for. This measures PatchCore (anomalib) the way
src/eval/profiling.py measures our encoder -- threads pinned, batch size 1,
5 warm-up passes discarded, median of 20 -- so both sets of numbers can sit in
one table.

Two stages, run as separate processes:

    python scripts/profile_patchcore.py --stage fit --category bottle
    # close everything else and let the laptop cool for ~15 minutes
    python scripts/profile_patchcore.py --stage profile --category bottle

* fit      builds PatchCore as the week-1 validation does (hyperparameters from
           configs/baseline.yaml) and saves its memory bank outside the repo.
           This is the slow part (coreset selection) and it heats the CPU,
           which is why timing is a separate step.
* profile  runs in a FRESH process: builds the model, loads the bank, times the
           forward pass, and reads peak RAM above an idle Python process.
           Measuring RAM in the process that did the fit would report the fit's
           leftovers, not what an inspector needs.

Writes results/manifests/patchcore_cost.json (tracked in git).
Both stages must see the same FEWSHOT_RESULTS_ROOT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.profiling import describe_machine, measure_latency, pin_threads  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import repo_root, results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

# PatchCore's pre-processor resizes to 256 and centre-crops to 224 before the model
IMAGE_SIZE = 224


def bank_path(category: str) -> Path:
    return results_root() / "profiling" / f"patchcore_{category}_bank.pt"


def meta_path(category: str) -> Path:
    return bank_path(category).with_suffix(".json")


def build_model(config: dict):
    from anomalib.models import Patchcore

    pc = config["model"]["patchcore"]
    return Patchcore(
        backbone=pc["backbone"],
        layers=pc["layers"],
        coreset_sampling_ratio=pc["coreset_sampling_ratio"],
        num_neighbors=pc["num_neighbors"],
    )


def stage_fit(args: argparse.Namespace, config: dict) -> int:
    import torch
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from lightning.pytorch import seed_everything as lightning_seed

    seed = config["experiment"]["seed"]
    seed_everything(seed)
    lightning_seed(seed, workers=True)
    threads = pin_threads(args.threads)

    datamodule = MVTecAD(
        root=config["dataset"]["root"],
        category=args.category,
        train_batch_size=config["eval"]["batch_size"],
        eval_batch_size=config["eval"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )
    model = build_model(config)
    engine = Engine(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        default_root_dir=str(results_root() / "profiling" / "patchcore_fit"),
    )

    print(f"fitting PatchCore on {args.category} ({threads} threads) -- this takes a while")
    started = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    fit_seconds = time.perf_counter() - started

    bank = model.model.memory_bank.detach().cpu().clone()
    path = bank_path(args.category)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)

    try:
        train_images = len(datamodule.train_data)
    except Exception:
        train_images = None

    meta = {
        "fit_seconds": round(fit_seconds, 1),
        "train_images": train_images,
        "threads": threads,
        "bank_shape": list(bank.shape),
    }
    meta_path(args.category).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"bank {tuple(bank.shape)} saved to {path}")
    print(f"fit took {fit_seconds / 60:.1f} min")
    print("now let the machine cool, then run --stage profile")
    return 0


def stage_profile(args: argparse.Namespace, config: dict) -> int:
    threads = pin_threads(args.threads)

    # idle check first, before any heavy work -- a busy machine invalidates timings
    machine = describe_machine()
    load = machine.get("cpu_percent_before")
    if load is not None and load > 25:
        print(f"WARNING: background CPU load is {load}% -- close other programs and rerun")

    import psutil
    import torch

    from anomalib.models import Patchcore  # noqa: F401  (import cost belongs to the idle baseline)

    process = psutil.Process()
    try:
        machine["cpu_mhz_at_start"] = round(psutil.cpu_freq().current)
    except Exception:
        pass

    # "idle Python process": libraries imported, no model, no bank
    baseline_rss = process.memory_info().rss

    path = bank_path(args.category)
    if not path.exists():
        print(f"no saved bank at {path} -- run --stage fit first (same FEWSHOT_RESULTS_ROOT)")
        return 1
    bank = torch.load(path)

    model = build_model(config)
    net = model.model
    net.memory_bank = bank
    net.eval()

    # pixel content does not change CPU timing; seeded noise at the model's input size
    example = torch.randn(
        1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=torch.Generator().manual_seed(0)
    )

    def forward():
        with torch.no_grad():
            return net(example)

    stats = measure_latency(forward, batch_size=1, warmup=args.warmup, iters=args.iters)

    mem = process.memory_info()
    peak = getattr(mem, "peak_wset", mem.rss)  # true peak on Windows, last RSS elsewhere
    peak_ram_mb = (peak - baseline_rss) / 1e6

    params = sum(p.numel() for p in net.parameters())
    params_mb = params * 4 / 1e6
    bank_mb = bank.numel() * bank.element_size() / 1e6

    meta = {}
    if meta_path(args.category).exists():
        meta = json.loads(meta_path(args.category).read_text(encoding="utf-8"))

    pc = config["model"]["patchcore"]
    record = {
        "method": "patchcore (anomalib)",
        "category": args.category,
        "patchcore": pc,
        "threads": threads,
        "batch_size": 1,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_median_ms": round(stats.median_ms, 2),
        "latency_p90_ms": round(stats.p90_ms, 2),
        "peak_ram_mb": round(peak_ram_mb, 1),
        "parameters": params,
        "parameters_mb_fp32": round(params_mb, 1),
        "memory_bank_shape": list(bank.shape),
        "memory_bank_mb": round(bank_mb, 1),
        "model_size_mb_fp32": round(params_mb + bank_mb, 1),
        "fit": meta,
        "machine": machine,
        "notes": [
            "latency is the model forward only (feature extraction, nearest-neighbour search, "
            "anomaly map); image decode and resize are excluded, as in the CLIP numbers",
            "input is seeded noise at 224x224; pixel content does not affect CPU timing",
            "peak RAM is the peak working set above an idle Python process with torch and "
            "anomalib imported, measured in a fresh process after loading the model and bank",
            "the memory bank is a 10% coreset of ALL training patches, so cost grows with the "
            "training set size, unlike the few-shot prototype memory",
        ],
    }

    out = Path(args.out) if args.out else repo_root() / "results" / "manifests" / "patchcore_cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"\nPatchCore {args.category}, {threads} threads, batch 1")
    print(f"  latency   median {stats.median_ms:.1f} ms   p90 {stats.p90_ms:.1f} ms")
    print(f"  peak RAM  {peak_ram_mb:.0f} MB above an idle process")
    print(f"  params    {params / 1e6:.1f} M ({params_mb:.0f} MB fp32)")
    print(f"  bank      {tuple(bank.shape)} ({bank_mb:.0f} MB)")
    print(f"  written to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["fit", "profile"])
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default=None, help="defaults to results/manifests/patchcore_cost.json")
    args = parser.parse_args()

    config = load_config("baseline.yaml")
    return stage_fit(args, config) if args.stage == "fit" else stage_profile(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
