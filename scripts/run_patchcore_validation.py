"""Week-1 environment check: PatchCore on one MVTec AD category.

Not a contribution, a sanity check. If anomalib is wired up correctly,
PatchCore on a single category should land in the range the paper and the
anomalib benchmark report. If it does not, the environment is wrong and every
number we produce later is suspect.

    python scripts/run_patchcore_validation.py --category bottle

Reads configs/baseline.yaml, so backbone, layers, coreset ratio and neighbour
count are versioned rather than typed at the prompt.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

# anomalib benchmark / PatchCore paper figures for this category, for comparison
PUBLISHED = {
    "bottle": {"image_auroc": (0.995, 1.000), "pixel_auroc": (0.975, 0.990)},
    "cable": {"image_auroc": (0.980, 0.998), "pixel_auroc": (0.965, 0.985)},
    "capsule": {"image_auroc": (0.975, 0.995), "pixel_auroc": (0.980, 0.992)},
    "hazelnut": {"image_auroc": (0.995, 1.000), "pixel_auroc": (0.980, 0.990)},
}


def _flatten(results: list[dict] | dict) -> dict:
    """anomalib returns a one-element list of metric dicts from engine.test."""
    if isinstance(results, list):
        merged: dict = {}
        for entry in results:
            merged.update(entry)
        return merged
    return dict(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline.yaml")
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    category = args.category or config["dataset"]["category"]
    seed = config["experiment"]["seed"]
    patchcore_config = config["model"]["patchcore"]

    seed_everything(seed)

    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from lightning.pytorch import seed_everything as lightning_seed

    lightning_seed(seed, workers=True)

    datamodule = MVTecAD(
        root=config["dataset"]["root"],
        category=category,
        train_batch_size=config["eval"]["batch_size"],
        eval_batch_size=config["eval"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )

    model = Patchcore(
        backbone=patchcore_config["backbone"],
        layers=patchcore_config["layers"],
        coreset_sampling_ratio=patchcore_config["coreset_sampling_ratio"],
        num_neighbors=patchcore_config["num_neighbors"],
    )

    run_dir = results_root() / "validation" / "patchcore"
    engine = Engine(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        default_root_dir=str(run_dir),
    )

    started = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    metrics = _flatten(engine.test(model=model, datamodule=datamodule))
    test_seconds = time.perf_counter() - started

    record = {
        "category": category,
        "seed": seed,
        "config": args.config,
        "patchcore": patchcore_config,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "fit_seconds": round(fit_seconds, 1),
        "test_seconds": round(test_seconds, 1),
        "device": "cpu",
        "machine": f"{platform.processor()} | python {platform.python_version()}",
    }

    print("\n=== patchcore validation ===")
    for key, value in record["metrics"].items():
        print(f"  {key:<28} {value}")
    print(f"  fit {fit_seconds:.1f}s, test {test_seconds:.1f}s")

    expected = PUBLISHED.get(category)
    if expected:
        print("\n  published range (anomalib benchmark / patchcore paper):")
        for metric, (low, high) in expected.items():
            got = next(
                (v for k, v in record["metrics"].items() if k.lower().replace("_", "") == metric.replace("_", "")),
                None,
            )
            verdict = "?" if got is None else ("in range" if low <= got <= high else "OUT OF RANGE")
            print(f"    {metric:<14} {low:.3f}-{high:.3f}   got {got}   {verdict}")
        record["published_range"] = {k: list(v) for k, v in expected.items()}

    out_dir = results_root() / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"patchcore_{category}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n  written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
