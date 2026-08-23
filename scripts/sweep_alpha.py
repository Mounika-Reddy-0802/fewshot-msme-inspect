"""Sweep the adapter's alpha, per shot count.

    python scripts/sweep_alpha.py
    python scripts/sweep_alpha.py --alphas 0 0.1 0.2 0.3 0.5 --shots 1 5 10

alpha mixes the cache branch with the text branch:

    anomaly = (1 - alpha) * cache + alpha * text

A single alpha for every shot count is almost certainly wrong. Text prompts are
the only evidence available when the cache is empty, and near-useless once the
cache holds ten images of the part. This measures where the crossover actually
is instead of guessing, and reports the best alpha per shot count.

Runs entirely off cached feature banks, so it is cheap: no images are re-encoded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mvtec import complete_categories  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.eval.features import FeatureBank, bank_path  # noqa: E402
from src.eval.pipeline import evaluate_split  # noqa: E402
from src.model.adapter import AdapterConfig  # noqa: E402
from src.model.encoder import FrozenCLIPEncoder  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import dataset_dir, repo_root, results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="adapter.yaml")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    backbone = config["backbone"]["name"]
    root = dataset_dir(config["dataset"]["name"])
    splits_dir = repo_root() / "results" / "manifests" / "splits"

    categories = args.categories or complete_categories(root)
    banks = {}
    for category in categories:
        path = bank_path(results_root(), backbone, category)
        if path.exists():
            banks[category] = FeatureBank.load(path)
    if not banks:
        print("no cached feature banks; run scripts/run_benchmark.py first", file=sys.stderr)
        return 1

    encoder = FrozenCLIPEncoder(backbone, config["backbone"]["pretrained"], device="cpu")
    base = config["model"]["adapter"]

    print(f"backbone {backbone}   categories {len(banks)}   "
          f"alphas {args.alphas}   shots {args.shots}\n")

    rows = []
    scores = defaultdict(list)          # (shots, alpha) -> image aurocs
    for category, bank in sorted(banks.items()):
        for shots in args.shots:
            for seed in args.seeds:
                manifest_path = splits_dir / f"mvtec_ad_{category}_{shots}shot_seed{seed}.json"
                if not manifest_path.exists():
                    continue
                manifest = load_split(manifest_path)
                for alpha in args.alphas:
                    cfg = AdapterConfig(
                        alpha=alpha,
                        beta=base["beta"],
                        top_k=base["top_k"],
                        trainable=False,
                    )
                    result = evaluate_split(
                        manifest, root, bank, encoder=encoder,
                        top_k=base["top_k"], method="adapter", adapter_config=cfg,
                        temperature=config["model"]["scoring"]["temperature"],
                        image_size=tuple(config["dataset"]["image_size"]),
                    )
                    row = result.as_row()
                    row["alpha"] = alpha
                    rows.append(row)
                    scores[(shots, alpha)].append(row["image_auroc"])
        print(f"  {category} done", flush=True)

    print(f"\n{'shots':<7}" + "".join(f"a={a:<7.2f}" for a in args.alphas) + "  best")
    best_per_shot = {}
    for shots in args.shots:
        cells, best_alpha, best_value = [], None, -1.0
        for alpha in args.alphas:
            values = scores[(shots, alpha)]
            mean = sum(values) / len(values) if values else float("nan")
            cells.append(f"{mean:<9.3f}")
            if values and mean > best_value:
                best_alpha, best_value = alpha, mean
        best_per_shot[shots] = {"alpha": best_alpha, "image_auroc": round(best_value, 4)}
        print(f"{shots:<7}" + "".join(cells) + f"  alpha={best_alpha} ({best_value:.3f})")

    print("\nalpha=0 is cache only, alpha=1 is prompts only.")

    out = Path(args.out) if args.out else results_root() / "sweeps" / "alpha.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (repo_root() / "results" / "manifests" / "alpha_sweep.json").write_text(
        json.dumps({"backbone": backbone, "alphas": args.alphas, "shots": args.shots,
                    "seeds": args.seeds, "categories": sorted(banks),
                    "best_per_shot": best_per_shot}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
