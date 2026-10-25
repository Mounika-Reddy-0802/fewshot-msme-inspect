"""Which transformer blocks carry the useful patch features? (release 1.1)

    python scripts/run_layer_ablation.py --dataset mvtec_ad --blocks 6
    python scripts/run_layer_ablation.py --dataset mvtec_ad --blocks 9

Reads the cached multi-layer feature bank written by
``run_benchmark.py --layers 6 9 12`` and scores with any subset of its blocks,
without re-encoding a single image. That works because the multi-layer patch
vector is the l2-normalised block tokens side by side (each divided by sqrt(3)),
so one block's slice, renormalised, is exactly that block's feature. The output
CSV has the run_benchmark format, so compare_paired.py pairs it with anything.

Block 12 alone is the v1.0 final-layer feature and needs no run here.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data.splits import categories as manifest_categories  # noqa: E402
from src.data.splits import category_root, load_manifest, manifest_paths, split_for  # noqa: E402
from src.eval.features import FeatureBank, bank_path  # noqa: E402
from src.eval.pipeline import evaluate_split  # noqa: E402
from src.model.encoder import l2_normalise  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import dataset_dir, repo_root, results_root  # noqa: E402

LAYERS = (6, 9, 12)
WIDTH = 768


class BlockView:
    """A FeatureBank restricted to some of its blocks, renormalised."""

    def __init__(self, bank: FeatureBank, blocks: tuple[int, ...], name: str) -> None:
        self.bank = bank
        self.idx = torch.cat([torch.arange(LAYERS.index(b) * WIDTH, (LAYERS.index(b) + 1) * WIDTH)
                              for b in blocks])
        self.dim = len(self.idx)
        self.grid = bank.grid
        self.backbone = name

    def patches(self, paths):
        return l2_normalise(self.bank.patches(paths)[..., self.idx].float())

    def images(self, paths):
        return self.bank.images(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mvtec_ad")
    parser.add_argument("--blocks", nargs="+", type=int, required=True)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    blocks = tuple(args.blocks)
    config = load_config("prototype.yaml")
    data_root = dataset_dir(args.dataset).parent
    root = category_root(args.dataset, data_root)
    manifests = [load_manifest(p) for p in
                 manifest_paths(repo_root() / "results" / "manifests" / "splits", args.dataset)]
    source = "ViT-B-16-quickgelu-ml6-9-12"
    name = f"ViT-B-16-quickgelu-ml{'-'.join(map(str, blocks))}"
    out = results_root() / "benchmark" / f"{args.dataset}_{name}-sliced.csv"

    rows, started = [], time.perf_counter()
    for category in manifest_categories(manifests[0]):
        path = bank_path(results_root(), source, category, WIDTH * len(LAYERS))
        if not path.exists():
            print(f"{category}: no multi-layer bank; run run_benchmark.py --layers 6 9 12 first")
            continue
        view = BlockView(FeatureBank.load(path), blocks, name)
        for manifest in manifests:
            for k in (1, 5, 10):
                split = split_for(manifest, category, k, data_root)
                result = evaluate_split(split, root, view,
                                        top_k=config["model"]["scoring"]["top_k"],
                                        image_size=tuple(config["dataset"]["image_size"]))
                rows.append(result.as_row())
        done = [r for r in rows if r["category"] == category]
        print(f"{category:<16} " + "  ".join(
            f"{k}-shot {sum(r['image_auroc'] for r in done if r['shots'] == k) / 3:.3f}"
            for k in (1, 5, 10)), flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {out}  ({(time.perf_counter() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
