"""Image vs text vs fused prototypes, at the image level (Blueprint ablation).

    python scripts/run_text_fusion.py
    python scripts/run_text_fusion.py --datasets visa --backbone ViT-B-32-quickgelu

Three sources of evidence for "is this part defective":

* **image** -- the prototype memory's score (nearest good patch), w = 0;
* **text** -- CLIP's pooled image embedding against prompt ensembles for a normal
  and a defective <object>, w = 1. This is zero-shot: it ignores the support set;
* **fused** -- (1 - w) * image + w * text for w in between.

Only image-level AUROC is affected: text cannot score a 768-dim patch, so the
anomaly maps (and pixel AUROC) are the memory's in every row.

The weights are a fixed grid reported in full, not a value picked on test data.
Everything runs on cached feature banks; only the prompts need the encoder.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data.splits import categories, load_manifest, manifest_paths, split_for  # noqa: E402
from src.eval.features import FeatureBank, bank_path  # noqa: E402
from src.eval.metrics import image_auroc  # noqa: E402
from src.eval.pipeline import category_text_embeddings  # noqa: E402
from src.model.encoder import FrozenCLIPEncoder  # noqa: E402
from src.model.memory import PrototypeMemory  # noqa: E402
from src.model.windows import text_image_scores  # noqa: E402
from src.utils.paths import dataset_dir, repo_root, results_root  # noqa: E402

WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)

# What the prompt calls each product. Category folder names are not always
# words ("pcb3", "macaroni1"); a workshop would type the plain name.
OBJECT_NAMES = {
    "pcb1": "printed circuit board", "pcb2": "printed circuit board",
    "pcb3": "printed circuit board", "pcb4": "printed circuit board",
    "macaroni1": "macaroni", "macaroni2": "macaroni",
    "chewinggum": "chewing gum", "pipe_fryum": "pipe fryum", "metal_nut": "metal nut",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mvtec_ad", "visa"])
    parser.add_argument("--backbone", default="ViT-B-16-quickgelu")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    encoder = FrozenCLIPEncoder(args.backbone, "openai", device="cpu", project_patches=False)
    record = {"backbone": args.backbone, "weights": list(WEIGHTS),
              "temperature": args.temperature, "datasets": {}}

    for dataset in args.datasets:
        root = dataset_dir(dataset).parent
        manifests = [load_manifest(p) for p in
                     manifest_paths(repo_root() / "results" / "manifests" / "splits", dataset)]
        # (shots, w) -> list over (category, seed) of AUROC; paired by position
        auroc = defaultdict(list)
        for category in categories(manifests[0]):
            bank = FeatureBank.load(bank_path(results_root(), args.backbone, category,
                                              encoder.embed_dim))
            text = category_text_embeddings(encoder, OBJECT_NAMES.get(category, category))
            for manifest in manifests:
                for shots in (1, 5, 10):
                    split = split_for(manifest, category, shots, root)
                    paths = [t["image"] for t in split["test"]]
                    labels = [t["label"] for t in split["test"]]
                    memory = PrototypeMemory(bank.dim, bank.grid)
                    memory.add_class(category, bank.patches(split["support"]))
                    image = memory.score(bank.patches(paths), category).image
                    zero = text_image_scores(bank.images(paths), text, args.temperature)
                    for w in WEIGHTS:
                        fused = (1 - w) * image + w * zero
                        auroc[(shots, w)].append(image_auroc(labels, fused.numpy()))

        rows = {}
        for shots in (1, 5, 10):
            base = auroc[(shots, 0.0)]
            rows[str(shots)] = {}
            for w in WEIGHTS:
                vals = auroc[(shots, w)]
                deltas = [a - b for a, b in zip(vals, base)]
                se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if w else 0.0
                rows[str(shots)][str(w)] = {
                    "mean_image_auroc": round(statistics.fmean(vals), 4),
                    "delta_vs_image_only": round(statistics.fmean(deltas), 4),
                    "effect_over_se": round(abs(statistics.fmean(deltas)) / se, 1) if se else None,
                }
        record["datasets"][dataset] = {"n_pairs_per_cell": len(auroc[(1, 0.0)]), "per_shot": rows}

        print(f"\n{dataset}   image AUROC by text weight w (0 = memory only, 1 = text only)")
        print("shots  " + "".join(f"w={w:<11}" for w in WEIGHTS))
        for shots in ("1", "5", "10"):
            cells = []
            for w in WEIGHTS:
                r = rows[shots][str(w)]
                tag = "" if not w else f"({r['delta_vs_image_only']:+.3f})"
                cells.append(f"{r['mean_image_auroc']:.3f}{tag:<9}")
            print(f"{shots:>5}  " + " ".join(cells))

    out = repo_root() / "results" / "manifests" / "text_fusion.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
