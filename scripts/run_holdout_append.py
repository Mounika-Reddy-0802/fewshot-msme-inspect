"""Held-out class append versus from-scratch adaptation (Blueprint I2 + I3).

    python scripts/run_holdout_append.py --shots 5
    python scripts/run_holdout_append.py --held-out transistor wood zipper --shots 10

The deployment story: a memory already inspecting 12 products receives three it
has never seen. Each is appended from k good photos, with no retraining. Three
questions, each answered per seed:

1. **Regression** -- do the 12 existing products score bit-identically after the
   appends? (``torch.equal``, not "close enough".)
2. **Appended vs from scratch** -- does an appended product score exactly as it
   would in a memory built with all 15 from the start? For a training-free
   memory the honest expectation is "identical", and this checks it rather than
   assuming it.
3. **Cost** -- wall-clock to append a product end to end on the laptop CPU:
   reading the photos, preprocessing, encoding, storing. The storing step alone
   is ~2 ms; the photos dominate.

The held-out classes default to the three alphabetically last MVTec categories,
a rule fixed before looking at any result. Nothing in this method is fitted to
them: the memory head has no learned parameters, and top_k=1 is the default,
not a tuned value.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.splits import (  # noqa: E402
    categories as manifest_categories,
    load_manifest,
    manifest_paths,
    resolve_split,
    split_for,
)
from src.eval.features import FeatureBank, bank_path  # noqa: E402
from src.eval.metrics import image_auroc  # noqa: E402
from src.model.encoder import FrozenCLIPEncoder  # noqa: E402
from src.model.memory import PrototypeMemory  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import dataset_dir, repo_root, results_root  # noqa: E402


def score_all(memory, banks, splits, names) -> dict[str, torch.Tensor]:
    return {
        n: memory.score(banks[n].patches([t["image"] for t in splits[n]["test"]]), n).image
        for n in names
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="prototype.yaml")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--held-out", nargs="+", default=None)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--out", default="incremental_holdout.json",
                        help="file name under results/manifests/")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    backbone = args.backbone or config["backbone"]["name"]
    project = config["backbone"].get("project_patches", True)
    dataset = config["dataset"]["name"]
    data_root = dataset_dir(dataset).parent
    manifests = [load_manifest(p) for p in
                 manifest_paths(repo_root() / "results" / "manifests" / "splits", dataset)]

    everything = manifest_categories(manifests[0])
    held_out = args.held_out or everything[-3:]
    existing = [c for c in everything if c not in held_out]

    encoder = FrozenCLIPEncoder(backbone, config["backbone"]["pretrained"], device="cpu",
                                project_patches=project)
    banks = {c: FeatureBank.load(bank_path(results_root(), backbone, c, encoder.embed_dim))
             for c in everything}
    top_k = config["model"]["scoring"]["top_k"]

    print(f"{backbone}   {args.shots}-shot   {len(manifests)} seed(s)")
    print(f"existing {len(existing)}   held out {held_out}\n")

    per_seed, timings = [], {c: [] for c in held_out}
    for manifest in manifests:
        seed = manifest["seed"]
        splits = {c: split_for(manifest, c, args.shots, data_root) for c in everything}

        def fresh():
            return PrototypeMemory(encoder.embed_dim, encoder.grid_size, top_k=top_k)

        # the deployed memory: 12 products
        deployed = fresh()
        for c in existing:
            deployed.add_class(c, banks[c].patches(splits[c]["support"]))
        before = score_all(deployed, banks, splits, existing)

        # from scratch: all 15 in one build
        scratch = fresh()
        for c in everything:
            scratch.add_class(c, banks[c].patches(splits[c]["support"]))
        scratch_scores = score_all(scratch, banks, splits, held_out)

        # append the held-out products one at a time
        for c in held_out:
            deployed.add_class(c, banks[c].patches(splits[c]["support"]))

            # end-to-end cost, from photos on disk, into a throwaway memory so the
            # accuracy comparison above stays on the cached (identical) features
            support_paths, _ = resolve_split(splits[c], data_root)
            start = time.perf_counter()
            images = [encoder.preprocess(Image.open(p)) for p in support_paths]
            patches = encoder.encode_images(torch.stack(images)).patches
            fresh().add_class(c, patches)
            timings[c].append(time.perf_counter() - start)

        after = score_all(deployed, banks, splits, existing)
        appended = score_all(deployed, banks, splits, held_out)

        regressed = [c for c in existing if not torch.equal(before[c], after[c])]
        differs = [c for c in held_out if not torch.equal(appended[c], scratch_scores[c])]
        aurocs = {c: round(image_auroc([t["label"] for t in splits[c]["test"]],
                                       appended[c].numpy()), 4) for c in held_out}
        per_seed.append({"seed": seed, "regressed": regressed,
                         "appended_differs_from_scratch": differs,
                         "appended_image_auroc": aurocs})
        print(f"seed {seed:<3} regression on existing: {regressed or 'none'}   "
              f"appended == from scratch: {'yes' if not differs else differs}   "
              f"AUROC {aurocs}")

    summary = {
        "question": "can a held-out product be appended without retraining, "
                    "matching from-scratch and leaving existing products untouched",
        "backbone": backbone,
        "shots": args.shots,
        "seeds": [m["seed"] for m in manifests],
        "existing": existing,
        "held_out": held_out,
        "held_out_rule": "the three alphabetically last categories, fixed in advance",
        "no_regression_all_seeds": all(not s["regressed"] for s in per_seed),
        "appended_equals_scratch_all_seeds": all(
            not s["appended_differs_from_scratch"] for s in per_seed),
        "append_seconds_end_to_end": {
            c: {"median": round(statistics.median(t), 3), "runs": [round(x, 3) for x in t]}
            for c, t in timings.items()
        },
        "append_bytes_per_class": args.shots * encoder.num_patches * encoder.embed_dim * 4,
        "mean_appended_image_auroc": {
            c: round(statistics.fmean(s["appended_image_auroc"][c] for s in per_seed), 4)
            for c in held_out
        },
        "per_seed": per_seed,
        "threads": args.threads,
    }
    for c, t in summary["append_seconds_end_to_end"].items():
        print(f"append {c:<12} {t['median']:.2f} s end to end (median over seeds)")

    path = repo_root() / "results" / "manifests" / args.out
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest -> {path}")
    ok = summary["no_regression_all_seeds"] and summary["appended_equals_scratch_all_seeds"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
