"""Demonstrate append-only incremental class addition.

    python scripts/run_incremental.py --shots 5

The claim: an MSME can start inspecting a new part by storing a handful of photos
of good ones. No retraining, no GPU, and -- critically -- no risk of disturbing
the parts already being inspected.

That last property is the one worth proving rather than asserting. Many
"incremental" methods re-fit a shared component (a normaliser, a codebook, a
classifier head) when a class arrives, which quietly shifts every existing
decision boundary. Here each class owns its bank, so this script adds all fifteen
categories one at a time and checks after every addition that the scores for all
previously-added categories are **bit-identical** to what they were.

It also records what each addition costs: time, memory, and prototypes stored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data.mvtec import complete_categories  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.eval.features import FeatureBank, bank_path  # noqa: E402
from src.eval.metrics import auroc  # noqa: E402
from src.model.memory import PrototypeMemory  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import dataset_dir, repo_root, results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="prototype.yaml")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    backbone = config["backbone"]["name"]
    root = dataset_dir(config["dataset"]["name"])
    splits_dir = repo_root() / "results" / "manifests" / "splits"

    categories = complete_categories(root)
    banks, manifests = {}, {}
    for category in categories:
        path = bank_path(results_root(), backbone, category)
        manifest = splits_dir / f"mvtec_ad_{category}_{args.shots}shot_seed{args.seed}.json"
        if path.exists() and manifest.exists():
            banks[category] = FeatureBank.load(path)
            manifests[category] = load_split(manifest)

    if len(banks) < 2:
        print("need at least two cached feature banks; run scripts/run_benchmark.py first",
              file=sys.stderr)
        return 1

    order = sorted(banks)
    print(f"backbone {backbone}   shots {args.shots}   seed {args.seed}")
    print(f"adding {len(order)} classes one at a time\n")

    memory = PrototypeMemory(
        dim=next(iter(banks.values())).dim,
        grid=next(iter(banks.values())).grid,
        top_k=config["model"]["scoring"]["top_k"],
    )

    baseline: dict[str, torch.Tensor] = {}
    history, drifted = [], []

    print(f"{'#':>2}  {'added':<12}{'add ms':>8}{'KiB':>8}{'total KiB':>11}"
          f"{'AUROC':>8}   existing classes unchanged")
    for index, category in enumerate(order, 1):
        manifest = manifests[category]
        support = banks[category].patches(manifest["support"])

        started = time.perf_counter()
        memory.add_class(category, support)
        add_ms = (time.perf_counter() - started) * 1000

        # score this new class, and re-score every class already in memory
        unchanged = True
        for known in order[:index]:
            m = manifests[known]
            scores = memory.score(banks[known].patches([t["image"] for t in m["test"]]), known)
            if known in baseline:
                if not torch.equal(baseline[known], scores.image):
                    unchanged = False
                    drifted.append({"after_adding": category, "changed": known})
            else:
                baseline[known] = scores.image

        labels = [t["label"] for t in manifest["test"]]
        class_auroc = auroc(baseline[category], labels)

        added_kib = (support.numel() * 4) / 1024
        history.append({
            "step": index,
            "added": category,
            "add_ms": round(add_ms, 3),
            "added_kib": round(added_kib, 1),
            "total_kib": round(memory.memory_bytes() / 1024, 1),
            "classes": len(memory),
            "prototypes": memory.total_prototypes(),
            "image_auroc": round(class_auroc, 4),
            "existing_unchanged": unchanged,
        })
        print(f"{index:>2}  {category:<12}{add_ms:>8.2f}{added_kib:>8.0f}"
              f"{memory.memory_bytes() / 1024:>11.0f}{class_auroc:>8.3f}   "
              f"{'yes' if unchanged else 'NO -- DRIFT'}")

    total_ms = sum(h["add_ms"] for h in history)
    print(f"\nclasses {len(memory)}   prototypes {memory.total_prototypes()}   "
          f"memory {memory.memory_bytes() / 1024 / 1024:.2f} MiB")
    print(f"total time to add all {len(order)} classes: {total_ms:.1f} ms")
    print(f"mean image AUROC across classes: "
          f"{sum(h['image_auroc'] for h in history) / len(history):.4f}")

    if drifted:
        print(f"\nFAILED: {len(drifted)} existing class scores changed", file=sys.stderr)
    else:
        print(f"\nappend-only holds: after every addition, all previously added "
              f"classes scored bit-identically")

    out = Path(args.out) if args.out else results_root() / "incremental" / f"{backbone}_{args.shots}shot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "backbone": backbone,
        "shots": args.shots,
        "seed": args.seed,
        "order": order,
        "history": history,
        "append_only_verified": not drifted,
        "drifted": drifted,
        "total_add_ms": round(total_ms, 2),
        "final_memory_bytes": memory.memory_bytes(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"written to {out}")
    return 1 if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main())
