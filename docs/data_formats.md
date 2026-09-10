# Shared data formats

Member B owns dataset loading, splits and the evaluation harness. The code in
`src/data/` is a **stand-in** written so the modelling and profiling work was not
blocked — but the *formats* below are the shared interface, and changing them
breaks work on both sides.

Per the working agreement in [GIT_RULES.md](../GIT_RULES.md): agree first, then one
member commits the change. A real loader from Member B can replace `src/data/`
entirely as long as it reads and writes these formats.

## Split manifests

Schema version **1**. One JSON file per (dataset, category, shots, seed):

```
results/manifests/splits/mvtec_ad_bottle_5shot_seed0.json
```

These live inside the repo and **are committed**, unlike everything else under
`results/`. They are the reproducibility record — without them the few-shot
numbers cannot be checked by anyone else.

```json
{
  "schema_version": 1,
  "dataset": "mvtec_ad",
  "category": "bottle",
  "shots": 5,
  "seed": 0,
  "support": ["train/good/001.png", "train/good/012.png", "..."],
  "test": [
    {"image": "test/good/000.png", "label": 0, "defect_type": "good", "mask": null},
    {"image": "test/broken_large/000.png", "label": 1, "defect_type": "broken_large",
     "mask": "ground_truth/broken_large/000_mask.png"}
  ],
  "counts": {
    "support": 5, "train_normals_available": 209, "test_total": 83,
    "test_normal": 20, "test_anomalous": 63, "test_with_mask": 63
  }
}
```

`label` is 0 for normal and 1 for anomalous. `mask` is null for normal images —
only defective ones have ground truth, which is what makes pixel-level metrics
possible.

### Four guarantees, and why each one matters

**Paths are relative to the category directory.** A manifest generated on Windows
resolves unchanged on Kaggle. Nothing records `C:\Users\...`.

**Shots are nested: 1-shot ⊂ 5-shot ⊂ 10-shot.** Support images are drawn by
shuffling the training normals once per (category, seed) and taking the first k.
So going from 1 to 5 shots differs by exactly four added images — a gain is
attributable to the extra data, not to having drawn a luckier set. Verified on
real data: `bottle` seed 0 picks `train/good/012.png` at k=1, and that same file
is in the k=5 support.

**The test split never changes with k.** Accuracy at 1 shot and at 10 shots is
measured on identical queries, so the numbers are directly comparable.

**Seeds derive through sha256, not `hash()`.** Python's `hash()` is salted per
process, so a generator built on it silently produces different splits between
runs. `_derive_seed` in `src/data/splits.py` uses sha256 of
`(seed, dataset, category)`.

### Regenerating

```bash
export FEWSHOT_DATA_ROOT=/path/to/datasets
python scripts/make_splits.py                          # every downloaded category
python scripts/make_splits.py --categories bottle cable
python scripts/make_splits.py --shots 0 1 5 10 --seeds 0 1 2
```

Defaults: shots `0 1 5 10`, seeds `0 1 2` (three seeds so few-shot results can
carry error bars rather than a single lucky draw). Output goes to
`results/manifests/splits/` in the repo, deliberately not to
`FEWSHOT_RESULTS_ROOT`, so the manifests end up committed.

Regeneration is deterministic and idempotent — rerunning with the same seeds
rewrites byte-identical files, so a noisy diff means something actually changed.

## Reading a split

```python
from src.data.splits import load_split, resolve_split

manifest = load_split("results/manifests/splits/mvtec_ad_bottle_5shot_seed0.json")
support, test = resolve_split(manifest, root)   # absolute paths for this machine
```

`support` is a list of `Path`; `test` is a list of `Sample`
(`image`, `label`, `defect_type`, `mask`).

## Feature banks

Not built yet — this section gets filled in when the bank builder lands, and the
layout needs agreeing before either side depends on it.
