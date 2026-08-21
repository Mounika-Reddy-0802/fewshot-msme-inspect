# Week 1 — environment validation with PatchCore

Not a contribution. The point is to prove the environment is wired up correctly
before we start producing numbers of our own: if a well-known baseline does not
reproduce here, nothing we measure later can be trusted.

**Result: it reproduces.** PatchCore on MVTec AD `bottle` lands inside the
published range on both image- and pixel-level AUROC.

## What was run

Category `bottle` only. The full MVTec AD archive is ~5 GB of fifteen
categories, and week 1 does not need it — dataset-wide downloads and the
seeded split generators are Member B's task.

```bash
# roots are environment variables, never hardcoded paths
export FEWSHOT_DATA_ROOT="C:/Users/HP/datasets"

# 1. fetch one category (209 train / 83 test / 63 masks)
python scripts/download_mvtec_category.py --category bottle

# 2. run the baseline, config-driven from configs/baseline.yaml
python scripts/run_patchcore_validation.py --category bottle
```

Hyperparameters come from [`configs/baseline.yaml`](../configs/baseline.yaml)
rather than the command line, so this run is versioned:

```yaml
model:
  method: patchcore
  patchcore:
    backbone: wide_resnet50_2
    layers: [layer2, layer3]
    coreset_sampling_ratio: 0.1
    num_neighbors: 9
experiment:
  seed: 42
```

## Output

```
+-------------------------------------------------------+
|        Test metric        |       DataLoader 0        |
|---------------------------+---------------------------|
|        image_AUROC        |            1.0            |
|       image_F1Score       |    0.9919999837875366     |
|        pixel_AUROC        |     0.985623300075531     |
|       pixel_F1Score       |    0.7269942164421082     |
+-------------------------------------------------------+

=== patchcore validation ===
  image_AUROC                  1.0
  image_F1Score                0.992
  pixel_AUROC                  0.9856
  pixel_F1Score                0.727
  fit 1330.0s, test 103.9s
  published range (anomalib benchmark / patchcore paper):
    image_auroc    0.995-1.000   got 1.0      in range
    pixel_auroc    0.975-0.990   got 0.9856   in range
```

Machine-readable copy: `results/validation/patchcore_bottle.json` (gitignored,
regenerate with the command above).

## Against the published numbers

| metric | this run | published | verdict |
|---|---|---|---|
| image AUROC | **1.0000** | 1.000 (PatchCore paper, `bottle`) | in range |
| pixel AUROC | **0.9856** | ~0.984 (anomalib benchmark, `bottle`) | in range |
| image F1-max | 0.9920 | — | reported for our own harness comparison |
| pixel F1-max | 0.7270 | — | reported for our own harness comparison |

`bottle` is one of the easier categories, so image AUROC saturating at 1.000 is
expected and is a weak signal on its own. Pixel AUROC at 0.9856 is the more
informative check — it depends on the feature extractor, the coreset sampler and
the anomaly-map upsampling all being correct, and it matches.

## Environment

| | |
|---|---|
| CPU | Intel Core i7-13700H, 14 cores / 20 threads, 16 GB RAM |
| OS | Windows 11 |
| Python | 3.13.5 |
| torch | 2.8.0+cpu |
| anomalib | 2.6.0 |
| lightning | 2.6.5 |
| device | CPU only — no GPU used anywhere in this run |

Wall-clock on CPU: **22 min fit** (dominated by coreset selection, 21,400
indices from 214,016 patches) and **104 s test** over 83 images. Worth recording
now: this is the kind of cost our own method has to beat on MSME hardware.

## Deviations worth knowing

Three things differ from a textbook setup. None affect the metrics, but they
should be visible rather than buried:

1. **Dataset source.** The official archive is a single 5 GB tarball of all
   fifteen categories, which would have broken the "one category only" limit for
   week 1. `bottle` was pulled from the `foersben/mvtec-ad` mirror, which keeps
   the original folder layout. Verified faithful: 209 train / 83 test / 63 masks,
   matching the official counts for this category exactly.
2. **torch install.** The CPU wheel index (`download.pytorch.org`) stalled at
   ~9 kB/s on this connection. The venv was created with
   `--system-site-packages` so it reuses an already-present `torch 2.8.0+cpu` —
   the exact version pinned in `requirements.txt`, so the environment is the one
   we intended. On a normal connection the documented
   `pip install torch==2.8.0 --index-url .../cpu` path is the one to use.
3. **Offline run.** `HF_HUB_OFFLINE=1` was set for the run itself, after caching
   the `timm/wide_resnet50_2.racm_in1k` weights, so a flaky link could not stall
   a 22-minute job partway through.

## Reproducing

```bash
export FEWSHOT_DATA_ROOT=/path/to/datasets
python scripts/download_mvtec_category.py --category bottle
python scripts/run_patchcore_validation.py --category bottle
```

The script prints the comparison itself and exits non-zero only on failure; the
published ranges it checks against live in `PUBLISHED` at the top of
[`scripts/run_patchcore_validation.py`](../scripts/run_patchcore_validation.py).
