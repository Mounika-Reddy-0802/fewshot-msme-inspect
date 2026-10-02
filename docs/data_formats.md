# Shared data formats

These are the interfaces between the data side (Member B) and the modelling side
(Member A). Every script reads and writes exactly these formats, so changing one
breaks work on both sides. Per [GIT_RULES.md](../GIT_RULES.md): agree first, then
one member commits the change.

The earlier version of this document described per-(category, shots, seed)
manifest files, `make_splits.py` and `load_split()`. That design was never built;
what follows is what runs.

## Datasets on disk

Datasets live under `$FEWSHOT_DATA_ROOT` (default `./data`), never in git. Each
dataset keeps MVTec's folder shape per category:

```
<data_root>/<dataset>/<subdir>/<category>/train/good/*
<data_root>/<dataset>/<subdir>/<category>/test/good/*
<data_root>/<dataset>/<subdir>/<category>/test/<defect_type>/*
<data_root>/<dataset>/<subdir>/<category>/ground_truth/<defect_type>/*   (if masks exist)
```

`<subdir>` comes from `CATEGORY_SUBDIR`, defined identically in
`scripts/generate_splits.py` and `src/data/splits.py` (they must stay in sync):

| dataset | subdir | how it gets this shape | masks |
|---|---|---|---|
| `mvtec_ad` | (none) | `scripts/download_mvtec_full.py` | `<stem>_mask.png` |
| `visa` | `visa_pytorch` | `scripts/download_visa.py` | `<stem>.png` |
| `neu_det` | (none) | `scripts/reorganize_neu_det.py`, then `scripts/build_neu_det_oneclass.py` | none |

**The label comes from the folder.** A test image whose parent folder is `good`
is normal (label 0); anything else is anomalous (label 1), and the folder name is
its defect type.

**NEU-DET has no defect-free images**, so `build_neu_det_oneclass.py` builds a
one-class-as-normal layout: for each of the six classes c, `c/train/good` holds
c's train images, `c/test/good` c's 60 validation images, and `c/test/<other>`
12 validation images from each other class (60 in all, drawn with a fixed seed
independent of the split seeds). So each category's test set is 50/50 and the
question is "given photos of one normal steel surface, is this surface
different?" Image AUROC only; there are no masks.

## Split manifests

One JSON file **per (dataset, seed)**, committed under
`results/manifests/splits/` (in the repo, not under `$FEWSHOT_RESULTS_ROOT`):

```
results/manifests/splits/mvtec_ad.json          seed 42 (the default seed keeps the plain name)
results/manifests/splits/mvtec_ad_seed1.json    seed 1
results/manifests/splits/mvtec_ad_seed2.json    seed 2
```

Same pattern for `visa` and `neu_det`. Written by:

```bash
python scripts/generate_splits.py --dataset <mvtec_ad|visa|neu_det> --seeds 42 1 2
```

Schema:

```json
{
  "dataset": "visa",
  "seed": 42,
  "shot_counts": [0, 1, 5, 10],
  "categories": {
    "candle": {
      "shots": {"0": [], "1": ["visa/visa_pytorch/candle/train/good/0123.JPG"], "5": [...], "10": [...]},
      "test": ["visa/visa_pytorch/candle/test/bad/000.JPG", "..."],
      "num_train_good_available": 900
    }
  }
}
```

* Paths are relative to `$FEWSHOT_DATA_ROOT`, include the dataset directory and
  its subdir, and always use forward slashes, so a manifest generated on Windows
  is byte-identical to one generated on Linux.
* `load_manifest()` rejects a file missing any of `dataset`, `seed`,
  `shot_counts`, `categories`.

### Guarantees, each enforced by a test or `scripts/check_splits.py`

* **Nested shots.** 1-shot is a subset of 5-shot is a subset of 10-shot, so a gain
  from more shots is the added images, not a luckier draw.
* **Fixed test set.** The test list does not depend on shot count, so 1-shot and
  10-shot are scored on identical queries.
* **No leakage.** No support image appears in its category's test set.
* **Seeds differ.** Two seeds never draw the same support set for a category.

`python scripts/check_splits.py` checks all of these across every committed
manifest and exits non-zero on any violation.

## Reading a split

```python
from src.data.splits import load_manifest, manifest_paths, split_for, resolve_split

paths = manifest_paths("results/manifests/splits", "mvtec_ad")   # default seed first
manifest = load_manifest(paths[0])
split = split_for(manifest, "bottle", 5, data_root)               # one (category, shots)
support, test = resolve_split(split, data_root)                   # absolute paths
```

`split_for()` returns paths relative to the category directory, a 0/1 label and
defect type per test image, and derives each anomalous image's mask path by
trying both naming conventions above. If no mask is found the mask is `None` and
pixel metrics for that image print as `n/a` rather than silently becoming zero.
`test` from `resolve_split()` is a list of `Sample(image, label, defect_type, mask)`.

## Feature banks

The backbone is frozen, so an image's features never change: they are extracted
once and cached.

```
$FEWSHOT_RESULTS_ROOT/feature_banks/<backbone>-d<dim>/<category>.pt
$FEWSHOT_RESULTS_ROOT/feature_banks/<backbone>-int8-d<dim>/<category>.pt   (ONNX INT8 encoder)
```

`<dim>` is the patch-feature width: 768 for ViT-B/32 and B/16, 1024 for L/14
(unprojected, the project convention), or 512 if `project_patches` is on. It is
part of the path because features of different widths live in different spaces
and must never be mixed. Path from `src.eval.features.bank_path()`.

Each file is a `torch.save` dict:

| key | content |
|---|---|
| `backbone`, `category` | identity |
| `grid` | patch grid, e.g. `(14, 14)` |
| `dim` | patch-feature width |
| `index` | `{relative image path: row}` |
| `images` | `(N, 512)` pooled image embeddings, float16, l2-normalised |
| `patches` | `(N, H*W, dim)` patch tokens, float16, l2-normalised |

Read with `FeatureBank.load(path)`, then `.patches(paths)` / `.images(paths)`.
float16 halves the size; similarities live in [-1, 1], where it has ample
precision.

## Result records

Everything under `$FEWSHOT_RESULTS_ROOT` (benchmark CSVs, figures, ONNX graphs) is
regenerable and gitignored. What is committed is `results/manifests/*.json`: one
record per experiment, naming exactly the runs behind each number in
`docs/week2_results.md`. `python scripts/reproduce.py --list` maps each section to
the commands that regenerate it.
