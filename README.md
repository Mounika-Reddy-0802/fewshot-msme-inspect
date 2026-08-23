# fewshot-msme-inspect

Open-vocabulary few-shot defect detection that an MSME can actually run: a frozen
OpenCLIP backbone, an appendable prototype memory and a lightweight adapter,
benchmarked for accuracy **and** CPU cost on free-tier hardware.

## What this repo is

A software-only research track (8 weeks, 2 members, no dedicated hardware). The
headline claims we are building evidence for:

* a **CPU-only compute-efficiency benchmark** — accuracy x latency x RAM x FLOPs
  across ViT-B/16, ViT-B/32 and ViT-L/14, in FP32 and INT8;
* **append-only incremental class addition** — a new defect class is added by
  appending prototypes, not by retraining;
* a **zero-cost feasibility analysis** for a small manufacturer.

Datasets: MVTec AD, VisA, NEU-DET (0/1/5/10-shot seeded splits), MPDD optional.
Baselines: PatchCore / PaDiM via anomalib, plus CLIP zero-shot prompting.

## Layout

```
configs/        one YAML per experiment style; base.yaml holds the shared defaults
src/data/       MVTec loader, seeded few-shot split generator
src/model/      encoder.py   frozen OpenCLIP, pooled + patch features
                memory.py    append-only prototype memory
                adapter.py   Tip-Adapter cache head
src/eval/       features.py  feature bank cache
                pipeline.py  end-to-end split evaluation
                metrics.py   image/pixel AUROC, F1-max
                profiling.py latency, RAM, FLOPs, params
src/baselines/  reference baselines (anomalib, CLIP zero-shot)
src/utils/      config loading, path resolution, seeding
scripts/        runnable entry points, one per stage of the pipeline
notebooks/      kaggle_setup.ipynb mirrors this environment for free GPU bursts
tests/          145 tests; the append-only and seeding guarantees are asserted here
docs/           validation logs and notes
results/        run artifacts (gitignored) except results/manifests/ (tracked)
data/           datasets live here by default and are never committed
```

## The pipeline

Each stage writes what the next one reads, so they can be run independently and
re-run cheaply.

```bash
python scripts/download_mvtec_category.py --all   # 5 GB, verifies against published counts
python scripts/make_splits.py                     # 180 seeded manifests -> committed
python scripts/run_benchmark.py --methods memory adapter
python scripts/run_profiling.py --threads 14      # on an idle machine
python scripts/run_incremental.py                 # append-only demonstration
python scripts/make_figures.py                    # frontier figure + manifest
```

Two design decisions worth knowing before reading the code:

**Features are extracted once.** The backbone is frozen, so an image's features
never change. `run_benchmark.py` caches them per (backbone, category); every
shot count, seed and scoring head then reuses one pass over the images. A warm
re-run of the whole 180-cell grid takes seconds.

**Shot counts are nested.** The 1-shot support set is a subset of the 5-shot set,
which is a subset of the 10-shot set. Going from 1 to 5 shots therefore measures
the four added images, not a different draw. Verified across all 45
(category, seed) pairs.

## Setup

Python 3.10+ and about 6 GB of disk for the environment.

```bash
python -m venv .venv                 # or somewhere outside a synced folder
source .venv/Scripts/activate        # Windows bash; Linux/macOS: .venv/bin/activate
pip install -r requirements.txt
```

The CPU build of torch is what the benchmark measures, so install it from the
CPU index:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
```

Datasets and results are large, so their roots come from environment variables
rather than hardcoded paths. Both default to a folder inside the repo:

```bash
export FEWSHOT_DATA_ROOT=/path/to/datasets      # default: ./data
export FEWSHOT_RESULTS_ROOT=/path/to/results    # default: ./results
```

Check the install:

```bash
python scripts/validate_env.py
pytest -q
```

## GPU bursts

Everything reported in the paper is measured locally on CPU. When something is
just slow rather than interesting -- building feature banks, adapter sweeps --
run it on a free Kaggle GPU session with
[`notebooks/kaggle_setup.ipynb`](notebooks/kaggle_setup.ipynb), which reproduces
this environment there.

Kaggle needs three things switched on in the notebook's right-hand panel:
Accelerator set to GPU, Internet set to On, and a `GITHUB_TOKEN` secret (the repo
is private). Attaching an MVTec dataset with **Add Data** mounts it read-only
under `/kaggle/input`, so no download is needed -- the notebook finds it by its
category folders rather than by a hardcoded dataset name.

## Running an experiment

Everything is config driven and seeded. Pick a config, override on the command
line if needed:

```bash
python -c "from src.utils.config import load_config; print(load_config('prototype.yaml', ['dataset.shots=10']))"
```

`base.yaml` holds the defaults; `baseline.yaml`, `prototype.yaml` and
`adapter.yaml` inherit from it with `extends:` and state only what they change.

## Who does what

* **Member A** — modelling and efficiency: backbone, prototype memory, adapter,
  ONNX/INT8 export, CPU profiling harness, frontier figures.
* **Member B** — data and evaluation: dataset loaders, seeded 0/1/5/10-shot split
  generator, feature banks, evaluation harness (image/pixel AUROC, F1-max),
  dissemination.

Working agreements, commit rules and freeze dates live in [GIT_RULES.md](GIT_RULES.md).

## License

MIT — see [LICENSE](LICENSE).
