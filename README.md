# fewshot-msme-inspect

Open-vocabulary few-shot defect detection that an MSME can actually run: a frozen
OpenCLIP backbone, an append-only prototype memory scored over multi-scale
windows, and optional fusion with typed text prompts -- benchmarked for accuracy
**and** CPU cost on a laptop. Nothing is trained.

## Results at a glance

10 good photos per product, mean over 3 seeds, image / pixel AUROC:

| | MVTec AD | VisA | NEU-DET (image only) | CPU ms/image, burst |
|---|---|---|---|---|
| ViT-B/32, memory | 0.904 / 0.940 | 0.779 / 0.935 | 0.893 | 37 (ONNX INT8: 33) |
| ViT-B/16, memory | 0.904 / 0.936 | 0.831 / 0.951 | 0.887 | 88 (ONNX INT8: 72) |
| ViT-B/16 + windows | 0.921 / 0.939 | 0.851 / 0.949 | **0.935** | ~90 (est.) |
| ViT-B/16 + windows + text | **0.925** / 0.939 | **0.855** / 0.949 | 0.881 | ~90 (est.) |
| ViT-L/14, memory | **0.933** / 0.945 | 0.843 / **0.964** | 0.897 | 397 |

Text fusion helps on products with a meaningful "normal" (MVTec, VisA) and hurts
on NEU-DET, where the normal class is itself a defect type. A laptop throttles to
5-6x the burst latency under continuous use.

Adding a product is bit-identical to building it in from scratch and leaves all
other products' scores unchanged. Details, statistics and caveats:
[docs/week2_results.md](docs/week2_results.md). Paper draft: [paper/](paper/).

## What this repo is

A software-only research track (8 weeks, 2 members, no dedicated hardware). The
headline claims we are building evidence for:

* a **CPU-only compute-efficiency benchmark** — accuracy x latency x RAM x FLOPs
  across ViT-B/16, ViT-B/32 and ViT-L/14, in FP32 and INT8;
* **append-only incremental class addition** — a new defect class is added by
  appending prototypes, not by retraining;
* a **zero-cost feasibility analysis** for a small manufacturer.

Datasets: MVTec AD, VisA, NEU-DET (1/5/10-shot seeded splits, 3 seeds), MPDD optional.
Baselines: PatchCore / PaDiM via anomalib, plus CLIP zero-shot prompting.

## Layout

```
configs/        one YAML per experiment style; base.yaml holds the shared defaults
src/data/       MVTec loader, seeded few-shot split generator
src/model/      encoder.py   frozen OpenCLIP, pooled + patch features
                memory.py    append-only prototype memory
                windows.py   multi-scale window scoring, text fusion
                adapter.py   Tip-Adapter cache head (measured, rejected)
                export.py    ONNX export, INT8
src/eval/       features.py  feature bank cache
                pipeline.py  end-to-end split evaluation
                metrics.py   image/pixel AUROC, F1-max
                profiling.py latency, RAM, FLOPs, params
src/baselines/  reference baselines (anomalib, CLIP zero-shot)
src/utils/      config loading, path resolution, seeding, architecture freeze check
src/demo/       inspector behind the Streamlit demo
app/            demo.py -- streamlit run app/demo.py
scripts/        runnable entry points, one per stage of the pipeline
notebooks/      colab_reproduce.ipynb (zero setup), kaggle_setup.ipynb (GPU bursts)
tests/          the append-only, seeding and freeze guarantees are asserted here
paper/          main.tex + refs.bib, draft
docs/           validation logs and notes
results/        run artifacts (gitignored) except results/manifests/ (tracked)
data/           datasets live here by default and are never committed
```

## The pipeline

Each stage writes what the next one reads, so they can be run independently and
re-run cheaply.

```bash
python scripts/download_mvtec_category.py --all   # 5 GB, verifies against published counts
python scripts/generate_splits.py --seeds 42 1 2  # seeded manifests; already committed
python scripts/run_benchmark.py                   # prototype memory, 3 seeds
python scripts/run_benchmark.py --config proposed.yaml   # + windows + text
python scripts/run_profiling.py --threads 6       # idle machine; 14 threads is unstable
python scripts/run_incremental.py                 # append-only demonstration
python scripts/make_figures.py                    # frontier figure + manifest
```

Or regenerate any table in the results doc with one command:
`python scripts/reproduce.py --list`.

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

## Demo

```bash
streamlit run app/demo.py
```

Add a product from a few photos of good parts (or load any MVTec AD / VisA
category from the committed split when `FEWSHOT_DATA_ROOT` is set), then inspect
parts: anomaly score, heatmap, and pass/fail against a threshold suggested from
the good photos themselves. Add a second product and re-inspect a part of the
first -- the score is bit-identical. Runs on CPU; the logic lives in
`src/demo/inspector.py` and uses exactly the benchmarked encoder and memory.

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
