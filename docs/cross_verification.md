# Cross-verification, code review and demo rehearsal

Blueprint §19 asks that every headline number be recomputed by the member who did
not produce it. This file records what has been checked, how, and what is still
open. **Status: partly done.** The MVTec and VisA checks below are a same-machine
clean-room rerun, not the independent check the Blueprint asks for; the NEU-DET
numbers have not been checked by anyone but their author.

## 1. Headline numbers, as committed

Image AUROC, prototype memory, mean over 3 seeds (42, 1, 2), FP32.

| dataset | backbone | 1-shot | 5-shot | 10-shot | source |
|---|---|---|---|---|---|
| MVTec AD | ViT-B/16 | 0.856 | 0.888 | 0.904 | week2_results §1 |
| MVTec AD | ViT-B/32 | 0.837 | 0.884 | 0.904 | week2_results §2 |
| VisA | ViT-B/16 | 0.773 | 0.811 | 0.831 | week2_results §6 |
| VisA | ViT-B/32 | 0.681 | 0.761 | 0.779 | week2_results §6 |
| NEU-DET | ViT-B/16 | 0.810 | 0.854 | 0.870 | Member B, unverified |
| NEU-DET | ViT-B/32 | 0.818 | 0.867 | 0.893 | Member B, unverified |

**Correction to the first version of this file (2026-10-02).** It listed MVTec
B/32 as 0.846 / 0.886 / 0.904 and VisA B/32 as 0.681 / 0.748 / 0.765. Neither was a
B/32 FP32 result: the MVTec row is the mean of B/16 and B/32 (what
`summarise_results.py` printed before it learned to filter by dataset and
backbone), and the VisA 5- and 10-shot values are the INT8 run. Use
`--dataset` and `--backbone` with `summarise_results.py`, or
`compare_paired.py`, so runs are never pooled.

## 2. Clean-room rerun (Member A's machine)

Every feature was re-extracted from the raw images into an empty results folder
(no cached feature banks), then the benchmark was re-run and compared row by row
with the committed CSVs. This catches stale or corrupted caches and confirms the
pipeline reproduces from data; it does not catch a mistake shared by the code, so
it is not a substitute for an independent rerun.

| run | rows | rows matching | max abs diff, image AUROC | max abs diff, pixel AUROC | means 1 / 5 / 10 |
|---|---|---|---|---|---|
| MVTec AD, ViT-B/16 | 135 | 135 | 0.00019 (one row) | 7e-7 | 0.856 / 0.888 / 0.904 |
| VisA, ViT-B/32 | 108 | 108 | 0.0001 | 7e-7 | 0.681 / 0.761 / 0.779 |

Commands (Windows, Intel i7-13700H, torch 2.8.0+cpu):

```bash
export FEWSHOT_DATA_ROOT=<datasets> FEWSHOT_RESULTS_ROOT=<empty folder>
python scripts/run_benchmark.py --threads 12
python scripts/run_benchmark.py --dataset visa --backbone ViT-B-32-quickgelu --threads 12
```

The single 0.0002 difference is float noise from re-extraction (fp16 storage,
thread scheduling); it does not move any reported mean.

## 3. Still open

* **Independent rerun by Member B** of `python scripts/reproduce.py mvtec backbones
  incremental` on her own machine, compared against week2_results §1, 2, 5 and 6.
* **NEU-DET verification by Member A.** Needs the same copy of the data Member B
  used (Kaggle `kaustubhdikshit/neu-surface-defect-database`). NEU-DET also still
  needs the L/14 and proposed-config runs, paired comparisons and a write-up in
  week2_results.

### NEU-DET class breakdown (Member B's run, unverified)

| class | ViT-B/16 (1 / 5 / 10-shot) | ViT-B/32 (1 / 5 / 10-shot) |
|---|---|---|
| crazing | 0.991 / 0.993 / 0.994 | 0.876 / 0.917 / 0.931 |
| inclusion | 0.658 / 0.780 / 0.831 | 0.871 / 0.901 / 0.946 |
| patches | 0.704 / 0.779 / 0.828 | 0.803 / 0.874 / 0.901 |
| pitted_surface | 0.878 / 0.921 / 0.917 | 0.709 / 0.834 / 0.864 |
| rolled-in_scale | 0.974 / 0.979 / 0.982 | 0.938 / 0.992 / 0.994 |
| scratches | 0.724 / 0.757 / 0.768 | 0.709 / 0.683 / 0.723 |
| **mean** | **0.810 / 0.854 / 0.870** | **0.818 / 0.867 / 0.893** |

## 4. Incremental class addition (both machines)

| | Member A | Member B |
|---|---|---|
| classes, prototypes, memory | 15, 14,700, 43.07 MiB | 15, 14,700, 43.07 MiB |
| mean image AUROC (5-shot, B/16) | 0.8863 | 0.8863 |
| append-only check | passed, bit-identical | passed, bit-identical |
| total append time | 27.8 ms | 21.7 ms |

Accuracy and memory agree exactly; time is machine-dependent, so the docs now
quote 1.4-1.9 ms per class.

## 5. Code review of the data-side changes since Aug 24

Reviewed: `src/data/splits.py`, `scripts/generate_splits.py`,
`scripts/run_benchmark.py`, `scripts/check_splits.py`,
`scripts/build_neu_det_oneclass.py`, `src/eval/pipeline.py`.

* **Fixed, `scripts/check_splits.py`.** It read manifests from
  `$FEWSHOT_RESULTS_ROOT/manifests/splits`, but committed manifests live in the
  repo. With the documented setup (`FEWSHOT_RESULTS_ROOT` pointing outside the
  repo) it found no manifests and checked nothing. Now reads from the repo; a
  test sets `FEWSHOT_RESULTS_ROOT` elsewhere and fails on the old behaviour.
* **Checked, NEU-DET path handling.** Under the one-class layout the same image
  (e.g. `inclusion_250.jpg`) appears in several categories' test sets. Labels come
  from the parent folder and `split_for()` strips paths at the first
  `/<category>/`, which is always the category directory itself, so labels and
  paths resolve correctly. Feature banks are per category, so no cross-category
  cache collision. No masks: pixel metrics correctly print `n/a`.
* **Checked, `CATEGORY_SUBDIR`.** `neu_det` added identically in both files.
* **Fixed, `requirements.txt`.** A second unpinned `kaggle` line had been added;
  removed (the first remains).
* **Note, `results/manifests/incremental_5shot.json`** was overwritten by a rerun
  on Member B's machine. Accuracy is unchanged; only the timing differs (section 4).

## 6. Demo rehearsal

`streamlit run app/demo.py`, driven headless with Streamlit's `AppTest`, twice,
with ViT-B/32 (the default):

| step | rehearsal 1 | rehearsal 2 |
|---|---|---|
| add bottle, 5 photos + "a broken or cracked rim" | added | added |
| inspect broken_large/000.png | DEFECTIVE, 0.490 vs 0.350 | DEFECTIVE, 0.490 vs 0.350 |
| inspect good/000.png | OK, 0.323 vs 0.350 | OK, 0.323 vs 0.350 |
| add screw live + "a damaged thread" | added | added |
| re-inspect broken_large/000.png | "Bit-identical" | "Bit-identical" |

**One weakness found.** Adding metal_nut from exactly 3 photos (+ "a bent or
scratched nut"): defective parts scored higher on average (0.388 vs 0.373 over 5
of each), but the suggested threshold, the worst leave-one-out score among the
3 good photos (0.421), flagged none of the 5 defects. With 3 photos the
suggestion is too cautious. For the viva, add the live product from 5 photos, or
lower the threshold slider by hand and explain why. The rule was not changed:
tuning it on these test images would be fitting to the test set.
