# Open-Vocabulary Few-Shot Defect Detection for MSMEs

Presentation reference. Every number here is measured in this repository and
traceable to a committed manifest; anything not yet done is marked as such rather
than estimated.

---

# Part 1 — Background and Motivation

## 1.1 The main idea

Let a small manufacturer start automated visual inspection of a new part by
**photographing a handful of good examples** — no defect photos, no labelling, no
model training, and no GPU.

A frozen CLIP image encoder turns each photo into per-patch feature vectors.
Those vectors are stored in an append-only memory. A new part is inspected by
asking, for each of its patches, *how similar is this to anything I have seen on
a good part?* Patches that match nothing are flagged as defects.

Nothing is trained. Adding a product class means appending vectors to a list.

## 1.2 What exists today

| approach | representative | what it needs | limitation for an MSME |
|---|---|---|---|
| Reconstruction | autoencoders, GANs | hundreds of normal images, training | needs a GPU and ML staff |
| Embedding + memory | **PatchCore**, PaDiM | ~200+ normal images per class, ImageNet CNN | retrain/refit per product; no language interface |
| Synthetic defects | CutPaste, DRAEM | training on simulated flaws | simulated defects rarely match real ones |
| Zero-shot vision-language | **WinCLIP**, CLIP prompting | text prompts only | accuracy too low to deploy alone |
| Few-shot vision-language | WinCLIP+, APRIL-GAN, AnomalyCLIP | 1–10 images, some train a projection head | evaluated on GPUs; cost never reported |

PatchCore is the strongest baseline and we reproduced it as a sanity check:
**image AUROC 1.000, pixel AUROC 0.9856** on MVTec `bottle`, inside the published
range. It needed all 209 normal training images and 22 minutes of CPU.

## 1.3 Gaps and drawbacks

1. **Cost is never reported.** Papers report AUROC, almost never latency, RAM or
   FLOPs, and essentially never on CPU. An MSME cannot tell whether a method runs
   on the laptop it already owns.
2. **Adding a product class is disruptive.** Methods that re-fit a shared
   component — a normaliser, a coreset, a classifier head — silently shift the
   decision boundaries of classes already in production.
3. **Few-shot claims are fragile.** With one random draw of support images,
   results swing widely. We measured a **±0.102 spread** on `capsule` at 1 shot
   across three seeds; many papers report a single run.
4. **Defect photos do not exist.** A new production line has good parts and no
   catalogue of failures, so any method needing defect examples is unusable at
   the moment it is most needed.

## 1.4 USP — what is different here

1. **A CPU-only cost benchmark alongside accuracy.** Latency, peak RAM, FLOPs and
   parameters, measured with pinned threads, discarded warm-up and recorded
   machine load.
2. **Append-only class addition, proven rather than asserted.** Adding all 15
   MVTec classes one at a time leaves every previously added class scoring
   **bit-identically** (`torch.equal`, not "close enough"). Cost: **~1.4-1.9 ms and
   3.0 MB per class.**
3. **Zero training anywhere in the pipeline.** No epochs, no optimiser, no
   learning rate. Deployment is a file copy.
4. **Honest negative results.** The Tip-Adapter head we built adds nothing, and
   we show mathematically why. Reporting that is more useful than hiding it.
5. **Free-tier reproducibility.** Committed split manifests, three seeds, and a
   Kaggle notebook mirroring the environment. Total spend: ₹0.

## 1.5 Objectives

1. Build a frozen-backbone, append-only few-shot defect detector needing only
   normal images.
2. Benchmark accuracy **and** CPU cost across backbones (ViT-B/32, ViT-B/16,
   ViT-L/14) and precisions (FP32, INT8).
3. Demonstrate and measure append-only incremental class addition.
4. Evaluate on MVTec AD, VisA and NEU-DET with seeded 0/1/5/10-shot splits.
5. Produce a ₹0 feasibility analysis for a small manufacturer.

## 1.6 Problem statement

> Given only a handful of photographs of non-defective examples of a product, and
> **only a CPU**, detect and localise manufacturing defects in unseen images of
> that product — such that adding a new product class requires no retraining and
> provably does not alter inspection of classes already in production — and report
> the accuracy **and** the compute cost of doing so.

---

# Part 2 — Technical

## 2.1 System architecture

```
                 ┌──────────────────────────────────────────┐
   good photos   │  PREPROCESS                              │
   (1/5/10)  ──► │  resize 224 bicubic → centre-crop 224    │
                 │  → CLIP normalise                        │
                 └───────────────────┬──────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │  FROZEN ENCODER   src/model/encoder.py   │
                 │  OpenCLIP ViT-B/16-quickgelu (150 M)     │
                 │  requires_grad = False, eval mode        │
                 │  out: 14×14 = 196 patches × 768 dims     │
                 └───────────────────┬──────────────────────┘
                                     ▼
                 ┌──────────────────────────────────────────┐
                 │  FEATURE BANK     src/eval/features.py   │
                 │  float16 on disk, keyed by path + dim    │
                 │  extracted once, reused by every run     │
                 └───────────────────┬──────────────────────┘
                                     ▼
    ┌────────────────────────────────┴─────────────────────────────┐
    ▼                                                              ▼
┌─────────────────────────────┐              ┌──────────────────────────────┐
│ PROTOTYPE MEMORY            │              │ QUERY IMAGE                  │
│ src/model/memory.py         │              │ same encoder → 196×768       │
│ one bank per class,         │              └──────────────┬───────────────┘
│ l2-normalised, append-only  │                             │
└──────────────┬──────────────┘                             │
               └───────────────► cosine similarity ◄────────┘
                                        │
                                        ▼
                        anomaly(patch) = 1 − max cosine sim
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        image score = max over patches         14×14 map → bilinear ↑224
                                               → Gaussian blur σ=4
                    │                                       │
                    ▼                                       ▼
             image AUROC, F1-max                  pixel AUROC, F1-max
                                src/eval/metrics.py (Member B)
```

Two side branches exist and are reported honestly:

* **Tip-Adapter head** (`src/model/adapter.py`) — a kernel over the whole cache
  instead of a hard max, optionally blended with text prompts. **Measured to add
  nothing** (§2.3).
* **ONNX / INT8 export** (`src/model/export.py`) — exports the visual tower and
  quantises it. 4× smaller, 1.3–1.8× faster, and **costs 0.01–0.02 AUROC** (§3.5).

## 2.2 How each module works

| module | role | key detail |
|---|---|---|
| `src/utils/config.py` | YAML configs with `extends:` inheritance, `${data_root}` interpolation, dotted CLI overrides | every experiment is a config, not an argument list |
| `src/utils/paths.py` | resolves data/results roots from env vars | no machine-specific path anywhere in the repo |
| `src/utils/seed.py` | seeds python, numpy, torch | reproducibility |
| `src/data/mvtec.py` | MVTec directory conventions; verifies downloads against published counts | a half-finished download looks valid; only counts reveal it |
| `src/data/splits.py` | reads Member B's split manifests; derives labels and mask paths | one definition of a split in the project |
| `src/model/encoder.py` | frozen OpenCLIP; returns pooled **and** patch features | patch tokens are what make pixel-level scoring possible |
| `src/model/memory.py` | append-only per-class prototype banks; cosine scoring | each class owns its bank → additions cannot disturb others |
| `src/model/adapter.py` | Tip-Adapter kernel `exp(−β(1−s))` + text blend | keys are an `nn.Parameter` so Tip-Adapter-F can unfreeze them |
| `src/model/export.py` | ONNX export + dynamic INT8 quantisation | disables PyTorch's fused attention so the graph is exportable |
| `src/eval/features.py` | feature bank cache (fp16) | extract once, reuse everywhere |
| `src/eval/pipeline.py` | runs one (category, shots, seed) split end to end | mask loading mirrors image preprocessing so they stay aligned |
| `src/eval/scoring.py` | map upsampling + Gaussian blur, metric aggregation | reflect padding, not zeros — zeros darken borders to 0.21 and every edge reads as anomalous |
| `src/eval/metrics.py` | **Member B** — image/pixel AUROC, F1-max | single source of truth for metrics |
| `src/eval/profiling.py` | latency, peak RAM, FLOPs, params, memory-search time | pins threads, discards warm-up, records machine load |

### Why "frozen" matters architecturally

Because the backbone never changes, a given image's features are constant
forever. They are extracted once and cached, so 15 categories × 3 shot counts ×
3 seeds × 2 scoring heads becomes vector arithmetic on cached tensors instead of
270 passes over the dataset. A warm re-run of the entire grid takes seconds.

## 2.3 Database used

**None — deliberately.** There is no SQL or NoSQL database. Persistence is:

| artefact | format | location | tracked in git |
|---|---|---|---|
| split manifests | JSON | `results/manifests/splits/` | ✅ yes |
| feature banks | PyTorch `.pt`, float16 | `$FEWSHOT_RESULTS_ROOT/feature_banks/` | ❌ regenerable |
| prototype memory | PyTorch `.pt` + JSON summary | results root | ❌ regenerable |
| benchmark results | CSV + `.meta.json` | results root | ❌ regenerable |
| result manifests | JSON | `results/manifests/` | ✅ yes |
| ONNX graphs | `.onnx` + JSON sidecar | results root | ❌ regenerable |

Rationale: an MSME deployment should be a folder, not a database installation.
The memory for 15 product classes at 5 shots is **43 MB** — a flat file is the
right tool.

## 2.4 Languages and libraries

**Language:** Python 3.13 (project targets 3.10+). No other language.

| library | version | used for |
|---|---|---|
| PyTorch | 2.8.0 **+cpu** | tensors, model execution |
| open_clip_torch | 3.3.0 | CLIP backbones |
| anomalib | 2.6.0 | PatchCore baseline |
| lightning | 2.6.5 | anomalib's trainer |
| scikit-learn | — | ROC / precision-recall curves |
| onnx / onnxruntime | 1.22.0 / 1.29.0 | export and INT8 |
| fvcore | — | FLOP counting |
| psutil | — | RAM and CPU load |
| matplotlib | — | figures |
| pytest | — | 159 tests |
| Streamlit | 1.47.0 | demo (`streamlit run app/demo.py`) |

---

# Part 3 — Machine Learning Specifics

## 3.1 Datasets

| dataset | categories | images | status |
|---|---|---|---|
| **MVTec AD** | 15 | **5,354** (3,629 train, 1,725 test) | ✅ downloaded, verified, fully evaluated |
| **VisA** | 12 | ~10,800 | ✅ downloaded, 3-seed splits, evaluated on both backbones |
| **NEU-DET** | 6 | 1,800 | ✅ one-class-as-normal splits, evaluated on 3 backbones, verified across machines |
| MPDD | 6 | — | optional, not started |

MVTec AD was verified **per category against published counts** — all 15 match
exactly. This caught a real failure: a first download reported success while
`leather` held 20 of 245 images and 8 categories were missing entirely.

**Splits.** Seeded 0/1/5/10-shot, generated by `scripts/generate_splits.py`, with
two properties enforced by tests:

* **Nested** — the 1-shot support set is a subset of the 5-shot, which is a subset
  of the 10-shot. A gain from more shots is therefore attributable to the added
  images, not a luckier draw.
* **Fixed test split** — shot count never changes the test images, so 1-shot and
  10-shot accuracy are measured on identical queries.

Three seeds (42, 1, 2) are committed → **135 evaluations** per backbone.

## 3.2 Preprocessing

Identical to CLIP's own evaluation transform, because the backbone was trained
with it:

```
Resize(224, interpolation=bicubic)
CenterCrop(224, 224)
convert to RGB
ToTensor()
Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
          std =(0.26862954, 0.26130258, 0.27577711))
```

**Ground-truth masks** get the *same geometry* — resize + centre-crop — with
**nearest-neighbour** interpolation to stay binary. If the mask transform did not
mirror the image transform, masks would not align with anomaly maps and pixel
AUROC would be quietly meaningless. Evaluation happens at 224×224.

No augmentation of any kind. There is no training, so there is nothing to
augment.

## 3.3 Techniques and models

**Main method — frozen CLIP + append-only prototype memory**

1. Encode k good images → k × 196 patch vectors of 768 dims, l2-normalised.
2. Store them in that class's bank (append; never re-fit).
3. For a query patch, compute cosine similarity to every stored prototype.
4. `anomaly = 1 − max similarity` (top-k averaged, k = 1).
5. Image score = max over patches. Pixel map = the 14×14 grid upsampled to 224
   and Gaussian-blurred.

**Baselines**

* **PatchCore** (anomalib) — WideResNet-50-2, layers 2+3, coreset ratio 0.1,
  9 neighbours. Full-shot reference.
* **CLIP zero-shot prompting** — prompt ensembles for "normal" and "defective",
  softmax over the two. *Only available with 512-dim projected features.*

**Tip-Adapter head — built, measured, and rejected**

Replaces the hard max with a kernel `A(s) = exp(−β(1−s))` over the whole cache,
optionally blended with text: `anomaly = (1−α)·cache + α·text`.

| shots | α=0 | α=0.1 | α=0.3 | α=0.5 | α=1.0 |
|---|---|---|---|---|---|
| 1 | **0.820** | 0.819 | 0.814 | 0.799 | 0.646 |
| 10 | **0.879** | 0.877 | 0.870 | 0.859 | 0.646 |

Text always hurts once ≥1 support image exists, and wider kernels never beat
nearest-neighbour. At its best setting (α=0, top_k=1) the adapter reproduces the
memory head **exactly** — because `exp(−β(1−s))` is strictly monotonic in `s`, so
it preserves ranking, and AUROC is rank-based. **It cannot help.** Only
Tip-Adapter-F, with trainable keys, remains worth attempting.

## 3.4 Parameters considered

| parameter | value | swept? |
|---|---|---|
| backbone | ViT-B/32, ViT-B/16, ViT-L/14 (all -quickgelu) | ✅ all three measured |
| patch feature dim | **768 unprojected** vs 512 projected | ✅ controlled ablation |
| shots (k) | 0, 1, 5, 10 | ✅ |
| seeds | 42, 1, 2 | ✅ |
| `top_k` neighbours | 1 | ✅ swept 1/3/5/10/25 |
| adapter `α` | 0.3 (config) | ✅ swept 0→1 |
| adapter `β` | 5.5 | fixed |
| softmax temperature | 0.07 | config-driven |
| Gaussian blur σ | 4.0 | fixed (standard) |
| input resolution | 224×224 | fixed |
| eval batch size | 16 | fixed |
| torch threads | 14, pinned | fixed for comparability |
| precision | FP32, INT8 (dynamic) | ✅ both measured |

> **The `-quickgelu` suffix is not cosmetic.** OpenAI CLIP weights were trained
> with QuickGELU activation. Pairing them with a plain `ViT-B-16` builds the wrong
> activation and open_clip only *warns*. We measured the damage: cosine similarity
> 0.9752 between correct and incorrect features. A test now makes this a hard
> error.

## 3.5 Evaluation criteria

| metric | question it answers |
|---|---|
| **Image AUROC** | is this part defective? |
| **Pixel AUROC** | where is the defect? |
| **Image F1-max** | best achievable F1 over all thresholds |
| **Pixel F1-max** | same, per pixel |

F1-**max** rather than F1 at a fixed threshold: no operating threshold has been
chosen, and that is a deployment decision.

**Cost metrics** (equally weighted in this project): CPU latency at batch size 1
(median and p90), peak RAM, GFLOPs, parameter count, memory-search time, bytes
per class.

### Headline results — ViT-B/16, 15 categories, 3 seeds

| | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| **image AUROC** | 0.856 | 0.888 | 0.904 |
| **pixel AUROC** | 0.916 | 0.932 | **0.936** |

| backbone | image 10-shot | pixel 10-shot | GFLOPs | ms/image burst | ms sustained | RAM |
|---|---|---|---|---|---|---|
| ViT-B/32 | 0.904 | **0.940** | **3.0** | **37** | ~230 | 1.5 GB |
| ViT-B/16 | 0.904 | 0.936 | 11.3 | 88 | ~650 | 1.5 GB |
| ViT-L/14 | **0.933** | 0.945 | 52.0 | 397 | ~2,000 | 3.7 GB |

Torch, 6 threads, batch 1. **Sustained** = after ~5 minutes of continuous
inference: the laptop steps its clock down and runs 5–6× slower. Plan capacity
on that figure — B/16 still inspects 10,000 parts in under 2 hours a day.

**On MVTec AD the two backbones are statistically indistinguishable** — every
paired difference is under 2× its standard error. So there ViT-B/32 delivers the
same accuracy for **3.7× fewer FLOPs, 2.2× lower latency and 4× less memory**.

### Second dataset — VisA, 12 categories, 3 seeds

| | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| image AUROC, B/16 | 0.773 | 0.811 | 0.831 |
| pixel AUROC, B/16 | 0.934 | 0.947 | **0.951** |
| image AUROC, B/32 | 0.681 | 0.761 | 0.779 |

**On VisA, B/16 wins clearly**: +0.05 to +0.09 image AUROC, 4.4–5.5× the paired
standard error. VisA defects cover 0.04–0.9% of the image against 4.6% on MVTec;
a B/32 patch is 2% of the frame, so most VisA defects are smaller than one patch.
The backbone choice depends on defect size, and that is a more useful rule for
an MSME than "always use the cheap one".

### Third backbone — ViT-L/14

| 10-shot | MVTec image | MVTec pixel | VisA image | VisA pixel |
|---|---|---|---|---|
| ViT-L/14 | **0.933** | 0.945 | 0.843 | **0.964** |
| gain over B/16 | +0.030 (4.6× SE) | +0.009 | +0.012 (not significant) | +0.013 (3.9× SE) |

Most accurate on both datasets; the biggest gains are the fine-detail categories
(`toothbrush` +0.117, `screw` +0.086). Cost: 1.7× B/16's memory per class.

### The proposed system — windows + text fusion (ViT-B/16)

| 10-shot image AUROC | MVTec | VisA |
|---|---|---|
| prototype memory alone | 0.904 | 0.831 |
| + multi-scale windows | 0.921 | 0.851 |
| + text-prompt fusion (w = 0.5) | 0.911 | 0.837 |
| **both** (`configs/proposed.yaml`) | **0.925** | **0.855** |

Both additions are single-pass: no extra encoder call. Text helps most at 1 shot
(+0.016 MVTec, +0.018 VisA) and lets a worker type what a defect looks like.

### INT8 — MVTec AD, 3 seeds, paired against FP32

| backbone | image AUROC 10-shot | pixel AUROC 10-shot | ms/image (ONNX, 6 threads) | model |
|---|---|---|---|---|
| ViT-B/32 FP32 → INT8 | 0.904 → 0.891 | 0.940 → 0.933 | 40 → **30** (1.3×) | 352 → 89 MB |
| ViT-B/16 FP32 → INT8 | 0.904 → 0.885 | 0.936 → 0.920 | 132 → 72 (1.8×) | 345 → 88 MB |

INT8 costs 0.005–0.02 image AUROC on both datasets. On VisA, INT8 B/16 keeps
0.819 at 10 shots — well above FP32 B/32's 0.779 — at half FP32 B/16's latency.
**Large defects: INT8 B/32** (cheapest measured). **Small defects: INT8 B/16.**
Precision never changes which backbone to pick.

### Other measured findings

* **More shots buy predictability, not just accuracy.** Seed-to-seed std falls
  0.027 → 0.013 → 0.012 from 1 to 10 shots.
* **768-dim unprojected beats 512-dim projected** by ~+0.02, at ~4× the paired
  standard error. CLIP's text-alignment projection discards detail useful for
  image-to-image matching.
* **Memory search is ~1% of inference cost** (0.3–2.9 ms against 37–397 ms), so
  extra shots are effectively free.
* **Append-only verified**: 15 classes added sequentially, all prior scores
  bit-identical, ~1.4-1.9 ms and 3.0 MB per class.

## 3.6 Training configuration

> **There is no training.** This is the central design decision, not an omission.

| | |
|---|---|
| epochs | **none** |
| activation function | GELU / QuickGELU — **inherited frozen**, never trained |
| batch size | 16 — **inference only** |
| learning rate | **none** — no optimiser exists |
| loss function | **none** |
| optimiser | **none** |
| trainable parameters | **0 of 150 M** (`requires_grad=False`, asserted by test) |
| gradient steps taken | **0** |

"Fitting" a class = appending l2-normalised vectors to a list: **~1.4-1.9 ms**, no
gradients, no GPU.

`configs/adapter.yaml` contains a `training:` block (epochs 20, lr 0.001, AdamW,
cosine schedule, weight decay 0.01, batch size 8). **Nothing reads it.** It is
reserved for Tip-Adapter-F in Week 5 and is explicitly marked unread in the
config so no one assumes it took effect.

## 3.7 Transfer learning / pre-trained models

Transfer learning is the *entire* method: a model trained on internet
image–text pairs is applied to industrial inspection **without adaptation**.

| model | weights | params | role |
|---|---|---|---|
| **ViT-B/16-quickgelu** | OpenAI CLIP (WIT-400M) | 150 M | primary backbone |
| **ViT-B/32-quickgelu** | OpenAI CLIP (WIT-400M) | 151 M | cheap ablation |
| ViT-L/14-quickgelu | OpenAI CLIP | 428 M | most accurate; evaluated on both datasets |
| WideResNet-50-2 | ImageNet (timm) | 69 M | PatchCore baseline only |

**Why CLIP rather than an ImageNet CNN?** Its joint image–text space makes the
system *open-vocabulary*: a new product class is named, not enumerated in a fixed
label set. We exploit that for zero-shot prompting; the few-shot path uses only
the visual tower.

**Fine-tuning: none.** No layer is unfrozen, no head is trained, no adapter is
optimised. That is what makes deployment a file copy and class addition
provably non-disruptive.

---

# Part 4 — Honest Status

## Completed

Repo and config layer · PatchCore validation reproducing published numbers ·
MVTec AD downloaded and verified (5,354/5,354) · seeded 3-seed splits · frozen
encoder · append-only prototype memory · Tip-Adapter (evaluated, rejected) ·
evaluation harness · CPU profiling harness · full MVTec benchmark on two
backbones · dimension ablation · incremental-class demonstration · frontier
figures · VisA on three backbones · ViT-L/14 · INT8 accuracy and latency ·
windowed + text-fused scoring · held-out class append vs from-scratch ·
sustained-load latency · peak RAM · ₹0 cost table · Streamlit demo with typed
defect descriptions · architecture freeze · paper draft · **187 tests**.

## Not yet done

| item | why it matters |
|---|---|
| **MPDD, static INT8** | optional; cut first under the Blueprint's cut order |
| **WinCLIP / AnomalyCLIP reproduction** | cited, not reproduced; WinCLIP needs one encoder pass per window |

## Known limitations

* **Zero-shot localisation fails.** At 512 dims, zero-shot pixel AUROC was
  **0.323 — below chance**; the text-prompted map points away from the defect.
  Raw CLIP patch tokens are poorly aligned for dense prediction, which is why
  WinCLIP uses windowed aggregation. Under the 768-dim convention, prompting is
  unavailable entirely.
* **`screw` is a resolution failure**, not a method failure: 0.705 at 10 shots
  with B/16, 0.792 with L/14's finer grid. Its defects are thinner than a patch.
* **NEU-DET is image-level only** (no masks), and text fusion hurts there: its
  "normal" class is itself a defect type.
* **Small defects cost detection accuracy.** VisA image AUROC is 0.831 at 10 shots
  against 0.904 on MVTec, even though pixel AUROC is higher (0.951).
* **`toothbrush` remains non-monotonic** — the smallest category (60 train, 42
  test), ±0.055 spread at 5 shots.

---

# Part 5 — Three Slides Worth Building

**1. The value of one photograph.**
Pixel AUROC goes from 0.323 (text prompts, worse than chance) to **0.916** with a
single good photo. That is the project in one number.

**2. Defect size decides the backbone.**
On MVTec AD, ViT-B/32 equals ViT-B/16 to four decimal places at 10 shots for 3.7×
fewer FLOPs. On VisA, where defects are smaller than one B/32 patch, B/16 wins by
+0.05 AUROC at 5× the standard error. One measurable rule — defect area against
patch area — tells a workshop which one to buy compute for.

**3. Adding a product cannot break the ones already running.**
15 classes added one at a time; after every addition, every previous class scored
**bit-identically**. ~1.4-1.9 ms and 3.0 MB each. This is the property a factory
actually needs, and it is verified with `torch.equal`, not argued.
