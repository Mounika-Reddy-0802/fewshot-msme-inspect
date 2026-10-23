# Results: frozen CLIP + prototype memory on MVTec AD, VisA and NEU-DET

Reproducible from the committed manifests and the scripts in `scripts/`. Raw CSVs
live outside the repo (gitignored); the manifests identifying each run are
committed.

**Setup.** Frozen OpenCLIP ViT-B/16, no training of any kind. Support images are
normal only. 15 MVTec AD categories (VisA's 12 in section 6), shot counts 1/5/10, evaluated on the fixed
test split so numbers are comparable across shot counts. CPU only: Intel
i7-13700H, 14 threads pinned, no GPU anywhere.

**Splits** come from Member B's generator (`scripts/generate_splits.py`) --
one definition of a split for the whole project. Her shot sets are nested
(1 subset of 5 subset of 10), so adding shots is measured against the same
starting images. **Three seeds** (42, 1, 2) are committed, and every number below
is a mean over them with the seed-to-seed spread reported.

**Features** are the raw 768-dim transformer tokens, matching her feature-bank
convention (section 3 measures what that choice is worth).

One consequence of that convention, stated up front: **there is no 0-shot
column.** 768-dim patch tokens are not in the joint image-text space, so a text
prompt cannot score a patch. Section 12 covers what that costs -- little.

---

## 1. Few-shot accuracy

Image AUROC, ViT-B/16, prototype memory. Mean over 3 seeds, +/- population std.

| category | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| leather | 0.989 +/-0.001 | 0.995 +/-0.007 | 0.996 +/-0.005 |
| bottle | 0.985 +/-0.007 | 0.992 +/-0.001 | 0.994 +/-0.005 |
| tile | 0.986 +/-0.010 | 0.992 +/-0.004 | 0.992 +/-0.003 |
| carpet | 0.952 +/-0.037 | 0.987 +/-0.011 | 0.982 +/-0.006 |
| metal_nut | 0.938 +/-0.004 | 0.961 +/-0.020 | 0.978 +/-0.003 |
| wood | 0.971 +/-0.007 | 0.966 +/-0.007 | 0.972 +/-0.008 |
| grid | 0.884 +/-0.023 | 0.914 +/-0.012 | 0.931 +/-0.013 |
| zipper | 0.858 +/-0.043 | 0.914 +/-0.004 | 0.917 +/-0.006 |
| pill | 0.888 +/-0.002 | 0.893 +/-0.002 | 0.901 +/-0.014 |
| hazelnut | 0.774 +/-0.035 | 0.882 +/-0.013 | 0.899 +/-0.025 |
| capsule | 0.757 +/-0.102 | 0.828 +/-0.022 | 0.874 +/-0.009 |
| cable | 0.772 +/-0.067 | 0.860 +/-0.022 | 0.858 +/-0.009 |
| toothbrush | 0.762 +/-0.043 | 0.736 +/-0.055 | 0.829 +/-0.047 |
| transistor | 0.678 +/-0.013 | 0.718 +/-0.014 | 0.733 +/-0.011 |
| screw | 0.643 +/-0.011 | 0.684 +/-0.004 | 0.705 +/-0.022 |
| **mean** | **0.856** | **0.888** | **0.904** |

Pixel AUROC on the same runs -- the harder question, since it asks *where* the
defect is, not merely whether there is one.

| category | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| leather | 0.985 +/-0.003 | 0.986 +/-0.000 | 0.986 +/-0.000 |
| carpet | 0.985 +/-0.005 | 0.986 +/-0.003 | 0.983 +/-0.002 |
| toothbrush | 0.978 +/-0.001 | 0.984 +/-0.002 | 0.986 +/-0.000 |
| bottle | 0.972 +/-0.002 | 0.976 +/-0.001 | 0.977 +/-0.000 |
| hazelnut | 0.965 +/-0.007 | 0.980 +/-0.002 | 0.981 +/-0.002 |
| pill | 0.933 +/-0.003 | 0.942 +/-0.001 | 0.946 +/-0.001 |
| tile | 0.918 +/-0.004 | 0.928 +/-0.002 | 0.931 +/-0.002 |
| wood | 0.916 +/-0.011 | 0.928 +/-0.004 | 0.929 +/-0.003 |
| capsule | 0.914 +/-0.004 | 0.930 +/-0.001 | 0.935 +/-0.001 |
| screw | 0.901 +/-0.013 | 0.916 +/-0.004 | 0.923 +/-0.002 |
| grid | 0.882 +/-0.015 | 0.916 +/-0.003 | 0.925 +/-0.005 |
| cable | 0.874 +/-0.010 | 0.881 +/-0.007 | 0.889 +/-0.005 |
| zipper | 0.843 +/-0.018 | 0.875 +/-0.010 | 0.883 +/-0.004 |
| metal_nut | 0.861 +/-0.006 | 0.898 +/-0.003 | 0.911 +/-0.006 |
| transistor | 0.807 +/-0.028 | 0.848 +/-0.018 | 0.858 +/-0.002 |
| **mean** | **0.916** | **0.932** | **0.936** |

### More shots buy predictability, not just accuracy

Mean seed-to-seed standard deviation falls sharply with shot count:

| shots | mean image AUROC | mean std across seeds |
|---|---|---|
| 1 | 0.856 | **0.027** |
| 5 | 0.888 | 0.013 |
| 10 | 0.904 | 0.012 |

A workshop deploying at 1 shot faces real variance depending on *which* good
photo it happened to take -- `capsule` swings +/-0.102, `cable` +/-0.067. By 10
shots that is halved. This is invisible with a single seed, and is an argument
for taking ten reference photos rather than one that has nothing to do with mean
accuracy.

Pixel AUROC of **0.920 from a single photo** is the strongest result here. For
scale, PatchCore -- trained on all 209 normal images of `bottle` -- reaches 0.986
on that category; one photo gets 0.973 on the same category.

---

## 2. Backbone: the cheap one is not worse

Three seeds, 15 categories, paired (both backbones scored on identical splits).

| backbone | patches | image 1/5/10 | pixel 1/5/10 | GFLOPs | ms/image | prototypes/class @10-shot |
|---|---|---|---|---|---|---|
| ViT-B/16 | 14x14 = 196 | 0.856 / 0.888 / 0.904 | 0.916 / 0.932 / 0.936 | 11.3 | 132 | 1,960 |
| ViT-B/32 | 7x7 = 49 | 0.837 / 0.884 / 0.904 | **0.920** / **0.934** / **0.940** | **3.0** | **60** | **490** |

**No difference between them survives its error bars.** Paired over 45
measurements per cell:

| metric | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| image, B/16 - B/32 | +0.019 (1.2x SE) | +0.004 (0.3x SE) | +0.000 (0.0x SE) |
| pixel, B/16 - B/32 | -0.004 (0.9x SE) | -0.003 (0.6x SE) | -0.004 (0.7x SE) |

Nothing reaches 2x its standard error. At 10 shots the two are identical on image
AUROC to four decimal places, and B/32 is fractionally *ahead* on pixel AUROC at
every shot count while storing a quarter as many prototypes.

So the finding is this: **ViT-B/32 matches ViT-B/16 for 3.7x fewer FLOPs, 2.2x
lower latency and 4x less memory.** For an MSME choosing hardware, the cheap
backbone is not a compromise. The only place B/16 may hold an edge is 1-shot
image detection (+0.019), and even that is 1.2x SE -- suggestive, not established.

**This holds on MVTec AD only.** On VisA, whose defects are far smaller, B/16 wins
clearly at every shot count (section 6). The finding is "B/32 suffices when
defects are large relative to a 32px patch", not "B/32 suffices".

### This claim took three attempts; the first two were wrong

It is worth recording how, because the same trap will recur.

1. **"B/32 beats B/16 at every shot count."** Measured with 512-dim projected
   features on a different set of splits. Wrong -- it did not survive the change
   of feature convention.
2. **"B/16 wins at 1 and 10 shots."** Measured at 768 dims but on a single seed,
   with no error bars. Unsupported -- the gaps were inside seed noise.
3. **"They are statistically indistinguishable."** Three seeds, paired design,
   differences compared against the paired standard error. This one is checkable.

Each earlier version looked like a clean result at the time. What separated them
was not more data but a control: change one variable, pair the comparison, and
measure the effect against the right spread.

---

## 3. What the feature dimension is worth

Committed record: `results/manifests/dim_ablation.json`.

CLIP's visual tower emits 768-dim tokens and then projects them to 512 dims to
align with text. For image-to-image matching that projection is not obviously
useful, so it was measured directly -- **identical splits, identical shot counts,
identical backbone, only the projection differs**:

Three seeds, 15 categories, 45 paired measurements per shot count:

| shots | 512 projected | 768 unprojected | delta | vs paired SE | categories favouring 768 |
|---|---|---|---|---|---|
| 1 | 0.8294 | **0.8558** | +0.026 | 4.3x | 13/15 |
| 5 | 0.8686 | **0.8881** | +0.020 | 3.7x | 11/15 |
| 10 | 0.8837 | **0.9040** | +0.020 | 4.4x | 12/15 |

Around **+0.02**, in the same direction at every shot count, at roughly four
times the paired standard error. Member B's convention is the right one.

**Two methodological notes, each worth more than the result itself.**

*Change one thing at a time.* The first comparison was run against the earlier
512-dim results, which used different splits, and showed +0.048 at 1-shot. That
changed the feature dimension *and* the splits together; roughly two thirds of
the apparent gain was the split change.

*Pair the comparison, then use the paired spread.* A single-seed version of the
controlled experiment gave +0.014, and the marginal seed-to-seed spread at 1-shot
is 0.027 -- larger than the effect, which looked fatal. It is not, because the
design is paired: the same split is scored two ways, so split luck cancels and
the relevant quantity is the paired standard error (0.006). Comparing an effect
against the wrong spread would have discarded a real result.

---

## 4. Compute cost

Records: `torch_latency.json`, `sustained_torch.json`, `peak_ram.json`,
`cost_feasibility.json` (all under `results/manifests/`).

**Burst latency.** Torch FP32, 6 threads (one per performance core; section 7
explains why not 14), batch size 1 -- a workshop inspects one part at a time --
5 warm-up passes discarded, median of 20, machine idle (7-15% background load):

| backbone | params | GFLOPs | ms/image | memory search 1-shot | 10-shot | RAM |
|---|---|---|---|---|---|---|
| ViT-B/32 | 151M | 3.0 | **37** | 0.26 ms | 0.39 ms | 1.47 GB |
| ViT-B/16 | 150M | 11.3 | 88 | 0.69 ms | 1.70 ms | 1.46 GB |
| ViT-L/14 | 428M | 52.0 | 397 | 0.87 ms | 2.94 ms | 3.68 GB |

RAM is the inspector's peak working set above an idle Python process, measured in
a fresh process per configuration. Torch loads CLIP's text tower too; the ONNX
visual tower alone needs 0.64-0.67 GB in FP32 and 0.38-0.39 GB in INT8.

(The first version of this table used 14 threads and read 60 / 132 ms; the
6-thread numbers are lower and repeatable. The old run is kept as
`profiling/backbones_14threads_d512.csv` outside the repo.)

**The memory search is about 1% of inference cost.** Going from 1 shot to 10
adds 0.1-2 ms to a 37-397 ms budget, so shots are effectively free. The backbone
is the only meaningful cost lever.

### Sustained load: the laptop slows 5-6x

Ten minutes of back-to-back inference per backbone, median per 30-second window:

| backbone | first window | after ~7 min | slowdown |
|---|---|---|---|
| ViT-B/32 | 41 ms | 225-255 ms | **5.8x** |
| ViT-B/16 | 472 ms (already hot) | 630-680 ms | -- |
| ViT-L/14 | 2,000 ms (already hot) | 1,930-2,040 ms | -- |

B/32 ran first on a cool machine: 40 ms for four minutes, then a step between
minutes 4.5 and 7 to ~230 ms, where it stayed. The reported CPU clock dropped
from 2,400 to 1,520 MHz in the same windows. B/16 and L/14 started on a machine
that had already stepped down, so their whole runs are the throttled state
(B/16's 472-680 ms against 88 ms burst; L/14's ~2 s against 397 ms). This is the
laptop's power/thermal management, not the method, and it confirms the earlier
unplanned observation of a 3.6x slowdown after an hour of benchmarking.

**What it means for an MSME.** Plan capacity on the sustained figure. At 10,000
parts a day, B/16 keeps the laptop busy ~1.8 hours and B/32 ~0.7 hours even
throttled; L/14 needs ~5.6 hours and 3.7 GB. Burst numbers are what a reviewer
compares with other papers; sustained numbers are what a shop floor gets.

### The Rs 0 table

Assumptions: 45 W CPU (the i7-13700H base power; the whole laptop draws more),
Rs 8/kWh, 10,000 parts/day, 300 days/year. Energy uses burst latency.

| config | ms burst | ms sustained | Rs per year | busy h/day (worst) | RAM MB |
|---|---|---|---|---|---|
| B/32 ONNX INT8 | 33 | -- | 9.80 | 0.09 | 378 |
| B/32 ONNX FP32 | 41 | -- | 12.28 | 0.11 | 667 |
| B/16 ONNX INT8 | 72 | -- | 21.67 | 0.20 | 390 |
| B/16 ONNX FP32 | 133 | -- | 40.04 | 0.37 | 641 |
| B/32 torch FP32 | 37 | 239 | 11.02 | 0.67 | 1467 |
| B/16 torch FP32 | 88 | 656 | 26.36 | 1.82 | 1455 |
| L/14 torch FP32 | 397 | 2012 | 119.14 | 5.59 | 3680 |
| *PatchCore, 10 photos* | 93 | -- | 27.99 | 0.26 | 596 |
| *PatchCore, all 209 photos* | 170 | -- | 51.00 | 0.47 | 1210 |

**The baseline's "before" numbers** (`patchcore_cost_i7_10shot.json`,
`patchcore_cost_i7_fullshot.json`), timed with `scripts/profile_patchcore.py` on
the same laptop and harness, each run in a fresh process after a 5-minute
cool-down, repeats within 10%. PatchCore given our 10 photos runs at 93 ms and
0.6 GB -- about the same as our B/16 in torch (88 ms) and 2.8x slower than our
cheapest configuration, B/32 INT8 (33 ms). Full PatchCore, fitted on all 209
bottle images, needs a 17.5-minute fit and runs at 170 ms with a 131 MB memory
bank; adding a product to our memory takes 2-4 s (section 5). One timing taken
two minutes after another run on a hot CPU read 553 ms and was discarded -- the
same throttling as above. Member B's earlier profile on a 2-core laptop under
34% load (806 ms, `patchcore_cost.json`) is kept as a second-machine figure.

Hardware spend: Rs 0 (an existing laptop). GPU: none used in this session's
runs. The whole study's benchmarks used at least 6.8 CPU-hours, about Rs 2.45 of
electricity. A commercial machine-vision comparison line is not included until a
cited quote is available (`--commercial-inr`).

---

## 5. Append-only incremental class addition

Committed record: `results/manifests/incremental_5shot.json`.

All 15 categories added to one memory, one at a time. After every addition, every
previously added class was re-scored and compared with `torch.equal`.

| | |
|---|---|
| classes added | 15 |
| cost per class | **~1.4-1.9 ms, 3.0 MB** (two machines) |
| total memory, 15 classes | 43.1 MiB |
| mean image AUROC (5-shot) | 0.886 |
| existing-class scores after each addition | **bit-identical** |

(3.0 MB per class rather than the 2.0 MB reported earlier: 768-dim features are
1.5x the size of 512-dim ones. The accuracy that buys is in section 3.)

Bit-identity is the claim worth making. Methods that re-fit a shared component --
a normaliser, a codebook, a classifier head -- shift existing decision boundaries
when a class arrives. Here each class owns its bank, so adding a new part
provably cannot alter inspection of parts already in production.

Millisecond timings for an operation this small are load-sensitive (a second run
measured 6.5 ms total rather than 15.8 ms). The per-class cost in bytes is the
robust figure.

### Held-out products, against from-scratch

Records: `incremental_holdout_5shot.json`, `incremental_holdout_10shot.json`.

The Blueprint's harder version: twelve MVTec products are "in production"; three
held out in advance (the alphabetically last: transistor, wood, zipper) are
appended from 5 or 10 photos. Three seeds, both shot counts:

| check | result |
|---|---|
| the 12 existing products after the appends | **bit-identical** scores, every seed |
| appended product vs a memory built with all 15 from the start | **bit-identical**, every seed |
| 10-shot image AUROC of the appended products | transistor 0.733, wood 0.972, zipper 0.917 |
| end-to-end append (read photos, encode, store), idle machine | **2.0-2.3 s** at 5 shots, 3.3-3.7 s at 10 |

"Appended equals from-scratch" is the expected answer for a memory with no fitted
parameters; it is checked, not assumed. The cost is almost entirely encoding the
photos -- the store itself is ~2 ms. Nothing was tuned on the held-out products:
the memory has no learned parameters and top_k = 1 is the default.

---

## 6. Second dataset: VisA

Committed records: `results/manifests/visa_backbones.json`,
`results/manifests/figures_visa.json`.

Same method, nothing retuned: frozen CLIP, 768-dim patches, prototype memory,
top_k 1. 12 categories, 1/5/10 shots, three seeds (42, 1, 2) from Member B's
VisA manifests. Image AUROC, ViT-B/16, mean over seeds, +/- population std:

| category | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| pipe_fryum | 0.933 +/-0.035 | 0.972 +/-0.015 | 0.979 +/-0.008 |
| chewinggum | 0.891 +/-0.056 | 0.936 +/-0.012 | 0.942 +/-0.018 |
| cashew | 0.925 +/-0.006 | 0.929 +/-0.012 | 0.927 +/-0.011 |
| pcb4 | 0.716 +/-0.126 | 0.865 +/-0.049 | 0.892 +/-0.018 |
| fryum | 0.887 +/-0.005 | 0.879 +/-0.024 | 0.888 +/-0.009 |
| candle | 0.774 +/-0.044 | 0.850 +/-0.016 | 0.881 +/-0.005 |
| macaroni1 | 0.750 +/-0.023 | 0.786 +/-0.031 | 0.832 +/-0.016 |
| pcb1 | 0.785 +/-0.033 | 0.786 +/-0.054 | 0.769 +/-0.024 |
| pcb2 | 0.627 +/-0.014 | 0.670 +/-0.055 | 0.731 +/-0.023 |
| macaroni2 | 0.690 +/-0.012 | 0.694 +/-0.013 | 0.725 +/-0.016 |
| capsules | 0.725 +/-0.016 | 0.703 +/-0.023 | 0.711 +/-0.018 |
| pcb3 | 0.578 +/-0.015 | 0.661 +/-0.028 | 0.696 +/-0.029 |
| **mean** | **0.773** | **0.811** | **0.831** |

Pixel AUROC:

| category | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| cashew | 0.978 +/-0.003 | 0.985 +/-0.001 | 0.985 +/-0.001 |
| pcb1 | 0.966 +/-0.005 | 0.980 +/-0.003 | 0.985 +/-0.000 |
| chewinggum | 0.979 +/-0.003 | 0.982 +/-0.002 | 0.983 +/-0.002 |
| pipe_fryum | 0.974 +/-0.004 | 0.979 +/-0.002 | 0.980 +/-0.001 |
| macaroni1 | 0.922 +/-0.023 | 0.944 +/-0.007 | 0.955 +/-0.006 |
| pcb3 | 0.907 +/-0.013 | 0.945 +/-0.003 | 0.953 +/-0.002 |
| fryum | 0.925 +/-0.006 | 0.941 +/-0.003 | 0.945 +/-0.002 |
| pcb4 | 0.912 +/-0.004 | 0.941 +/-0.000 | 0.944 +/-0.002 |
| capsules | 0.927 +/-0.004 | 0.935 +/-0.004 | 0.939 +/-0.002 |
| pcb2 | 0.903 +/-0.019 | 0.926 +/-0.005 | 0.933 +/-0.001 |
| candle | 0.914 +/-0.015 | 0.925 +/-0.002 | 0.931 +/-0.002 |
| macaroni2 | 0.901 +/-0.023 | 0.884 +/-0.013 | 0.880 +/-0.015 |
| **mean** | **0.934** | **0.947** | **0.951** |

**Detection is harder than on MVTec; localisation is not.** Image AUROC drops by
about 0.08 against MVTec (0.831 vs 0.904 at 10 shots), while pixel AUROC is
actually *higher* (0.951 vs 0.936). The anomaly maps find VisA defects well; the
image score (max over patches) has trouble separating a tiny real defect from
the ordinary variation in a normal part. pcb3 at 0.696 and capsules at 0.711
are the weakest.

### The backbone result does not transfer

Paired over 36 measurements per shot count (12 categories x 3 seeds), delta is
B/16 - B/32:

| metric | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| image AUROC | +0.093 (5.5x SE) | +0.050 (4.4x SE) | +0.052 (5.1x SE) |
| pixel AUROC | +0.029 (4.1x SE) | +0.017 (3.5x SE) | +0.016 (3.4x SE) |
| categories favouring B/16 (image) | 10/12 | 9/12 | 10/12 |

On MVTec nothing reached 2x SE. On VisA everything exceeds 3x. The B/32 means
are 0.681 / 0.761 / 0.779 image and 0.905 / 0.930 / 0.935 pixel.

**Why: defect size against patch size.** Measured from the ground-truth masks,
a VisA defect covers **0.04-0.9%** of the image (macaroni2 0.04%, capsules 0.09%,
pcb3 0.62%, chewinggum 0.89%); an MVTec defect averages **4.6%**. At 224px one
B/32 patch is 1/49 = 2.0% of the image and one B/16 patch 1/196 = 0.5%. Most VisA
defects are smaller than a B/32 patch and get averaged away inside it. The
largest B/16 gains at 10 shots are exactly the small-defect categories --
capsules +0.147, macaroni1 +0.120, macaroni2 +0.119, candle +0.093.

This is the same mechanism as the `screw` failure on MVTec (section 12), now
visible across a whole dataset. For an MSME the practical rule is: if the
defects that matter are smaller than about 2% of the frame, pay for B/16 (or
crop tighter); otherwise B/32 is enough.

**Seeds matter more here.** Mean seed-to-seed std of image AUROC for B/16 is
0.032 / 0.028 / 0.016 at 1 / 5 / 10 shots, against 0.027 / 0.013 / 0.012 on
MVTec. pcb4 swings +/-0.126 at 1 shot.

**Monotonicity.** 8/12 categories improve monotonically for B/16 (10/12 for
B/32). The exceptions -- pcb1, cashew, fryum, capsules -- are within their seed
spread.

**Centre-crop loss is negligible.** VisA images are ~1500x1000; CLIP's centre
crop discards the outer sixth on each long side. Checked against the masks, 11
of 1,200 anomalous test images (4 capsules, 7 macaroni2) have their defect
entirely outside the crop -- under 1%, and identical for both backbones, so the
paired comparison is unaffected.

---

## 7. INT8: what quantisation costs and buys

Committed records: `results/manifests/int8_vit_b16.json`,
`results/manifests/int8_vit_b32.json`, `results/manifests/int8_latency.json`,
`results/manifests/onnx_export.json`.

The visual tower is exported to ONNX at 768 dims (cosine 1.00000 against torch,
so the export itself is exact) and dynamically quantised to INT8 weights -- no
calibration set, nothing an MSME would need to collect. Then the full MVTec grid
is re-run on the INT8 graph: same splits, 15 categories x 3 seeds, paired
against FP32. Delta is INT8 - FP32:

| backbone | metric | 1-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| ViT-B/16 | image AUROC | -0.019 (2.6x SE) | -0.020 (2.6x SE) | -0.019 (2.9x SE) |
| ViT-B/16 | pixel AUROC | -0.013 (5.3x SE) | -0.015 (8.4x SE) | -0.016 (7.8x SE) |
| ViT-B/32 | image AUROC | -0.004 (0.7x SE) | -0.009 (2.7x SE) | -0.013 (3.6x SE) |
| ViT-B/32 | pixel AUROC | -0.006 (3.9x SE) | -0.007 (5.3x SE) | -0.007 (5.7x SE) |

**INT8 costs accuracy, and B/16 pays about twice what B/32 does.** Pixel AUROC
falls in 14/15 categories for both. For B/16 the image loss is concentrated:
`cable` -0.107, `screw` -0.061, `hazelnut` and `metal_nut` -0.050. The export
smoke test predicted the ordering: FP32-vs-INT8 patch cosine was 0.873 for B/16
and 0.971 for B/32.

What it buys, ONNX FP32 against ONNX INT8 on the same runtime, batch size 1,
6 threads, two alternating runs each:

| backbone | FP32 ms | INT8 ms | speed-up | FP32 MB | INT8 MB |
|---|---|---|---|---|---|
| ViT-B/32 | 39-41 | 27-33 | **1.3x** | 352 | 89 |
| ViT-B/16 | 131-133 | 71-72 | **1.8x** | 345 | 88 |

Every graph is 4x smaller on disk.

### The same on VisA

Records: `results/manifests/int8_vit_b16_visa.json`,
`results/manifests/int8_vit_b32_visa.json`. 12 categories x 3 seeds, paired:

| backbone | metric | 1-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| ViT-B/16 | image AUROC | -0.016 (2.4x SE) | -0.005 (0.9x SE) | -0.012 (2.9x SE) |
| ViT-B/16 | pixel AUROC | -0.000 (0.1x SE) | -0.002 (1.3x SE) | -0.003 (1.7x SE) |
| ViT-B/32 | image AUROC | -0.011 (1.7x SE) | -0.014 (2.9x SE) | -0.014 (3.6x SE) |
| ViT-B/32 | pixel AUROC | -0.007 (2.8x SE) | -0.006 (3.7x SE) | -0.005 (3.6x SE) |

**The MVTec pattern does not repeat.** On VisA, B/16 loses no more than B/32 --
its pixel AUROC is essentially untouched -- and the worst category loss is
`macaroni1` -0.046 rather than MVTec's `cable` -0.107. How much INT8 costs
depends on the data, not only on how far the features move (the export's
patch-cosine check ranked B/16 as the more fragile backbone; on VisA it is not).

### The trade, for an MSME

| 10-shot image AUROC | MVTec | VisA | ms/image (ONNX, 6 threads) |
|---|---|---|---|
| B/32 FP32 | 0.904 | 0.779 | 40 |
| B/32 INT8 | 0.891 | 0.765 | **30** |
| B/16 FP32 | 0.904 | **0.831** | 132 |
| B/16 INT8 | 0.885 | 0.819 | 72 |

* **Large defects (MVTec-like): INT8 B/32.** Cheapest measured, within 0.013 of
  the best.
* **Small defects (VisA-like): INT8 B/16.** It keeps 0.819 against FP32 B/32's
  0.779 and FP32 B/16's 0.831, at about half FP32 B/16's latency. The backbone
  matters far more than the precision here.
* INT8 costs 0.005-0.02 image AUROC everywhere it was measured; nowhere does it
  change which backbone should be chosen.

### 14 threads was the wrong setting for ONNX

The first profiling pass used the same 14 pinned threads as the torch numbers
(section 4) and was unusable: FP32 B/32 measured 36 ms in one run and 526 ms in
the next, identical settings. This i7-13700H has 6 performance and 8 efficiency
cores; 14 threads put work on both, and every parallel op then waits for the
slowest core, while OneDrive sync was competing for the same cores. At 6 threads
repeats agree within ~20% even at 50-60% background load. Two consequences:

* The ONNX timings above are not comparable with the torch timings in section 4
  (different runtime and thread count). Compare INT8 with ONNX FP32 only.
* The section 4 torch numbers were first taken at 14 threads too; they have
  since been re-measured at 6 threads (37 / 88 ms rather than 60 / 132 ms).

---

## 8. The third backbone: ViT-L/14

Records: `results/manifests/mvtec_l14_vs_b16.json`,
`results/manifests/visa_l14_vs_b16.json`.

ViT-L/14 (428M parameters, 16x16 = 256 patches of 1024 dims at 224px), same
splits, same memory head, three seeds. Delta is L/14 - B/16, paired:

| dataset | metric | 1-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| MVTec AD | image AUROC | +0.028 (3.1x SE) | +0.032 (4.1x SE) | +0.030 (4.6x SE) |
| MVTec AD | pixel AUROC | +0.014 (2.6x SE) | +0.010 (2.5x SE) | +0.009 (2.3x SE) |
| VisA | image AUROC | +0.012 (1.5x SE) | +0.017 (2.3x SE) | +0.012 (1.7x SE) |
| VisA | pixel AUROC | +0.016 (4.6x SE) | +0.014 (4.1x SE) | +0.013 (3.9x SE) |

| L/14 | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| MVTec image / pixel | 0.883 / 0.930 | 0.920 / 0.942 | **0.933** / 0.945 |
| VisA image / pixel | 0.785 / 0.950 | 0.828 / 0.961 | 0.843 / **0.964** |

**L/14 is the most accurate backbone on both datasets.** On MVTec the gain is
clear on image AUROC and largest exactly where resolution was the limit:
`toothbrush` +0.117, `screw` +0.086 (0.705 -> 0.792), `hazelnut` +0.053. Its
finer 16x16 grid is the same mechanism as B/16 over B/32 (section 6), one step
further. On VisA the localisation gain is solid (4x SE) but the image gain is
not established (under 2.3x SE), and two categories go backwards (`macaroni2`
-0.048, `candle` -0.027).

Memory costs 1024 x 256 x 4 bytes = 1.0 MB per support image, 10.5 MB per class
at 10 shots -- 1.7x B/16. It costs 397 ms per image burst (4.5x B/16) and
3.7 GB of RAM (section 4).

---

## 9. Scoring ablations: windows and text

Records: `mvtec_windowed.json`, `visa_windowed.json`, `text_fusion.json`,
`text_fusion_win123.json` (all under `results/manifests/`). ViT-B/16, three
seeds, paired against the single-pass memory.

**Windows.** Patch tokens are average-pooled over 2x2 and 3x3 windows, each
scale is scored against equally pooled support, and the scales are averaged
back onto the patch grid (`src/model/windows.py`). This is WinCLIP's
multi-scale idea at single-pass cost: WinCLIP re-encodes every window, which is
unaffordable on a CPU; here the extra cost is two more memory searches, about 1%
of inference each.

| delta, windowed - single-pass | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| MVTec image | +0.010 (1.4x SE) | +0.019 (3.3x SE) | +0.017 (4.3x SE) |
| MVTec pixel | +0.000 | +0.003 (0.5x SE) | +0.002 (0.5x SE) |
| VisA image | +0.016 (2.7x SE) | +0.021 (5.2x SE) | +0.020 (5.5x SE) |
| VisA pixel | -0.005 (2.1x SE) | -0.002 (1.6x SE) | -0.002 (1.4x SE) |

Windows help *detection* on both datasets and do nothing for localisation --
slightly worse on VisA at 1 shot, where pooling smears defects that are already
smaller than a patch.

**Image vs text vs fused.** The pooled image embedding is in the joint
image-text space even though the 768-dim patches are not, so text can score a
whole image: the softmax weight on a "defective <object>" prompt ensemble
against a "normal" one. Fused as `(1 - w) * memory + w * text`, image AUROC:

| | w = 0 (image) | 0.1 | 0.25 | 0.5 | 1 (text only) |
|---|---|---|---|---|---|
| MVTec 1-shot | 0.856 | 0.859 | 0.863 | **0.872** | 0.802 |
| MVTec 10-shot | 0.904 | 0.906 | 0.908 | **0.911** | 0.802 |
| VisA 1-shot | 0.773 | 0.776 | 0.781 | **0.791** | 0.698 |
| VisA 10-shot | 0.831 | 0.832 | 0.834 | **0.837** | 0.698 |

Every fused gain is 2.4-4.8x its paired SE, largest at 1 shot, where the memory
knows least. Text alone is far worse. The weights are a fixed grid reported in
full; w = 0.5 is WinCLIP+'s equal weighting, not a value picked here. This
refines the Tip-Adapter finding in section 12: fused *per patch* in the 512-dim space
it hurts (the alpha sweep); fused *per image* it helps.

**Both together** (`configs/proposed.yaml`: windows {1,2,3}, w = 0.5):

| image AUROC | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| MVTec, memory only | 0.856 | 0.888 | 0.904 |
| MVTec, proposed | **0.881** | **0.913** | **0.925** |
| VisA, memory only | 0.773 | 0.811 | 0.831 |
| VisA, proposed | **0.815** | **0.841** | **0.855** |

For reference only: WinCLIP+ reports 93.1 / 94.4 / 95.2 (MVTec) and
83.8 / 84.6 / 87.3 (VisA) image AUROC at 1 / 2 / 4 shots, means over five seeds
(Jeong et al., CVPR 2023, Table 1; pixel AUROC 95.2-96.2 / 96.4-97.2 in Table 4),
with a ViT-B/16+ backbone at 240px, LAION-400M weights and one encoder pass per
window. We did not reproduce it; the backbones, weights and shot counts differ.

---

## 10. Third dataset: NEU-DET

Records: `neu_det_backbones.json`, `neu_det_l14_vs_b16.json`,
`neu_det_windowed.json`, `neu_det_text.json`, `figures_neu_det.json` (all under
`results/manifests/`). Layout and splits by Member B
(`scripts/build_neu_det_oneclass.py`, `results/manifests/splits/neu_det*.json`).

**Protocol.** NEU-DET has six classes of steel-surface defect and no defect-free
images, so each class takes a turn as "normal": support photos come from that
class, and the test set is its 60 held-out images plus 60 drawn evenly from the
other five classes. The question is "given photos of one surface, is this one
different?" -- a cross-domain test of the same frozen method on steel rather than
objects. There are no masks, so image AUROC only.

Image AUROC, ViT-B/16, mean over three seeds, +/- population std:

| class (as normal) | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| crazing | 0.991 +/-0.000 | 0.993 +/-0.002 | 0.994 +/-0.003 |
| rolled-in_scale | 0.974 +/-0.010 | 0.979 +/-0.009 | 0.982 +/-0.000 |
| pitted_surface | 0.878 +/-0.028 | 0.921 +/-0.022 | 0.917 +/-0.015 |
| inclusion | 0.658 +/-0.034 | 0.780 +/-0.049 | 0.831 +/-0.026 |
| patches | 0.704 +/-0.036 | 0.779 +/-0.010 | 0.828 +/-0.035 |
| scratches | 0.724 +/-0.025 | 0.757 +/-0.052 | 0.768 +/-0.041 |
| **mean** | **0.821** | **0.868** | **0.887** |

All configurations, mean image AUROC:

| configuration | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| ViT-B/32 | 0.818 | 0.867 | 0.893 |
| ViT-B/16 | 0.821 | 0.868 | 0.887 |
| ViT-L/14 | 0.838 | 0.883 | 0.897 |
| **ViT-B/16 + windows {1,2,3}** | **0.865** | **0.917** | **0.935** |
| ViT-B/16 + windows + text (w = 0.5) | 0.825 | 0.866 | 0.881 |

Paired over 18 measurements per shot count (6 classes x 3 seeds):

| delta | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| B/16 - B/32 | +0.004 (0.1x SE) | +0.001 (0.1x SE) | -0.007 (0.4x SE) |
| L/14 - B/16 | +0.016 (0.7x SE) | +0.015 (0.9x SE) | +0.011 (1.1x SE) |
| windows - single-pass | **+0.044 (3.3x SE)** | **+0.049 (4.9x SE)** | **+0.049 (5.6x SE)** |
| text fused - windows only | -0.040 (3.4x SE) | -0.051 (3.9x SE) | -0.054 (4.0x SE) |

**The backbone does not matter here, and the defect-size rule says why.**
Measured from the bounding-box annotations, a NEU-DET defect covers 39% of the
200x200 image on average (scratches 18%, pitted_surface 81%) -- even allowing
that boxes overstate area, far above a B/32 patch's 2%. Large defects: B/32 and
B/16 are indistinguishable, as on MVTec (4.6% defects) and unlike VisA
(0.04-0.9%). L/14's gain is not established: under 1.1x SE, and only 2 of 6
classes improve. Three datasets, one consistent rule.

**Windows matter most on textures.** +0.044 to +0.049 at 3.3-5.6x SE, every class
better at 10 shots, `inclusion` +0.071 and `patches` +0.103. A steel texture is
judged better over a neighbourhood than patch by patch -- the same reason windows
helped on MVTec, larger here because every NEU-DET image is a texture.

**Text hurts, and should.** Under this protocol the "normal" class is itself a
defect type, so a prompt like "a photo of a crazing with a defect" asks the wrong
question; fusing it costs 0.040-0.054 at 3.4-4.0x SE, every class worse. Text
fusion is for products with a meaningful normal; it is not part of the NEU-DET
recommendation.

**Verification.** Member B's per-class B/16 and B/32 numbers were reproduced on
Member A's machine to within 0.0005, from her own scripts and the Kaggle copy of
the data, and regenerating the split manifests here gave byte-identical files.
(The B/16 mean she first reported, 0.810 / 0.854 / 0.870, was mis-averaged; the
mean of her own per-class values is the 0.821 / 0.868 / 0.887 above.)

---

## 11. Baseline: PatchCore given the same photos

Records: `patchcore_vs_b_16_*.json`, `patchcore_vs_b_16_windows_*.json`,
`patchcore_vs_l_14_*.json` under `results/manifests/`. Code:
`src/baselines/patchcore_fewshot.py`, `scripts/run_patchcore_fewshot.py`.

PatchCore is the strongest memory-bank method, normally fitted on hundreds of
images. Here it gets **exactly our support photos**: same manifests, 1/5/10 shots,
seeds 42/1/2, the same test images, masks and metric code. anomalib's own
`PatchcoreModel`: WideResNet-50-2 (ImageNet), layers 2+3, 3x3 pooling,
re-weighted scores with 9 neighbours, Gaussian-smoothed maps. Coreset subsampling
is off -- with 10 photos or fewer the bank is already small, so this is PatchCore
at its strongest. Images at 224px with the same crop as ours.

| few-shot PatchCore | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| MVTec image / pixel | 0.842 / 0.938 | 0.899 / 0.958 | 0.926 / 0.965 |
| VisA image / pixel | 0.793 / 0.945 | 0.848 / 0.964 | 0.865 / 0.969 |

Paired against ours, delta is ours - PatchCore (paired SE in brackets):

| ours | metric | 1-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| B/16 | MVTec image | +0.014 (0.7x) | -0.011 (0.8x) | -0.022 (1.7x) |
| B/16 + windows | MVTec image | +0.024 (1.7x) | +0.008 (0.8x) | -0.005 (0.5x) |
| L/14 | MVTec image | **+0.042 (2.0x)** | +0.021 (1.4x) | +0.007 (0.5x) |
| B/16 | VisA image | -0.019 (1.2x) | **-0.037 (3.1x)** | **-0.034 (3.3x)** |
| B/16 + windows | VisA image | -0.003 (0.2x) | -0.017 (1.6x) | -0.015 (1.7x) |
| L/14 | VisA image | -0.007 (0.4x) | -0.020 (1.6x) | -0.022 (1.9x) |
| B/16 + windows | MVTec pixel | **-0.022 (3.0x)** | **-0.024 (4.7x)** | **-0.027 (6.4x)** |
| B/16 + windows | VisA pixel | **-0.016 (2.7x)** | **-0.019 (4.6x)** | **-0.020 (5.1x)** |
| L/14 | VisA pixel | +0.005 (0.9x) | -0.004 (1.0x) | -0.005 (1.7x) |

**Detection: a tie.** With windows (or L/14) our image AUROC is statistically
indistinguishable from PatchCore's on both datasets; L/14 is ahead at one shot on
MVTec (+0.042, 2.0x SE). Plain single-pass B/16 trails on VisA (-0.034, 3.3x SE),
which windows close. With text fusion added (section 9, image level, means only)
ours reaches 0.881 / 0.913 / 0.925 on MVTec and 0.815 / 0.841 / 0.855 on VisA,
against PatchCore's 0.842 / 0.899 / 0.926 and 0.793 / 0.848 / 0.865.

**Localisation: PatchCore wins.** Its WideResNet feature map is 28x28 at 224px,
against CLIP B/16's 14x14, and it shows: +0.02-0.03 pixel AUROC at 3-6x SE. Only
L/14's 16x16 grid draws level on VisA.

**Where each wins, per category (10 shots, windows vs PatchCore).** Ours on fine,
textured defects: `screw` +0.120, `grid` +0.087, VisA `capsules` +0.050,
`macaroni1` +0.049. PatchCore on structured layouts, where a defect is a part in
the wrong place: `transistor` -0.173, `hazelnut` -0.081, VisA `pcb3` -0.115,
`pcb4` -0.071.

**So the case for this method is not raw accuracy.** At equal photos it matches
PatchCore at detection and loses at localisation. What it adds is what PatchCore
does not have: a product added in seconds without touching the others (section
5), typed defect descriptions (section 9), one backbone shared by detection and
text, and a measured CPU cost (sections 4 and 7). Per image the two cost about the
same: PatchCore with 10 photos takes 93 ms against our B/16's 88 ms, and our
B/32 INT8 takes 33 ms (section 4).

Runtime of this baseline on the laptop: 33 minutes for MVTec, 3.3 hours for VisA
(all 27 categories x 3 seeds x 3 shot counts, test features embedded once per
category).

---

## 12. Limitations, stated plainly

**Five categories are still not monotonic in shots.** 10/15 improve
monotonically across three seeds. The five exceptions are small -- `toothbrush`
-0.026 at 5-shot, then `wood` -0.005, `carpet` -0.005, `cable` -0.002, `tile`
-0.000 -- and four of the five are inside their own seed spread.

This was much worse before: a single seed gave only **4/15** monotonic with the
largest violation at 0.057. Averaging over three seeds fixed it, which is what
seeds are for. `toothbrush` is the one that persists, and it is the smallest
category in MVTec (60 train, 42 test, 12 normal test images), so its +/-0.055
spread at 5 shots is unsurprising. More seeds would settle it.

**The Tip-Adapter adds nothing.** Two sweeps (`scripts/sweep_alpha.py`) settled
it. Mixing text into the cache monotonically hurts (alpha 0 → 1 takes 1-shot from
0.820 to 0.646), and widening the kernel never beats nearest-neighbour (top_k 1
and 3 tie, 5+ degrade). At its best setting the adapter reproduces the memory
head *exactly*, because `exp(-beta(1 - s))` at top_k=1 is a strictly monotonic
function of cosine similarity and AUROC is rank-based -- so it **cannot** differ.
Under the 768-dim convention this is now doubly true: the text branch is
unavailable, so the adapter is a pure cache. Only Tip-Adapter-F, with the cache
keys unfrozen and trained, remains worth attempting.

**Zero-shot is unavailable.** 768-dim patch tokens cannot be compared to 512-dim
text embeddings. When it was measured at 512 dims, zero-shot pixel AUROC was
**0.323 -- below chance**, the anomaly map pointing away from the true defect;
raw CLIP patch tokens are known to be poorly aligned for dense prediction, which
is why WinCLIP uses windowed aggregation and APRIL-GAN/AnomalyCLIP train
projection layers. Little was lost.

**`screw` is a resolution failure, not a method failure.** 0.711 at 10 shots
against PatchCore's ~0.98. Screw defects are hair-thin; at 224px on a 14x14 grid
each patch spans ~16px of an already-downsampled image, so the defect is smaller
than the model's spatial resolution. ViT-L/14's finer grid lifts it to 0.792 (section 8),
which supports that reading.

**Not yet done:** MPDD, static INT8,
and a reproduction of WinCLIP / AnomalyCLIP.

---

## Reproducing

```bash
export FEWSHOT_DATA_ROOT=/path/to/datasets
export FEWSHOT_RESULTS_ROOT=/path/to/results

python scripts/download_mvtec_full.py            # Member B
python scripts/generate_splits.py                # Member B
python scripts/run_benchmark.py --threads 14
python scripts/run_benchmark.py --backbone ViT-B-32-quickgelu --threads 14
python scripts/run_benchmark.py --project-patches --out .../proj512.csv   # dim ablation
python scripts/run_profiling.py --threads 14     # idle machine, see section 4
python scripts/run_incremental.py --shots 5
python scripts/make_figures.py

# VisA (section 6)
python scripts/run_benchmark.py --dataset visa --threads 14
python scripts/run_benchmark.py --dataset visa --backbone ViT-B-32-quickgelu --threads 14
python scripts/compare_paired.py visa_ViT-B-16-quickgelu visa_ViT-B-32-quickgelu \
    --question "does ViT-B/32 match ViT-B/16 on VisA, as it does on MVTec AD" \
    --out visa_backbones.json
python scripts/make_figures.py --dataset visa

# INT8 (section 7)
python scripts/export_onnx.py
python scripts/run_benchmark.py --precision int8 --threads 14
python scripts/run_benchmark.py --precision int8 --backbone ViT-B-32-quickgelu --threads 14
python scripts/compare_paired.py mvtec_ad_ViT-B-16-quickgelu mvtec_ad_ViT-B-16-quickgelu-int8 \
    --question "..." --out int8_vit_b16.json               # likewise for B/32
python scripts/run_benchmark.py --dataset visa --precision int8 --threads 6   # and --backbone ViT-B-32-quickgelu
python scripts/run_profiling.py --engine onnx-fp32 --threads 6 --backbones ViT-B-32-quickgelu ViT-B-16-quickgelu
python scripts/run_profiling.py --engine onnx-int8 --threads 6 --backbones ViT-B-32-quickgelu ViT-B-16-quickgelu
```

Everything above, one target per section: `python scripts/reproduce.py --list`.
