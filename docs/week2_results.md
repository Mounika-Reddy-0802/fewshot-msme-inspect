# Results: frozen CLIP + prototype memory on MVTec AD

Reproducible from the committed manifests and the scripts in `scripts/`. Raw CSVs
live outside the repo (gitignored); the manifests identifying each run are
committed.

**Setup.** Frozen OpenCLIP ViT-B/16, no training of any kind. Support images are
normal only. 15 MVTec AD categories, shot counts 1/5/10, evaluated on the fixed
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
prompt cannot score a patch. Section 6 covers what that costs -- little.

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

Idle machine, 14 threads pinned, batch size 1 (a workshop inspects one part at a
time), 5 warmup iterations discarded, median of 20.

| backbone | params | GFLOPs | ms/image | memory search 1-shot | 10-shot |
|---|---|---|---|---|---|
| ViT-B/32 | 151M | 3.0 | 60.4 | 0.17 ms | 0.27 ms |
| ViT-B/16 | 150M | 11.3 | 132.5 | 0.45 ms | 1.47 ms |

**The memory search is about 1% of inference cost.** Going from 1 shot to 10 adds
0.1-1.0 ms to a 60-132 ms budget, so shots are effectively free and there is no
reason for an MSME to skimp on them. The backbone is the only meaningful cost
lever.

### Latency is thermally sensitive

An earlier profiling run, taken immediately after ~60 minutes of continuous
benchmarking, reported **217 ms and 445 ms** for the same two backbones --
**3.6x slower** -- at a nearly identical reported CPU load (10.7% vs 7.5%). The
likely cause is thermal throttling of the laptop CPU.

Both figures are real and answer different questions. The table above is burst
latency on a cool machine; the slower figures are closer to sustained inspection
on laptop-class hardware. **A single latency number for this project would
mislead.** A proper sustained-load measurement belongs in week 5.

---

## 5. Append-only incremental class addition

Committed record: `results/manifests/incremental_5shot.json`.

All 15 categories added to one memory, one at a time. After every addition, every
previously added class was re-scored and compared with `torch.equal`.

| | |
|---|---|
| classes added | 15 |
| cost per class | **~1.9 ms, 3.0 MB** |
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

---

## 6. Limitations, stated plainly

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
than the model's spatial resolution.

**Not yet done:** ViT-L/14 (weights never downloaded), INT8 accuracy (export and
quantisation work -- see `results/manifests/onnx_export.json` -- but the graph was
built against 512-dim features and needs redoing at 768), VisA and NEU-DET
evaluation, and sustained-load latency.

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
```
