# Results: frozen CLIP + prototype memory on MVTec AD

Reproducible from the committed manifests and the scripts in `scripts/`. Raw CSVs
live outside the repo (gitignored); the manifests identifying each run are
committed.

**Setup.** Frozen OpenCLIP ViT-B/16, no training of any kind. Support images are
normal only. 15 MVTec AD categories, shot counts 1/5/10, evaluated on the fixed
test split so numbers are comparable across shot counts. CPU only: Intel
i7-13700H, 14 threads pinned, no GPU anywhere.

**Splits** come from Member B's `results/manifests/splits/mvtec_ad.json`
(seed 42) -- one definition of a split for the whole project. Her shot sets are
nested (1 ⊂ 5 ⊂ 10), so adding shots is measured against the same starting
images.

**Features** are the raw 768-dim transformer tokens, matching her feature-bank
convention (section 3 measures what that choice is worth).

Two consequences of those conventions, both stated up front:

* **There is no 0-shot column.** 768-dim patch tokens are not in the joint
  image-text space, so a text prompt cannot score a patch. Section 6 covers what
  that costs -- little.
* **There is one seed.** Every number below is a single measurement with no error
  bars. Section 6 quantifies the damage.

---

## 1. Few-shot accuracy

Image AUROC, ViT-B/16, prototype memory.

| category | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| bottle | 0.979 | 0.993 | 0.998 |
| tile | 0.980 | 0.992 | 0.990 |
| leather | 0.988 | 0.986 | 0.989 |
| carpet | 0.985 | 0.999 | 0.984 |
| metal_nut | 0.933 | 0.933 | 0.975 |
| wood | 0.965 | 0.956 | 0.963 |
| hazelnut | 0.823 | 0.900 | 0.932 |
| grid | 0.906 | 0.927 | 0.926 |
| zipper | 0.916 | 0.917 | 0.910 |
| pill | 0.890 | 0.896 | 0.905 |
| toothbrush | 0.733 | 0.703 | 0.894 |
| capsule | 0.860 | 0.803 | 0.886 |
| cable | 0.772 | 0.885 | 0.848 |
| transistor | 0.697 | 0.729 | 0.717 |
| screw | 0.635 | 0.678 | 0.711 |
| **mean** | **0.871** | **0.886** | **0.909** |

Pixel AUROC on the same runs -- the harder question, since it asks *where* the
defect is, not merely whether there is one.

| category | 1-shot | 5-shot | 10-shot |
|---|---|---|---|
| carpet | 0.990 | 0.989 | 0.980 |
| leather | 0.988 | 0.987 | 0.987 |
| toothbrush | 0.978 | 0.983 | 0.986 |
| bottle | 0.973 | 0.975 | 0.977 |
| hazelnut | 0.965 | 0.982 | 0.984 |
| pill | 0.935 | 0.942 | 0.947 |
| tile | 0.921 | 0.932 | 0.934 |
| wood | 0.916 | 0.924 | 0.927 |
| capsule | 0.917 | 0.930 | 0.934 |
| screw | 0.913 | 0.917 | 0.921 |
| grid | 0.898 | 0.913 | 0.917 |
| cable | 0.886 | 0.889 | 0.895 |
| zipper | 0.834 | 0.861 | 0.880 |
| transistor | 0.834 | 0.870 | 0.861 |
| metal_nut | 0.855 | 0.900 | 0.917 |
| **mean** | **0.920** | **0.933** | **0.936** |

Pixel AUROC of **0.920 from a single photo** is the strongest result here. For
scale, PatchCore -- trained on all 209 normal images of `bottle` -- reaches 0.986
on that category; one photo gets 0.973 on the same category.

---

## 2. Backbone: an accuracy-cost trade, not a free lunch

| backbone | patches | image 1/5/10 | pixel 1/5/10 | GFLOPs | ms/image |
|---|---|---|---|---|---|
| ViT-B/16 | 14x14 = 196 | **0.871** / 0.886 / **0.909** | 0.920 / 0.933 / 0.936 | 11.3 | 132 |
| ViT-B/32 | 7x7 = 49 | 0.839 / **0.889** / 0.899 | 0.919 / **0.934** / **0.939** | **3.0** | **60** |

ViT-B/16 leads on image AUROC at 1 and 10 shots; the two are level at 5. On
**pixel** AUROC they are indistinguishable -- B/32 is fractionally ahead at 5 and
10 shots despite storing a quarter as many prototypes.

So for localisation, the cheaper backbone is the better deal: same accuracy,
3.7x fewer FLOPs, 2.2x faster, 4x less memory. For image-level detection, B/16
buys about +0.02 AUROC for 2.2x the latency, and whether that is worth paying is
an MSME's decision rather than ours.

### Correction to an earlier claim

An earlier version of this document reported that **ViT-B/32 beat ViT-B/16 at
every shot count** while costing 3.7x less, and called it the headline finding.
That was measured with 512-dim projected features. It does not survive the move
to 768-dim: with the richer features, B/16's finer patch grid pays off and the
ordering reverses at 1 and 10 shots. The earlier claim was an artefact of the
projection, not a property of the backbones.

---

## 3. What the feature dimension is worth

Committed record: `results/manifests/dim_ablation.json`.

CLIP's visual tower emits 768-dim tokens and then projects them to 512 dims to
align with text. For image-to-image matching that projection is not obviously
useful, so it was measured directly -- **identical splits, identical shot counts,
identical backbone, only the projection differs**:

| shots | 512 projected | 768 unprojected | delta | categories favouring 768 |
|---|---|---|---|---|
| 1 | 0.857 | **0.871** | +0.014 | 10/15 |
| 5 | 0.869 | **0.886** | +0.017 | 11/15 |
| 10 | 0.892 | **0.909** | +0.017 | 12/15 |

A consistent but modest **+0.015**, in the same direction at every shot count.
`cable` gains most (+0.056 at 1-shot); `pill` and `capsule` genuinely lose.

**A methodological note worth more than the result.** The first comparison run
was against the earlier 3-seed results and showed **+0.048** at 1-shot. That
number changed two things at once -- the feature dimension *and* the splits -- and
roughly two thirds of the apparent gain was the split change. Only the controlled
comparison above supports a claim about the projection.

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

**One seed, so most categories are no longer monotonic in shots.** Under three
seeds, 13/15 categories improved monotonically with 2 violations, both < 0.01.
Under Member B's single seed 42, only **4/15** are strictly monotonic: 11
violations, median 0.009, largest 0.057 (`capsule` at 5-shot). The underlying
method did not get worse -- the averaging that hid single-draw noise is gone.
Multiple seeds are the fix, and are worth agreeing on before any result is
frozen.

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
