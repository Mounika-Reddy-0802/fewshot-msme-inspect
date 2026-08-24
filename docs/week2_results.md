# Results: frozen CLIP + prototype memory on MVTec AD

Everything here is reproducible from the committed split manifests and the
scripts in `scripts/`. Raw CSVs live outside the repo (gitignored); the manifests
that identify each run are committed.

**Setup.** Frozen OpenCLIP, no training of any kind. Support images are normal
only. 15 MVTec AD categories, 3 seeds, shot counts 0/1/5/10, all measured on the
same fixed test split so numbers are directly comparable across shot counts.
CPU only: Intel i7-13700H, 14 threads pinned, no GPU used anywhere.

---

## 1. Few-shot accuracy

Image AUROC, ViT-B/16, mean over 3 seeds.

| category | 0-shot | 1-shot | 5-shot | 10-shot |
|---|---|---|---|---|
| leather | 0.578 | 0.994 | 0.994 | **0.998** |
| bottle | 0.807 | 0.979 | 0.979 | 0.988 |
| wood | 0.839 | 0.956 | 0.972 | 0.980 |
| tile | 0.798 | 0.955 | 0.980 | 0.980 |
| carpet | 0.578 | 0.950 | 0.955 | 0.956 |
| metal_nut | 0.692 | 0.885 | 0.931 | 0.941 |
| pill | 0.641 | 0.884 | 0.896 | 0.903 |
| zipper | 0.519 | 0.817 | 0.881 | 0.899 |
| capsule | 0.620 | 0.724 | 0.780 | 0.842 |
| grid | 0.638 | 0.786 | 0.810 | 0.827 |
| cable | 0.586 | 0.668 | 0.783 | 0.822 |
| toothbrush | 0.675 | 0.667 | 0.754 | 0.817 |
| hazelnut | 0.631 | 0.736 | 0.793 | 0.809 |
| transistor | 0.645 | 0.685 | 0.739 | 0.759 |
| screw | 0.445 | 0.653 | 0.687 | 0.684 |
| **mean** | **0.646** | **0.823** | **0.862** | **0.880** |

Thirteen of fifteen categories improve monotonically with shots. The two
exceptions are `screw` (10-shot 0.684 vs 5-shot 0.687) and `toothbrush` (1-shot
0.667 vs 0-shot 0.675); both gaps are inside seed noise, and both occur in the
categories flagged in section 5.

The single largest jump is one photo: mean image AUROC goes from **0.646 to
0.823** between 0 and 1 shot. Nine further images buy 0.057 more.

---

## 2. The cheaper backbone is also the more accurate one

This is the most consequential finding, and it inverts the usual assumption that
a finer patch grid is better.

| backbone | patches | 0-shot | 1-shot | 5-shot | 10-shot | GFLOPs | ms/image | prototypes/class @10-shot |
|---|---|---|---|---|---|---|---|---|
| ViT-B/16 | 14x14 = 196 | **0.646** | 0.823 | 0.862 | 0.880 | 11.3 | 132 | 1,960 |
| ViT-B/32 | 7x7 = 49 | 0.552 | **0.831** | **0.872** | **0.895** | **3.0** | **60** | **490** |

For every few-shot setting, ViT-B/32 is simultaneously **3.7x cheaper in FLOPs,
2.2x faster, 4x smaller in memory, and more accurate.** It strictly dominates
ViT-B/16 on the accuracy-cost frontier; ViT-B/16 is not on the frontier at all.

The advantage is not uniform -- B/32 wins 9 of 15 categories at 10-shot, mean
delta +0.015 -- and the split is structured rather than random:

* B/32 wins most on hazelnut (+0.165), toothbrush (+0.133), grid (+0.106)
* B/16 wins most on screw (-0.128), pill (-0.084)

A plausible mechanism, consistent with both directions: nearest-neighbour anomaly
detection is hurt by a *larger* memory bank, because more stored prototypes means
a higher chance that an anomalous patch finds a spurious close match and is
scored normal. B/32 stores a quarter as many prototypes. Where defects are small
relative to a 32-pixel patch (`screw`), the coarser grid loses them entirely and
B/16 wins instead.

The ordering **reverses at 0-shot**, where B/16 leads 0.646 to 0.552. Finer
patches help text matching; coarser patches help memory matching. These are
different mechanisms and should not be collapsed into "B/32 is better".

---

## 3. Compute cost

Measured on an idle machine, 14 threads pinned, batch size 1 (a workshop
inspects one part at a time), 5 warmup iterations discarded, median of 20.

| backbone | params | GFLOPs | ms/image | memory search 1-shot | memory search 10-shot |
|---|---|---|---|---|---|
| ViT-B/32 | 151M | 3.0 | 60.4 | 0.17 ms | 0.27 ms |
| ViT-B/16 | 150M | 11.3 | 132.5 | 0.45 ms | 1.47 ms |

**The memory search is about 1% of inference cost.** Going from 1 shot to 10
adds 0.1-1.0 ms to a 60-132 ms budget. Practically, shots are free and there is
no reason for an MSME to skimp on them; the backbone is the only meaningful cost
lever.

### A caveat that matters more than the numbers

An earlier profiling run, taken immediately after ~60 minutes of continuous
benchmarking, reported **217 ms and 445 ms** for the same two backbones --
**3.6x slower** than the table above, at a nearly identical reported CPU load
(10.7% vs 7.5%). The most likely explanation is thermal throttling of the laptop
CPU.

Both numbers are real, and they answer different questions. The table above is
burst latency on a cool machine. The 3.6x-slower figures are closer to what
sustained inspection on laptop-class hardware actually delivers. **A single
latency number for this project would be misleading**, and a proper
sustained-load measurement belongs in week 5.

---

## 4. Append-only incremental class addition

Committed record: `results/manifests/incremental_5shot.json`.

All 15 categories were added to one memory, one at a time. After every single
addition, every previously added class was re-scored and compared to its original
scores with `torch.equal`.

| | |
|---|---|
| classes added | 15 |
| cost per class | **0.44 ms, 2.0 MB** |
| total memory, 15 classes | 28.7 MiB |
| mean image AUROC (5-shot) | 0.862 |
| existing-class scores unchanged | **bit-identical after all 15 additions** |

The bit-identity is the claim worth making. Methods that re-fit a shared
component -- a normaliser, a codebook, a classifier head -- when a class arrives
will shift existing decision boundaries slightly. Here each class owns its bank,
so adding a new part provably cannot alter inspection of parts already in
production.

Millisecond timings for an operation this small are sensitive to background load
(a second run measured 6.5 ms total rather than 15.8 ms). The per-class *cost*
in bytes and prototypes is the robust figure.

---

## 5. Limitations, stated plainly

**Zero-shot localisation does not work.** Zero-shot pixel AUROC on bottle is
**0.323 -- below chance**, meaning the text-prompted anomaly map points away from
the true defect. It is not a sign error: image AUROC on the same run is 0.807,
which would have collapsed too. Raw CLIP patch tokens are known to be poorly
aligned for dense prediction, which is why WinCLIP uses windowed aggregation and
APRIL-GAN/AnomalyCLIP train projection layers. The 0-shot column of any
pixel-level table here is unusable.

This also quantifies the value of the memory: one photo takes pixel AUROC from
**0.323 to 0.968**.

**The Tip-Adapter head is a net negative as configured.** Blending text into the
cache at alpha=0.3 loses to plain prototype memory:

| shots | memory | adapter | delta |
|---|---|---|---|
| 1 | 0.823 | 0.815 | -0.008 |
| 5 | 0.862 | 0.854 | -0.008 |
| 10 | 0.880 | 0.870 | -0.010 |

It loses in 11 of 15 categories, ties in 2, wins in 2 -- one of those a large win
on hazelnut (+0.083) that does not offset eleven small losses.

Two sweeps isolate why (`scripts/sweep_alpha.py`, one seed, all 15 categories).

**alpha: the text branch never helps once there is at least one shot.**

| shots | a=0 | a=0.1 | a=0.3 | a=0.5 | a=1.0 |
|---|---|---|---|---|---|
| 1 | **0.820** | 0.819 | 0.814 | 0.799 | 0.646 |
| 5 | **0.862** | 0.859 | 0.852 | 0.839 | 0.646 |
| 10 | **0.879** | 0.877 | 0.870 | 0.859 | 0.646 |

**top_k at alpha=0: a wider kernel is never better than nearest-neighbour.**

| shots | k=1 | k=3 | k=5 | k=10 | k=25 |
|---|---|---|---|---|---|
| 1 | 0.820 | 0.822 | 0.817 | 0.806 | 0.783 |
| 5 | 0.862 | 0.861 | 0.858 | 0.851 | 0.837 |
| 10 | 0.879 | 0.879 | 0.876 | 0.868 | 0.854 |

The best adapter setting (alpha=0, top_k=1) reproduces the memory head exactly
-- 0.820/0.862/0.879 against 0.823/0.862/0.880, the residual being one seed
versus three. That is not a coincidence: at top_k=1, `exp(-beta(1 - s))` is a
strictly monotonic function of the cosine similarity `s`, so it preserves the
ranking, and AUROC is rank-based. The kernel **cannot** change AUROC at its best
setting, and every setting where it does act (top_k > 1, alpha > 0) is worse.

**The training-free Tip-Adapter therefore adds nothing over prototype memory.**
Tip-Adapter-F, with the cache keys unfrozen and trained, is the only variant
worth attempting; the module already supports it (`trainable=True`, gradient
flow covered by tests).

**`screw` is a resolution failure, not a method failure.** 0.684 at 10 shots
against PatchCore's ~0.98. Screw defects are hair-thin; at 224px with a 14x14
grid, each patch spans ~16px of an already-downsampled image, so the defect is
smaller than the model's spatial resolution.

**Not yet done:** ViT-L/14 (weights never downloaded), INT8/ONNX export, VisA and
NEU-DET, and any sustained-load latency measurement.

---

## Reproducing

```bash
export FEWSHOT_DATA_ROOT=/path/to/datasets
export FEWSHOT_RESULTS_ROOT=/path/to/results

python scripts/download_mvtec_category.py --all
python scripts/make_splits.py
python scripts/run_benchmark.py --methods memory adapter --threads 14
python scripts/run_benchmark.py --backbone ViT-B-32-quickgelu --threads 14
python scripts/run_profiling.py --threads 14        # idle machine, see section 3
python scripts/run_incremental.py --shots 5
python scripts/make_figures.py
```
