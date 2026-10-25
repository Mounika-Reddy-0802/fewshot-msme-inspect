# Review guide

How to present this project in about 15 minutes, what to say, and how to answer
the questions faculty are likely to ask. Every number below comes from
`results/manifests/` and appears in the app's *Results* page and in the paper.

## Before the review

```bash
set FEWSHOT_DATA_ROOT=C:\Users\HP\datasets
set FEWSHOT_RESULTS_ROOT=C:\Users\HP\fewshot-results
C:\Users\HP\venvs\fewshot-msme-inspect\Scripts\streamlit run app/demo.py
```

* Laptop plugged in, other heavy programs closed, OneDrive sync paused.
* Open the app once beforehand so the model is loaded (first load takes ~20 s).
* In the sidebar choose **ViT-B/16 multi-layer** and leave **Multi-scale windows** on.
* Click *Forget all products* so the demo starts clean.

## The 15-minute walk-through

| minutes | page | what to say |
|---|---|---|
| 0-2 | Overview | The problem: a workshop has good parts, a laptop and changing products. The idea in the diagram: a frozen model, a memory of good patches, compare every patch. Nothing is trained. |
| 2-4 | How it works | Patches (show the grid on the bottle), the one-line score, why defect size matters, windows and text, the threshold. |
| 4-9 | Live demo | *Add a product*: bottle, 5 photos, description "a broken or cracked rim". *Inspect a part*: random defective part, point at the heatmap, the green real-defect outline, and the "How this verdict was reached" steps. Then a random good part. *Check a batch*: the whole bottle test set, live AUROC and the confusion table. Add `screw`, re-inspect the bottle: "bit-identical". |
| 9-13 | Results | Shots curve, the backbone/defect-size rule, what each idea adds (multi-layer is the big jump), PatchCore comparison, cost and the throttling chart. |
| 13-15 | About | Reproducibility checks, limitations, what we would do next. |

## Numbers to remember

* **MVTec AD, 10 photos, ViT-B/16:** last block 0.904 image / 0.936 pixel AUROC;
  with blocks 6, 9, 12: 0.938 / 0.957; plus windows: **0.945 / 0.962**. All 15
  products improve at 5 and 10 photos.
* **VisA, 10 photos:** 0.831 -> 0.872 (multi-layer) -> **0.881** (+ windows).
* **NEU-DET and MPDD: no gain from multi-layer.** NEU-DET is best with windows alone
  (0.935); MPDD 0.808 -> 0.824 is a tie (1.1 x SE). Say this before you are asked.
* **Against PatchCore given the same photos** (multi-layer + windows): detection
  ahead on MVTec (+0.050 / +0.035 / +0.018 at 1/5/10 photos), VisA (+0.034 / +0.020 /
  +0.016) and MPDD (+0.121 / +0.076 / +0.067), all above 2 x SE. Localisation level
  on MVTec; PatchCore still ~0.01 ahead on VisA and MPDD.
* **WinCLIP's protocol** (MVTec, 1/2/4 photos, 5 seeds): ours 89.4 / 91.2 / 92.9,
  WinCLIP+ (published) 93.1 / 94.4 / 95.2, PatchCore (our run) 84.0 / 87.1 / 89.1.
  Our PatchCore matches WinCLIP's own PatchCore row within 0.8 points.
* **Which block matters:** block 6 alone gives 0.936 image / 0.963 pixel - nearly all
  of the gain and the best localisation.
* **Defect-size rule:** B/32 patch = 2% of the image, B/16 = 0.5%. MVTec defects
  4.6%, NEU-DET 39% -> cheap B/32 is enough. VisA 0.04-0.9% -> B/16 needed (+0.05-0.09).
* **Cost:** 37 ms per part (B/32), 88 ms (B/16), 94-97 ms (B/16 multi-layer), 33 ms
  (B/32 INT8). Under continuous use the laptop slows ~6x after 4 minutes. 10,000
  parts/day = Rs 10-26 of electricity a year.
* **Adding a product:** 2-4 s from 5-10 photos; every other product's score
  unchanged bit for bit (checked on every seed).

## Likely questions

**Is anything trained?** No. The CLIP model is frozen (a test asserts zero
trainable parameters). "Adding a product" stores patch vectors in a list.

**What is new here? CLIP and PatchCore already exist.** The method is built from
known parts on purpose. The contributions are: (1) a CPU-only benchmark with
paired statistics across three backbones, two precisions, four datasets; (2) the
defect-size rule for choosing a backbone, confirmed on three datasets; (3) that
training-free multi-layer CLIP features put detection ahead of PatchCore and close
its localisation lead on MVTec (not on VisA's tiny defects);
(4) the throttling finding (burst latency overstates laptop throughput ~6x); (5)
append-only product addition verified bit-identical; (6) a cost analysis in rupees.

**Why blocks 6, 9 and 12?** The last block has mixed each patch with the whole
image, so it blurs where a defect is; earlier blocks keep local detail. Blocks 6, 9,
12 were picked before measuring (the second half of the network, spaced evenly in
the spirit of APRIL-GAN's multi-stage features). The slice test
afterwards showed block 6 alone is almost as good - we report that rather than
re-tune on the test set.

**Why does it not help on MPDD and NEU-DET?** Not established; we did not run an
experiment to find out. A plausible reading: NEU-DET defects are large (39% of the
image), so windows over the last block already see them; MPDD parts vary in pose
and lighting, and earlier blocks may react to that variation as much as to defects.
We state the result as dataset-dependent and do not claim more.

**Why not beat WinCLIP+?** WinCLIP+ runs the encoder once per window (many passes
per image) with a larger LAION-trained model at 240 px. We run it once, at about
95 ms on a laptop. We are 2-4 points below it; PatchCore is 6-9 below it.

**Why AUROC and not accuracy?** AUROC does not depend on a threshold, so it compares
methods fairly. The batch page also shows accuracy at the chosen threshold.

**Why only three seeds?** Three draws already show the spread (0.027 at one photo);
the WinCLIP comparison uses five seeds to match that paper's protocol.

**Is the comparison with PatchCore fair?** It gets exactly our photos, test images,
masks and metric code, with coreset subsampling off (its strongest setting here).

**Why is the suggested threshold sometimes wrong?** It is the highest score among
the good photos scored against each other. With 3-5 photos that estimate is noisy:
it can reject good parts or miss defects. The app shows where each part sits and
lets the operator move the line. A calibrated threshold is listed as future work.

**Would it work in a real factory?** Not shown yet. MPDD (metal parts under varied
lighting) is the closest public test; real workshop photos are the next step.

**How do you know the numbers are right?** Re-extraction from raw images reproduces
every result row; a second laptop reproduced MVTec and VisA to three decimals from a
fresh install; NEU-DET was reproduced across machines; Colab reproduces a product
exactly; 200+ automated tests.

**Was AI used?** Yes, an AI assistant helped with code, experiment organisation and
drafting text; this is stated in the paper. All experiments ran on our machines and
every number was checked against the result files.
