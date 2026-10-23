"""One command per figure and table.

    python scripts/reproduce.py --list
    python scripts/reproduce.py backbones          # section 2
    python scripts/reproduce.py int8 --dry-run     # print the commands only
    python scripts/reproduce.py all

Each target names the section of docs/week2_results.md it regenerates and runs
exactly the commands that produced it. Split manifests are committed, so no
target regenerates them; datasets must already be downloaded
(scripts/download_mvtec_full.py, scripts/download_visa.py) and
FEWSHOT_DATA_ROOT / FEWSHOT_RESULTS_ROOT set.

Benchmarks cache features, so a target whose features already exist re-runs in
minutes. Timing targets (profiling, sustained) must run on an idle machine.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B16, B32, L14 = "ViT-B-16-quickgelu", "ViT-B-32-quickgelu", "ViT-L-14-quickgelu"


def bench(*extra: str) -> list[str]:
    return ["scripts/run_benchmark.py", "--threads", "6", *extra]


def paired(a: str, b: str, question: str, out: str) -> list[str]:
    return ["scripts/compare_paired.py", a, b, "--question", question, "--out", out]


TARGETS: dict[str, tuple[str, list[list[str]]]] = {
    "mvtec": ("section 1: few-shot accuracy on MVTec AD", [
        bench(),
        ["scripts/summarise_results.py", "--spread"],
        ["scripts/summarise_results.py", "--spread", "--metric", "pixel_auroc"],
    ]),
    "backbones": ("sections 2 and 6: B/16 vs B/32 on MVTec AD and VisA", [
        bench(), bench("--backbone", B32),
        bench("--dataset", "visa"), bench("--dataset", "visa", "--backbone", B32),
        paired(f"visa_{B16}", f"visa_{B32}",
               "does ViT-B/32 match ViT-B/16 on VisA, as it does on MVTec AD", "visa_backbones.json"),
    ]),
    "incremental": ("section 5: append-only, sequential and held-out", [
        ["scripts/run_incremental.py", "--shots", "5"],
        ["scripts/run_holdout_append.py", "--shots", "5", "--out", "incremental_holdout_5shot.json"],
        ["scripts/run_holdout_append.py", "--shots", "10", "--out", "incremental_holdout_10shot.json"],
    ]),
    "int8": ("section 7: INT8 accuracy on both datasets and the fidelity probe", [
        ["scripts/export_onnx.py"],
        ["scripts/fidelity_probe.py"],
        bench("--precision", "int8"), bench("--precision", "int8", "--backbone", B32),
        bench("--dataset", "visa", "--precision", "int8"),
        bench("--dataset", "visa", "--precision", "int8", "--backbone", B32),
        paired(f"mvtec_ad_{B16}", f"mvtec_ad_{B16}-int8", "does ViT-B/16 accuracy survive dynamic int8 quantisation (768-dim patches)", "int8_vit_b16.json"),
        paired(f"mvtec_ad_{B32}", f"mvtec_ad_{B32}-int8", "does ViT-B/32 accuracy survive dynamic int8 quantisation (768-dim patches)", "int8_vit_b32.json"),
        paired(f"visa_{B16}", f"visa_{B16}-int8", "does ViT-B/16 accuracy survive dynamic int8 quantisation on VisA (768-dim patches)", "int8_vit_b16_visa.json"),
        paired(f"visa_{B32}", f"visa_{B32}-int8", "does ViT-B/32 accuracy survive dynamic int8 quantisation on VisA (768-dim patches)", "int8_vit_b32_visa.json"),
    ]),
    "l14": ("section 8: the third backbone", [
        bench("--backbone", L14), bench("--dataset", "visa", "--backbone", L14),
        paired(f"mvtec_ad_{L14}", f"mvtec_ad_{B16}", "does ViT-L/14 beat ViT-B/16 on MVTec AD", "mvtec_l14_vs_b16.json"),
        paired(f"visa_{L14}", f"visa_{B16}", "does ViT-L/14 beat ViT-B/16 on VisA", "visa_l14_vs_b16.json"),
    ]),
    "ablations": ("windowed scoring and image/text/fused prototypes", [
        bench("--scales", "1", "2", "3"), bench("--dataset", "visa", "--scales", "1", "2", "3"),
        paired(f"mvtec_ad_{B16}-win123", f"mvtec_ad_{B16}", "does multi-scale window scoring (1,2,3) beat single-pass on MVTec AD", "mvtec_windowed.json"),
        paired(f"visa_{B16}-win123", f"visa_{B16}", "does multi-scale window scoring (1,2,3) beat single-pass on VisA", "visa_windowed.json"),
        ["scripts/run_text_fusion.py"],
        ["scripts/run_text_fusion.py", "--scales", "1", "2", "3"],
    ]),
    "profiling": ("sections 4 and 7: latency, idle machine, 6 threads", [
        ["scripts/run_profiling.py", "--threads", "6", "--manifest", "torch_latency.json"],
        ["scripts/run_profiling.py", "--engine", "onnx-fp32", "--threads", "6", "--backbones", B32, B16],
        ["scripts/run_profiling.py", "--engine", "onnx-int8", "--threads", "6", "--backbones", B32, B16],
    ]),
    "sustained": ("latency under sustained load, idle machine", [
        ["scripts/run_sustained.py", "--minutes", "10", "--threads", "6",
         "--backbones", B32, B16, L14, "--manifest", "sustained_torch.json"],
    ]),
    "figures": ("frontier and shots-vs-accuracy figures", [
        ["scripts/make_figures.py"], ["scripts/make_figures.py", "--dataset", "visa"],
    ]),
    "cost": ("the Rs 0 cost and feasibility table", [["scripts/cost_table.py"]]),
    "patchcore": ("section 11: few-shot PatchCore on our splits, paired against ours", [
        ["scripts/run_patchcore_fewshot.py", "--dataset", "mvtec_ad"],
        ["scripts/run_patchcore_fewshot.py", "--dataset", "visa"],
        # CPU cost on an idle machine; let it cool between the fit and the profiles
        ["scripts/profile_patchcore.py", "--stage", "fit", "--category", "bottle"],
        ["scripts/profile_patchcore.py", "--stage", "profile", "--category", "bottle",
         "--out", "results/manifests/patchcore_cost_i7_fullshot.json"],
        ["scripts/profile_patchcore.py", "--stage", "fit-fewshot", "--category", "bottle", "--shots", "10"],
        ["scripts/profile_patchcore.py", "--stage", "profile", "--category", "bottle_10shot",
         "--out", "results/manifests/patchcore_cost_i7_10shot.json"],
        *[paired(f"{ds}_{ours}", f"{ds}_patchcore-wideresnet502",
                 f"how does our {ours} compare with few-shot PatchCore (WideResNet-50, "
                 f"same support photos) on {ds}",
                 f"patchcore_vs_{tag}_{ds}.json")
          for ds in ("mvtec_ad", "visa")
          for ours, tag in ((B16, "b_16"), (f"{B16}-win123", "b_16_windows"), (L14, "l_14"))],
    ]),
    "neu_det": ("NEU-DET, one-class-as-normal, three backbones and windows", [
        ["scripts/reorganize_neu_det.py"],
        ["scripts/build_neu_det_oneclass.py"],
        ["scripts/check_splits.py"],
        bench("--dataset", "neu_det"), bench("--dataset", "neu_det", "--backbone", B32),
        bench("--dataset", "neu_det", "--backbone", L14),
        bench("--dataset", "neu_det", "--scales", "1", "2", "3"),
        paired(f"neu_det_{B16}", f"neu_det_{B32}",
               "does ViT-B/32 match ViT-B/16 on NEU-DET", "neu_det_backbones.json"),
        paired(f"neu_det_{L14}", f"neu_det_{B16}",
               "does ViT-L/14 beat ViT-B/16 on NEU-DET", "neu_det_l14_vs_b16.json"),
        paired(f"neu_det_{B16}-win123", f"neu_det_{B16}",
               "does multi-scale window scoring (1,2,3) beat single-pass on NEU-DET",
               "neu_det_windowed.json"),
        ["scripts/make_figures.py", "--dataset", "neu_det"],
    ]),
}
ORDER = ["mvtec", "backbones", "incremental", "int8", "l14", "ablations",
         "neu_det", "patchcore", "profiling", "sustained", "figures", "cost"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list or not args.targets:
        for name in ORDER:
            print(f"  {name:<12} {TARGETS[name][0]}")
        return 0

    wanted = ORDER if args.targets == ["all"] else args.targets
    unknown = [t for t in wanted if t not in TARGETS]
    if unknown:
        print(f"unknown target(s) {unknown}; see --list", file=sys.stderr)
        return 2

    for name in wanted:
        print(f"\n### {name}: {TARGETS[name][0]}", flush=True)
        for command in TARGETS[name][1]:
            full = [sys.executable, *command]
            print("$ python " + " ".join(command), flush=True)
            if not args.dry_run:
                result = subprocess.run(full, cwd=ROOT)
                if result.returncode != 0:
                    print(f"failed: {' '.join(command)}", file=sys.stderr)
                    return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
