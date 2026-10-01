"""The 50-image fidelity probe: do the ONNX graphs score real parts like torch?

    python scripts/fidelity_probe.py
    python scripts/fidelity_probe.py --images 50 --backbones ViT-B-16-quickgelu

export_onnx.py checks agreement on random tensors, which says the graph was
traced correctly but nothing about real images. This takes a fixed set of real
MVTec test images (spread evenly over categories, from the seed-42 manifest),
scores each against its category's 5-shot memory under torch, ONNX FP32 and
ONNX INT8, and reports:

* patch-feature cosine against torch;
* anomaly-score agreement: max absolute difference and rank correlation.

Gate: ONNX FP32 must match torch (patch cosine >= 0.999) before any efficiency
number from it is reported. INT8 has no gate -- its cost is measured, not
assumed -- but its agreement is recorded next to its accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.splits import categories, load_manifest, manifest_path, resolve_split, split_for  # noqa: E402
from src.model.encoder import FrozenCLIPEncoder  # noqa: E402
from src.model.export import OnnxEncoder, onnx_path  # noqa: E402
from src.model.memory import PrototypeMemory  # noqa: E402
from src.utils.paths import dataset_dir, repo_root, results_root  # noqa: E402

GATE = 0.999


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", nargs="+",
                        default=["ViT-B-32-quickgelu", "ViT-B-16-quickgelu"])
    parser.add_argument("--images", type=int, default=50)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset = "mvtec_ad"
    root = dataset_dir(dataset).parent
    manifest = load_manifest(manifest_path(repo_root() / "results" / "manifests" / "splits", dataset))
    cats = categories(manifest)

    # fixed, evenly spread probe set: round-robin over categories
    per_cat = {}
    for c in cats:
        support, test = resolve_split(split_for(manifest, c, args.shots, root), root)
        per_cat[c] = (support, test)
    probe, i = [], 0
    while len(probe) < args.images:
        c = cats[i % len(cats)]
        test = per_cat[c][1]
        j = i // len(cats)
        if j < len(test):
            probe.append((c, test[(j * 7) % len(test)].image))   # stride so both labels appear
        i += 1

    record = {"images": len(probe), "categories": len(cats), "shots": args.shots,
              "gate_patch_cosine": GATE, "backbones": {}}
    passed = True
    for name in args.backbones:
        torch_enc = FrozenCLIPEncoder(name, "openai", device="cpu", project_patches=False)
        dim = torch_enc.embed_dim
        engines = {"torch": torch_enc}
        for precision in ("fp32", "int8"):
            path = onnx_path(results_root() / "onnx", name, dim, precision)
            if path.exists():
                engines[f"onnx-{precision}"] = OnnxEncoder(path, args.threads)

        def encode(encoder, paths):
            batch = torch.stack([encoder.preprocess(Image.open(p)) for p in paths])
            return encoder.encode_images(batch).patches

        outputs = {}
        for engine, enc in engines.items():
            memories = {}
            for c in {c for c, _ in probe}:
                m = PrototypeMemory(dim, torch_enc.grid_size)
                m.add_class(c, encode(enc, per_cat[c][0]))
                memories[c] = m
            patches, scores = [], []
            for c, p in probe:
                q = encode(enc, [p])
                patches.append(q[0])
                scores.append(float(memories[c].score(q, c).image[0]))
            outputs[engine] = (torch.stack(patches), torch.tensor(scores))

        ref_patches, ref_scores = outputs["torch"]
        rows = {}
        for engine, (patches, scores) in outputs.items():
            if engine == "torch":
                continue
            cos = torch.nn.functional.cosine_similarity(
                patches.flatten(0, 1), ref_patches.flatten(0, 1)).mean()
            rows[engine] = {
                "patch_cosine_vs_torch": round(float(cos), 5),
                "score_max_abs_diff": round(float((scores - ref_scores).abs().max()), 5),
                "score_rank_correlation": round(spearman(scores, ref_scores), 4),
            }
            print(f"{name:<22}{engine:<10} cosine {rows[engine]['patch_cosine_vs_torch']:.5f}  "
                  f"max |dscore| {rows[engine]['score_max_abs_diff']:.4f}  "
                  f"rank corr {rows[engine]['score_rank_correlation']:.4f}")
        fp32 = rows.get("onnx-fp32")
        gate_ok = bool(fp32) and fp32["patch_cosine_vs_torch"] >= GATE
        passed &= gate_ok
        record["backbones"][name] = {"engines": rows, "fp32_gate_passed": gate_ok}

    out = repo_root() / "results" / "manifests" / "fidelity_probe.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nfp32 gate {'passed' if passed else 'FAILED'}   manifest -> {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
