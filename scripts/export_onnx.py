"""Export the frozen encoder to ONNX and quantise it to INT8.

    python scripts/export_onnx.py
    python scripts/export_onnx.py --backbones ViT-B-32-quickgelu

Writes both precisions, reports file sizes, and -- importantly -- checks that the
INT8 graph still produces features close to the FP32 ones before anything is
benchmarked on it. A quantised model that silently returns rubbish would show up
as a mysterious accuracy drop three steps later.

The agreement check reports cosine similarity between FP32 and INT8 features on
random inputs. It is a smoke test, not evidence about accuracy: what matters for
the benchmark is whether AUROC survives, which `run_benchmark.py --precision
int8` measures directly.

Patch dim follows the config (768 unprojected by default), and is part of each
graph's file name, so exports at both dims can sit side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.model.encoder import FrozenCLIPEncoder  # noqa: E402
from src.model.export import OnnxEncoder, export_onnx, onnx_path, quantize_int8  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import repo_root, results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

DEFAULT_BACKBONES = ("ViT-B-32-quickgelu", "ViT-B-16-quickgelu")


def agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between two sets of features."""
    return float(
        torch.nn.functional.cosine_similarity(a.flatten(0, -2), b.flatten(0, -2)).mean()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="prototype.yaml")
    parser.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["experiment"]["seed"])
    pretrained = config["backbone"]["pretrained"]
    project_patches = config["backbone"].get("project_patches", True)
    out_dir = Path(args.out) if args.out else results_root() / "onnx"

    records = []
    for name in args.backbones:
        print(f"\n=== {name} ===", flush=True)
        try:
            encoder = FrozenCLIPEncoder(
                name, pretrained, device="cpu", project_patches=project_patches
            )
        except Exception as exc:
            print(f"  skipped: {type(exc).__name__} (weights unavailable?)")
            continue

        fp32_path = onnx_path(out_dir, name, encoder.embed_dim)
        print(f"  exporting fp32, {encoder.embed_dim}-dim patches ...", flush=True)
        export_onnx(encoder, fp32_path)

        print("  quantising to int8 ...", flush=True)
        int8_path = quantize_int8(fp32_path, onnx_path(out_dir, name, encoder.embed_dim, "int8"))

        fp32_mb = fp32_path.stat().st_size / 1e6
        int8_mb = int8_path.stat().st_size / 1e6

        # do the features survive quantisation?
        example = torch.randn(2, 3, *encoder.image_size)
        torch_out = encoder.encode_images(example)
        fp32_out = OnnxEncoder(fp32_path, args.threads).encode_images(example)
        int8_out = OnnxEncoder(int8_path, args.threads).encode_images(example)

        record = {
            "backbone": name,
            "patch_dim": encoder.embed_dim,
            "fp32_mb": round(fp32_mb, 1),
            "int8_mb": round(int8_mb, 1),
            "compression": round(fp32_mb / int8_mb, 2),
            "torch_vs_fp32_image": round(agreement(torch_out.image, fp32_out.image), 5),
            "torch_vs_fp32_patches": round(agreement(torch_out.patches, fp32_out.patches), 5),
            "fp32_vs_int8_image": round(agreement(fp32_out.image, int8_out.image), 5),
            "fp32_vs_int8_patches": round(agreement(fp32_out.patches, int8_out.patches), 5),
        }
        records.append(record)

        print(f"  fp32 {fp32_mb:6.1f} MB   int8 {int8_mb:6.1f} MB   "
              f"{record['compression']:.2f}x smaller")
        print(f"  torch vs onnx-fp32  cosine {record['torch_vs_fp32_image']:.5f} image, "
              f"{record['torch_vs_fp32_patches']:.5f} patches")
        print(f"  fp32  vs int8       cosine {record['fp32_vs_int8_image']:.5f} image, "
              f"{record['fp32_vs_int8_patches']:.5f} patches")

        if record["torch_vs_fp32_image"] < 0.999:
            print("  WARNING: export changed the features; the graph is not faithful")
        if record["fp32_vs_int8_image"] < 0.95:
            print("  WARNING: int8 features diverge substantially from fp32")

    if not records:
        print("nothing exported", file=sys.stderr)
        return 1

    manifest = repo_root() / "results" / "manifests" / "onnx_export.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # keep exports at other dims; records predating the dim field were 512
    fresh = {(r["backbone"], r["patch_dim"]) for r in records}
    kept = []
    if manifest.exists():
        for r in json.loads(manifest.read_text(encoding="utf-8"))["exports"]:
            r.setdefault("patch_dim", 512)
            if (r["backbone"], r["patch_dim"]) not in fresh:
                kept.append(r)
    manifest.write_text(json.dumps({"exports": kept + records}, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(records)} exports -> {out_dir}")
    print(f"manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
