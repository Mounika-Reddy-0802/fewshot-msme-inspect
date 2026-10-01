"""Peak RAM of each deployable configuration, one fresh process each.

    python scripts/measure_ram.py

profile_encoder()'s peak_ram_mb measures growth *during* a forward pass after
warm-up, which is ~0: the buffers are already allocated. What a workshop needs
is how much memory the inspector occupies. So each (backbone, engine) is loaded
in a fresh Python process, and the report is

    peak working set (model loaded + 5 forward passes)  -  baseline after imports

using the OS's own peak counter (psutil ``peak_wset`` on Windows, ``ru_maxrss``
elsewhere), so short spikes between samples are not missed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import repo_root  # noqa: E402

CONFIGS = [
    ("ViT-B-32-quickgelu", "torch"), ("ViT-B-32-quickgelu", "onnx-fp32"),
    ("ViT-B-32-quickgelu", "onnx-int8"),
    ("ViT-B-16-quickgelu", "torch"), ("ViT-B-16-quickgelu", "onnx-fp32"),
    ("ViT-B-16-quickgelu", "onnx-int8"),
    ("ViT-L-14-quickgelu", "torch"),
]

CHILD = r"""
import sys, json, os
sys.path.insert(0, {root!r})
import psutil, torch
torch.set_num_threads(6)
from src.model.encoder import FrozenCLIPEncoder
from src.model.export import OnnxEncoder, onnx_path
from src.utils.paths import results_root

def peak_mb():
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / 1e6
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3

base = psutil.Process().memory_info().rss / 1e6
name, engine = {name!r}, {engine!r}
if engine == "torch":
    enc = FrozenCLIPEncoder(name, "openai", device="cpu", project_patches=False)
else:
    import open_clip
    dim = open_clip.get_model_config(name)["vision_cfg"]["width"]
    enc = OnnxEncoder(onnx_path(results_root() / "onnx", name, dim, engine.split("-")[1]), 6)
x = torch.zeros(1, 3, *enc.image_size)
for _ in range(5):
    enc.encode_images(x)
print(json.dumps({{"baseline_mb": round(base, 1), "peak_mb": round(peak_mb(), 1),
                  "inspector_mb": round(peak_mb() - base, 1)}}))
"""


def main() -> int:
    root = str(repo_root())
    rows = []
    for name, engine in CONFIGS:
        code = CHILD.format(root=root, name=name, engine=engine)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if out.returncode != 0:
            print(f"{name} {engine}: failed\n{out.stderr[-500:]}", file=sys.stderr)
            continue
        result = json.loads(out.stdout.strip().splitlines()[-1])
        rows.append({"backbone": name, "engine": engine, **result})
        print(f"{name:<22}{engine:<10} {result['inspector_mb']:7.0f} MB above baseline "
              f"(peak {result['peak_mb']:.0f} MB)")

    target = repo_root() / "results" / "manifests" / "peak_ram.json"
    target.write_text(json.dumps({"method": "fresh process per config; OS peak working set "
                                  "minus RSS after imports; model load + 5 forwards, batch 1",
                                  "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
