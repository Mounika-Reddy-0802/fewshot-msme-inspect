"""Smoke test the environment.

Prints the versions the benchmark depends on, confirms the frozen backbone
loads and encodes, and confirms the configs resolve. Run this after any
install or upgrade:

    python scripts/validate_env.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import data_root, results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ImportError:
        return "MISSING"
    return getattr(module, "__version__", "unknown")


def main() -> int:
    print(f"python           {platform.python_version()} on {platform.system()}")
    for name in ("torch", "torchvision", "open_clip", "anomalib", "onnxruntime", "yaml"):
        print(f"{name:<16} {_version(name)}")

    import torch

    print(f"torch threads    {torch.get_num_threads()}  cuda={torch.cuda.is_available()}")
    print(f"data root        {data_root()}")
    print(f"results root     {results_root()}")

    seed_everything(load_config("base.yaml")["experiment"]["seed"])

    for name in ("baseline.yaml", "prototype.yaml", "adapter.yaml"):
        config = load_config(name)
        print(f"config ok        {name} -> {config['experiment']['name']}")

    try:
        import open_clip
    except ImportError:
        print("open_clip missing, skipping encode check")
        return 1

    backbone = load_config("base.yaml")["backbone"]
    print(f"loading          {backbone['name']} / {backbone['pretrained']} ...")
    model, _, _ = open_clip.create_model_and_transforms(
        backbone["name"], pretrained=backbone["pretrained"]
    )
    model.eval()
    with torch.no_grad():
        features = model.encode_image(torch.zeros(1, 3, 224, 224))
    print(f"encode ok        image features {tuple(features.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
