"""Cache CLS + patch-token CLIP features for every image the split manifest
needs (T2.3).

Extracts once per (image, backbone) pair -- pooled CLS embedding and patch
tokens -- for every image referenced by the frozen split manifest: the
10-shot references (a superset of 0/1/5-shot, since splits are nested) plus
every test image. This is the expensive step; everything downstream
(prototype memory, adapter training, evaluation) should read these cached
features instead of the network, per Stage 2 of the blueprint.

Run a smoke test on 2 images first, to confirm the open_clip API on your
installed version behaves as expected before committing to a full run:

    python scripts/build_feature_bank.py --category bottle --limit 2

Then the full run:

    python scripts/build_feature_bank.py --category bottle
    python scripts/build_feature_bank.py            # every category in the manifest

Writes one .pt file per image under
${results_root}/features/<backbone_name>/<category>/<split>/<stem>.pt
containing {"cls": tensor[D], "patch_tokens": tensor[N, D]}. This whole
features/ folder must stay out of Git (see GIT_RULES.md) -- it lives under
results/, which is gitignored except for manifests/, so it should already
be excluded automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import dataset_dir, results_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def load_manifest() -> dict:
    path = results_root() / "manifests" / "splits" / "mvtec_ad.json"
    return json.loads(path.read_text(encoding="utf-8"))


def images_to_process(manifest: dict, categories: list[str]) -> dict[str, list[str]]:
    """category -> sorted relative image paths needed (max-shot refs + all test)."""
    out = {}
    for category in categories:
        entry = manifest["categories"][category]
        max_shot_key = str(max(manifest["shot_counts"]))
        refs = set(entry["shots"][max_shot_key]) | set(entry["test"])
        out[category] = sorted(refs)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", nargs="+", default=None, help="defaults to every category in the manifest")
    parser.add_argument("--limit", type=int, default=None, help="cap images per category -- use for a smoke test")
    args = parser.parse_args()

    config = load_config("base.yaml")
    seed_everything(config["experiment"]["seed"])

    import torch
    import open_clip
    from PIL import Image

    backbone_cfg = config["backbone"]
    print(f"loading {backbone_cfg['name']} / {backbone_cfg['pretrained']} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone_cfg["name"], pretrained=backbone_cfg["pretrained"]
    )
    model.eval()
    model.visual.output_tokens = True  # ask the visual trunk to also return patch tokens

    manifest = load_manifest()
    categories = args.category or list(manifest["categories"].keys())
    todo = images_to_process(manifest, categories)

    data_root = dataset_dir("mvtec_ad").parent  # manifest paths are relative to this
    out_root = results_root() / "features" / backbone_cfg["name"]

    for category, rel_paths in todo.items():
        if args.limit:
            rel_paths = rel_paths[: args.limit]
        print(f"{category}: {len(rel_paths)} images")

        for rel_path in rel_paths:
            img_path = data_root / rel_path
            out_path = out_root / Path(rel_path).with_suffix(".pt")
            if out_path.exists():
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)

            with torch.no_grad():
                pooled, tokens = model.visual(image)

            print(f"    {rel_path}  cls={tuple(pooled.shape)}  patch_tokens={tuple(tokens.shape)}")

            torch.save(
                {"cls": pooled.squeeze(0).clone(), "patch_tokens": tokens.squeeze(0).clone()},
                out_path,
            )

        print(f"  done -> {out_root / category}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
