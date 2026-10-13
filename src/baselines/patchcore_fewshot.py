"""PatchCore as a few-shot baseline, on exactly our splits.

PatchCore (Roth et al., CVPR 2022) is the strongest memory-bank method, but it is
published with hundreds of normal images per product. The fair comparison for
this project gives it the *same* 1/5/10 good photos our method gets -- same
manifests, seeds and test images -- which is also how WinCLIP reports it.

anomalib's ``PatchcoreModel`` is used directly rather than through its data
module, so the support set is exactly the manifest's:

* WideResNet-50-2 (ImageNet), layers 2 + 3, 3x3 average pooling -- anomalib's
  defaults and configs/baseline.yaml;
* images and masks at 224 px with the same resize + centre crop as our pipeline,
  ImageNet normalisation (what this backbone was trained with);
* **no coreset subsampling**: with <= 10 photos the bank is already small, and
  subsampling would only discard information. This is PatchCore at its
  strongest in this regime.

Scores and maps come from anomalib's own nearest-neighbour search, re-weighting
(num_neighbors = 9) and Gaussian-smoothed anomaly map.
"""

from __future__ import annotations

from pathlib import Path

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(size: int = 224):
    from torchvision.transforms import (
        CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor,
    )
    return Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size),
        lambda im: im.convert("RGB"),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FewShotPatchCore:
    def __init__(self, backbone: str = "wide_resnet50_2", layers=("layer2", "layer3"),
                 num_neighbors: int = 9, size: int = 224) -> None:
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        self.model = PatchcoreModel(layers=list(layers), backbone=backbone,
                                    num_neighbors=num_neighbors).eval()
        self.transform = preprocess(size)
        self.size = size
        self.name = f"patchcore-{backbone.replace('_', '')}"

    @torch.no_grad()
    def embed(self, paths: list[Path], batch_size: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
        """(N, P, D) patch embeddings, float16, and the (H, W) patch grid."""
        from PIL import Image

        m = self.model
        out, grid = [], None
        for i in range(0, len(paths), batch_size):
            x = torch.stack([self.transform(Image.open(p)) for p in paths[i:i + batch_size]])
            features = m.feature_extractor(x)
            features = {k: m.feature_pooler(v) for k, v in features.items()}
            emb = m.generate_embedding(features)                     # (B, D, H, W)
            b, d, h, w = emb.shape
            grid = (h, w)
            out.append(emb.permute(0, 2, 3, 1).reshape(b, h * w, d).half())
        return torch.cat(out), grid

    @torch.no_grad()
    def score(self, support: torch.Tensor, query: torch.Tensor, grid: tuple[int, int],
              chunk: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """Image scores (B,) and anomaly maps (B, size, size) for ``query``
        against a memory bank of every ``support`` patch."""
        m = self.model
        m.memory_bank = support.float().reshape(-1, support.shape[-1])
        scores, maps = [], []
        h, w = grid
        for i in range(0, query.shape[0], chunk):
            q = query[i:i + chunk].float()
            b = q.shape[0]
            emb = q.reshape(-1, q.shape[-1])
            patch_scores, locations = m.nearest_neighbors(embedding=emb, n_neighbors=1)
            patch_scores = patch_scores.reshape(b, -1)
            locations = locations.reshape(b, -1)
            scores.append(m.compute_anomaly_score(patch_scores, locations, emb))
            amap = m.anomaly_map_generator(patch_scores.reshape(b, 1, h, w), (self.size, self.size))
            maps.append(amap.squeeze(1))
        return torch.cat(scores), torch.cat(maps)
