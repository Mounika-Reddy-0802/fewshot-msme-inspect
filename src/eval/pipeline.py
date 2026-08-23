"""End-to-end evaluation of one split.

Given a split manifest and a cached feature bank, this builds the prototype
memory from the support images, scores every test image, and returns the four
metrics.

Zero-shot (k=0) has no support images, so there is nothing to build a memory
from. It falls back to open-vocabulary prompting: patches are scored by how much
more they resemble a "defective" prompt than a "normal" one. That is what makes
the 0-shot column meaningful rather than blank.

Masks are compared at the model's input resolution (224x224) rather than the
original 900x900. Both the anomaly map and the mask go through the same resize,
so they stay aligned; the alternative -- upsampling every map to full resolution
-- costs a great deal of memory for no change in the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.eval.features import FeatureBank
from src.eval.metrics import Metrics, compute_metrics, upsample_maps
from src.model.memory import PrototypeMemory

NORMAL_PROMPTS = (
    "a photo of a {object}",
    "a photo of a flawless {object}",
    "a close-up photo of a normal {object}",
    "a photo of a {object} in good condition",
)
ANOMALOUS_PROMPTS = (
    "a photo of a {object} with a defect",
    "a photo of a damaged {object}",
    "a close-up photo of a {object} with a flaw",
    "a photo of a broken {object}",
)


@dataclass(frozen=True)
class SplitResult:
    category: str
    shots: int
    seed: int
    backbone: str
    method: str
    metrics: Metrics
    support_size: int
    test_size: int
    prototypes: int
    memory_bytes: int

    def as_row(self) -> dict:
        return {
            "backbone": self.backbone,
            "method": self.method,
            "category": self.category,
            "shots": self.shots,
            "seed": self.seed,
            **self.metrics.as_dict(),
            "support_size": self.support_size,
            "test_size": self.test_size,
            "prototypes": self.prototypes,
            "memory_bytes": self.memory_bytes,
        }


def load_masks(
    mask_paths: list[Path | None],
    size: tuple[int, int],
) -> torch.Tensor:
    """Binary masks at model resolution. Normal images get an all-zero mask.

    Resize + centre-crop mirror the image preprocessing, so a mask pixel lines up
    with the image pixel it labels. Nearest-neighbour keeps the mask binary.
    """
    from PIL import Image
    from torchvision.transforms import functional as TF

    masks = []
    for path in mask_paths:
        if path is None:
            masks.append(torch.zeros(size, dtype=torch.uint8))
            continue
        with Image.open(path) as raw:
            mask = raw.convert("L")
            mask = TF.resize(mask, size[0], interpolation=TF.InterpolationMode.NEAREST)
            mask = TF.center_crop(mask, list(size))
            masks.append((TF.pil_to_tensor(mask).squeeze(0) > 127).to(torch.uint8))
    return torch.stack(masks)


def category_text_embeddings(encoder, object_name: str) -> torch.Tensor:
    """Prompt ensembles for one product, as a (2, D) tensor: normal, anomalous.

    Each ensemble is averaged in embedding space and renormalised, which is the
    standard way to use a prompt ensemble and is markedly more stable than
    picking one wording.
    """
    label = object_name.replace("_", " ")
    normal = encoder.encode_text([p.format(object=label) for p in NORMAL_PROMPTS])
    anomalous = encoder.encode_text([p.format(object=label) for p in ANOMALOUS_PROMPTS])

    text = torch.cat([normal.mean(dim=0, keepdim=True), anomalous.mean(dim=0, keepdim=True)])
    return text / text.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def zero_shot_scores(
    encoder,
    patches: torch.Tensor,
    object_name: str,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Open-vocabulary patch scoring, no support images. (B, P) anomaly scores.

    Each patch is compared against a prompt ensemble for "normal" and one for
    "defective"; the anomaly score is the softmax weight on the defective side.
    """
    text = category_text_embeddings(encoder, object_name)
    logits = (patches.float() @ text.T) / temperature      # (B, P, 2)
    return logits.softmax(dim=-1)[..., 1]


def evaluate_split(
    manifest: dict,
    root: Path,
    bank: FeatureBank,
    encoder=None,
    top_k: int = 1,
    sigma: float = 4.0,
    image_size: tuple[int, int] = (224, 224),
    method: str = "memory",
    adapter_config=None,
    temperature: float = 0.07,
) -> SplitResult:
    """Run one (category, shots, seed) split and return its metrics.

    ``method`` selects the head: "memory" for nearest-prototype scoring,
    "adapter" for the Tip-Adapter cache kernel blended with text prompts. With
    no support images both fall back to pure zero-shot prompting, since there is
    nothing to build a cache from.
    """
    if method not in {"memory", "adapter"}:
        raise ValueError(f"unknown method {method!r}, expected 'memory' or 'adapter'")
    category = manifest["category"]
    shots = manifest["shots"]

    test_paths = [t["image"] for t in manifest["test"]]
    labels = torch.tensor([t["label"] for t in manifest["test"]])
    query_patches = bank.patches(test_paths)              # (N, P, D)

    if shots == 0:
        if encoder is None:
            raise ValueError("zero-shot needs the encoder for text prompts")
        patch_scores = zero_shot_scores(encoder, query_patches, category, temperature)
        h, w = bank.grid
        maps = patch_scores.reshape(-1, h, w)
        image_scores = patch_scores.max(dim=1).values
        prototypes = memory_bytes = 0
    else:
        memory = PrototypeMemory(dim=bank.dim, grid=bank.grid, top_k=top_k)
        memory.add_class(category, bank.patches(manifest["support"]))

        if method == "adapter":
            from src.model.adapter import AdapterConfig, TipAdapter

            config = adapter_config or AdapterConfig(top_k=top_k)
            if not isinstance(config, AdapterConfig):
                config = AdapterConfig.from_mapping(config)
            text = category_text_embeddings(encoder, category) if encoder else None
            head = TipAdapter.from_memory(memory, category, config, text)
            scored = head(query_patches)
        else:
            scored = memory.score(query_patches, category)

        maps, image_scores = scored.patches, scored.image
        prototypes = memory.total_prototypes()
        memory_bytes = memory.memory_bytes()

    # pixel metrics need the map at mask resolution
    base = Path(root) / category
    mask_paths = [base / t["mask"] if t["mask"] else None for t in manifest["test"]]
    masks = load_masks(mask_paths, image_size)
    pixel_scores = upsample_maps(maps, image_size, sigma=sigma)

    metrics = compute_metrics(
        image_scores=image_scores,
        image_labels=labels,
        pixel_scores=pixel_scores,
        pixel_labels=masks,
    )

    return SplitResult(
        category=category,
        shots=shots,
        seed=manifest["seed"],
        backbone=bank.backbone,
        method="zeroshot" if shots == 0 else method,
        metrics=metrics,
        support_size=len(manifest["support"]),
        test_size=len(test_paths),
        prototypes=prototypes,
        memory_bytes=memory_bytes,
    )
