"""Frozen OpenCLIP feature extractor.

The backbone never trains. Everything downstream -- prototype memory, adapter,
zero-shot prompting -- consumes what this produces, so two properties matter:

**Patch tokens, not just a pooled vector.** Pixel-level AUROC needs a score per
image region. ``encode_images`` returns the pooled embedding *and* the per-patch
tokens reshaped to their spatial grid (14x14 for ViT-B/16 at 224px).

**One shared space.** Raw patch tokens come out at the transformer width (768 for
ViT-B/16), not the joint image-text width (512). They are projected through the
model's own ``visual.proj`` so patches, images and text prompts are all directly
comparable. Without that, open-vocabulary prompting cannot score a patch at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EncodedBatch:
    """Features for one batch of images.

    ``image``   (B, D)        pooled embedding, l2-normalised
    ``patches`` (B, P, D)     per-patch embeddings, l2-normalised
    ``grid``    (H, W)        spatial layout of the P patches, H*W == P
    """

    image: torch.Tensor
    patches: torch.Tensor
    grid: tuple[int, int]

    def patch_map(self) -> torch.Tensor:
        """Patches reshaped to (B, H, W, D) for spatial work."""
        b, p, d = self.patches.shape
        h, w = self.grid
        return self.patches.reshape(b, h, w, d)


def l2_normalise(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


class FrozenCLIPEncoder:
    """A frozen OpenCLIP backbone.

    Loading with ``pretrained="openai"`` requires a ``-quickgelu`` model name:
    the OpenAI weights were trained with QuickGELU, and open_clip will happily
    build the wrong activation with only a warning. :func:`check_backbone_pair`
    turns that into an error.
    """

    def __init__(
        self,
        name: str = "ViT-B-16-quickgelu",
        pretrained: str = "openai",
        device: str = "cpu",
        project_patches: bool = True,
        layers: tuple[int, ...] | None = None,
    ) -> None:
        """``layers``: 1-based transformer blocks whose patch tokens are used, e.g.
        (6, 9, 12) for ViT-B/16. Each block's tokens get the final layer norm and
        are l2-normalised, then concatenated, so the cosine between two patches is
        the mean of the per-block cosines. None keeps the final block only."""
        import open_clip

        check_backbone_pair(name, pretrained)

        if layers is not None and project_patches:
            raise ValueError("multi-layer patches are unprojected; pass project_patches=False")
        self.backbone = name
        self.layers = tuple(int(i) for i in layers) if layers else None
        # the layer set is part of the identity, so its feature banks never mix
        self.name = name if not self.layers else f"{name}-ml{'-'.join(map(str, self.layers))}"
        self.pretrained = pretrained
        self.device = device
        self.project_patches = project_patches

        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        model = model.to(device).eval()
        model.requires_grad_(False)          # frozen, and provably so
        model.visual.output_tokens = True    # return patch tokens alongside the pooling

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(name)

        if self.layers:
            depth = len(model.visual.transformer.resblocks)
            bad = [i for i in self.layers if not 1 <= i <= depth]
            if bad:
                raise ValueError(f"{name} has blocks 1..{depth}; got {bad}")

    @property
    def grid_size(self) -> tuple[int, int]:
        grid = self.model.visual.grid_size
        return (int(grid[0]), int(grid[1]))

    @property
    def num_patches(self) -> int:
        h, w = self.grid_size
        return h * w

    @property
    def embed_dim(self) -> int:
        """Dimension of whatever ``encode_images`` returns for patches."""
        if self.project_patches:
            return int(self.model.visual.output_dim)
        return int(self.model.visual.transformer.width) * (len(self.layers) if self.layers else 1)

    @property
    def image_size(self) -> tuple[int, int]:
        size = self.model.visual.image_size
        return (int(size[0]), int(size[1])) if isinstance(size, (tuple, list)) else (int(size), int(size))

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> EncodedBatch:
        """Encode a preprocessed batch, (B, 3, H, W) -> pooled + patch features."""
        images = images.to(self.device)

        if self.layers:
            out = self.model.visual.forward_intermediates(
                images, indices=[i - 1 for i in self.layers],
                normalize_intermediates=True, output_fmt="NLC",
            )
            tokens = torch.cat([l2_normalise(t) for t in out["image_intermediates"]], dim=-1)
            return EncodedBatch(
                image=l2_normalise(out["image_features"]),
                patches=l2_normalise(tokens),
                grid=self.grid_size,
            )

        pooled, tokens = self.model.visual(images)

        if self.project_patches:
            proj = self.model.visual.proj
            if proj is None:
                raise RuntimeError(
                    f"{self.name} has no visual.proj, so patch tokens cannot be put "
                    "in the joint image-text space; pass project_patches=False"
                )
            tokens = tokens @ proj

        return EncodedBatch(
            image=l2_normalise(pooled),
            patches=l2_normalise(tokens),
            grid=self.grid_size,
        )

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        """Encode prompts into the joint space, l2-normalised. (N, D)."""
        tokens = self.tokenizer(prompts).to(self.device)
        return l2_normalise(self.model.encode_text(tokens))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def __repr__(self) -> str:
        h, w = self.grid_size
        return (
            f"FrozenCLIPEncoder({self.name}/{self.pretrained}, device={self.device}, "
            f"grid={h}x{w}, dim={self.embed_dim})"
        )


def check_backbone_pair(name: str, pretrained: str) -> None:
    """Refuse the QuickGELU mismatch instead of warning about it.

    open_clip pairs OpenAI weights with a non-QuickGELU model without erroring,
    and the resulting features are quietly different -- measurably so, cosine
    similarity ~0.975 against the correct ones. Silent and wrong is the worst
    failure mode for a benchmark, so this is fatal.
    """
    if pretrained == "openai" and not name.endswith("-quickgelu"):
        raise ValueError(
            f"{name!r} with pretrained='openai' silently builds the wrong activation. "
            f"Use {name}-quickgelu instead."
        )
