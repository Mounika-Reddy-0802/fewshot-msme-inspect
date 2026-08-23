"""Feature extraction and caching.

The backbone is frozen, so an image's features never change. Extract once per
(backbone, category), cache to disk, and every later experiment -- 0/1/5/10
shots, three seeds, adapter sweeps -- becomes vector arithmetic on cached
tensors instead of another pass over the dataset.

Stored as float16. The features are l2-normalised so their magnitudes are all
~1/sqrt(D); fp16 has ample precision there, and it halves a bank that would
otherwise run to gigabytes. They are widened back to float32 on load, before any
similarity is computed.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class ImagePathDataset(Dataset):
    """Loads images from a list of paths through the encoder's own preprocess."""

    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


@torch.no_grad()
def extract_features(
    encoder,
    paths: list[Path],
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every path. Returns (image (N, D), patches (N, P, D)) as float16."""
    if not paths:
        dim, patches = encoder.embed_dim, encoder.num_patches
        return torch.empty(0, dim, dtype=torch.float16), torch.empty(0, patches, dim, dtype=torch.float16)

    loader = DataLoader(
        ImagePathDataset(paths, encoder.preprocess),
        batch_size=batch_size,
        shuffle=False,          # order must match `paths`
        num_workers=num_workers,
    )

    images, patches = [], []
    for batch in loader:
        encoded = encoder.encode_images(batch)
        images.append(encoded.image.half().cpu())
        patches.append(encoded.patches.half().cpu())

    return torch.cat(images), torch.cat(patches)


class FeatureBank:
    """Cached features for one (backbone, category), keyed by relative path."""

    def __init__(
        self,
        backbone: str,
        category: str,
        grid: tuple[int, int],
        dim: int,
    ) -> None:
        self.backbone = backbone
        self.category = category
        self.grid = (int(grid[0]), int(grid[1]))
        self.dim = int(dim)
        self._index: dict[str, int] = {}
        self._images: torch.Tensor | None = None
        self._patches: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        encoder,
        root: Path,
        category: str,
        relative_paths: list[str],
        batch_size: int = 16,
        num_workers: int = 0,
    ) -> FeatureBank:
        base = Path(root) / category
        ordered = sorted(set(relative_paths))
        images, patches = extract_features(
            encoder, [base / p for p in ordered], batch_size, num_workers
        )

        bank = cls(encoder.name, category, encoder.grid_size, encoder.embed_dim)
        bank._index = {p: i for i, p in enumerate(ordered)}
        bank._images, bank._patches = images, patches
        return bank

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, relative_path: str) -> bool:
        return relative_path in self._index

    def _rows(self, relative_paths: list[str]) -> list[int]:
        missing = [p for p in relative_paths if p not in self._index]
        if missing:
            raise KeyError(
                f"{len(missing)} path(s) not in the {self.category} bank, "
                f"first: {missing[0]}"
            )
        return [self._index[p] for p in relative_paths]

    def images(self, relative_paths: list[str]) -> torch.Tensor:
        """Pooled embeddings for these paths, float32, in the order given."""
        return self._images[self._rows(relative_paths)].float()

    def patches(self, relative_paths: list[str]) -> torch.Tensor:
        """Patch embeddings (N, P, D), float32, in the order given."""
        return self._patches[self._rows(relative_paths)].float()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": self.backbone,
                "category": self.category,
                "grid": self.grid,
                "dim": self.dim,
                "index": self._index,
                "images": self._images,
                "patches": self._patches,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FeatureBank:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        bank = cls(state["backbone"], state["category"], state["grid"], state["dim"])
        bank._index = state["index"]
        bank._images, bank._patches = state["images"], state["patches"]
        return bank

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self._images, self._patches)
            if t is not None
        )

    def __repr__(self) -> str:
        return (
            f"FeatureBank({self.backbone}/{self.category}, {len(self)} images, "
            f"{self.nbytes() / 1e6:.1f} MB)"
        )


def bank_path(results_root: Path, backbone: str, category: str) -> Path:
    """Where a bank lives. Backbone name is part of the path -- mixing features
    from two backbones would be silently meaningless."""
    safe = backbone.replace("/", "-")
    return Path(results_root) / "feature_banks" / safe / f"{category}.pt"
