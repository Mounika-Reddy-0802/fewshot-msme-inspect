"""The demo's engine: add a product from good photos, then inspect parts.

Kept free of Streamlit so it can be tested, and so the demo runs exactly the
method the benchmark measures -- same encoder, same prototype memory, same map
post-processing -- rather than a look-alike.

There is no trained threshold, because nothing is trained. With two or more
good photos a threshold is *suggested* by scoring each photo against the others
(leave-one-out): the worst score a known-good part gets is a floor for what
"defective" must exceed. It is a starting point for the operator, not a
calibrated operating point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from src.eval.scoring import upsample_maps
from src.model.memory import PrototypeMemory


@dataclass(frozen=True)
class AddResult:
    product: str
    images: int
    prototypes: int
    bytes_added: int
    seconds: float


@dataclass(frozen=True)
class Inspection:
    score: float
    heatmap: torch.Tensor        # (H, W) at the encoder's input resolution
    encode_ms: float
    search_ms: float


class Inspector:
    def __init__(self, encoder, top_k: int = 1, sigma: float = 4.0) -> None:
        self.encoder = encoder
        self.sigma = sigma
        self.memory = PrototypeMemory(encoder.embed_dim, encoder.grid_size, top_k=top_k)
        self._support: dict[str, list[torch.Tensor]] = {}   # per image, for leave-one-out

    @property
    def products(self) -> list[str]:
        return self.memory.classes

    def _encode(self, images) -> torch.Tensor:
        batch = torch.stack([self.encoder.preprocess(img) for img in images])
        return self.encoder.encode_images(batch).patches

    def add_product(self, name: str, images) -> AddResult:
        """Append good photos of a product. Existing products are untouched."""
        if not images:
            raise ValueError("need at least one good photo")
        before = self.memory.memory_bytes()
        start = time.perf_counter()
        patches = self._encode(images)
        self.memory.add_class(name, patches)
        seconds = time.perf_counter() - start
        self._support.setdefault(name, []).extend(p for p in patches)
        return AddResult(
            product=name,
            images=len(self._support[name]),
            prototypes=self.memory.size(name),
            bytes_added=self.memory.memory_bytes() - before,
            seconds=seconds,
        )

    def inspect(self, name: str, image) -> Inspection:
        start = time.perf_counter()
        patches = self._encode([image])
        encoded = time.perf_counter()
        scores = self.memory.score(patches, name)
        searched = time.perf_counter()
        heatmap = upsample_maps(scores.patches, self.encoder.image_size, sigma=self.sigma)[0]
        return Inspection(
            score=float(scores.image[0]),
            heatmap=heatmap,
            encode_ms=(encoded - start) * 1000,
            search_ms=(searched - encoded) * 1000,
        )

    def suggested_threshold(self, name: str) -> float | None:
        """Highest leave-one-out score among the good photos, or None if there
        are fewer than two (a photo scored against itself is always 0)."""
        support = self._support.get(name, [])
        if len(support) < 2:
            return None
        worst = 0.0
        for i, held_out in enumerate(support):
            others = PrototypeMemory(self.memory.dim, self.memory.grid, self.memory.top_k)
            others.add_class("x", torch.stack(support[:i] + support[i + 1:]))
            worst = max(worst, float(others.score(held_out.unsqueeze(0), "x").image[0]))
        return worst
