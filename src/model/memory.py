"""Append-only prototype memory.

Adding a new product class appends its prototypes. It does not retrain, does not
touch the existing entries, and does not change any score already being produced
for the classes already in there. That is the whole claim, and
``test_memory.py`` asserts it bit-for-bit rather than taking it on trust.

Why an MSME cares: a workshop that starts inspecting a new part should not have
to retrain a model or send anything to a GPU. It stores a handful of photos of
good parts and carries on.

Scoring follows the usual normal-only anomaly formulation. The memory holds patch
embeddings of *known-good* images only. A query patch that looks like something
in memory is normal; one that matches nothing is suspicious:

    anomaly(patch) = 1 - max cosine similarity to any stored normal patch

The image score is the worst patch, because one bad region is enough to reject a
part.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.model.encoder import l2_normalise


@dataclass(frozen=True)
class AnomalyScores:
    """``image`` (B,) per-image score; ``patches`` (B, H, W) spatial map."""

    image: torch.Tensor
    patches: torch.Tensor


class PrototypeMemory:
    """Per-class banks of normal patch embeddings.

    Each class owns its own bank, which is what makes addition genuinely
    append-only: nothing is shared, re-fitted or re-normalised across classes.
    """

    def __init__(self, dim: int, grid: tuple[int, int], top_k: int = 1) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.dim = dim
        self.grid = (int(grid[0]), int(grid[1]))
        self.top_k = top_k
        self._banks: dict[str, torch.Tensor] = {}

    # --- building -----------------------------------------------------------

    def add_class(self, name: str, patches: torch.Tensor) -> None:
        """Append one class's prototypes.

        ``patches`` is (N, P, D) for N support images, or (M, D) already flat.
        Calling this again for the same class appends to it rather than
        replacing, so shots can arrive over time.
        """
        flat = self._flatten(patches)
        if name in self._banks:
            self._banks[name] = torch.cat([self._banks[name], flat], dim=0)
        else:
            self._banks[name] = flat

    def _flatten(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim == 3:
            patches = patches.reshape(-1, patches.shape[-1])
        elif patches.ndim != 2:
            raise ValueError(f"expected (N, P, D) or (M, D), got {tuple(patches.shape)}")
        if patches.shape[-1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {patches.shape[-1]}")
        return l2_normalise(patches.detach().float())

    # --- inspection ---------------------------------------------------------

    @property
    def classes(self) -> list[str]:
        return sorted(self._banks)

    def __contains__(self, name: str) -> bool:
        return name in self._banks

    def __len__(self) -> int:
        return len(self._banks)

    def size(self, name: str) -> int:
        return int(self._banks[name].shape[0])

    def total_prototypes(self) -> int:
        return sum(int(b.shape[0]) for b in self._banks.values())

    def memory_bytes(self) -> int:
        """On-disk / in-RAM footprint. Part of the efficiency story."""
        return sum(b.numel() * b.element_size() for b in self._banks.values())

    # --- scoring ------------------------------------------------------------

    def score(self, patches: torch.Tensor, name: str) -> AnomalyScores:
        """Score a query batch against one class's bank.

        ``patches`` is (B, P, D), l2-normalised. Returns per-image scores and
        the (B, H, W) anomaly map.
        """
        if name not in self._banks:
            raise KeyError(f"no class {name!r} in memory; have {self.classes}")
        if patches.ndim != 3:
            raise ValueError(f"expected (B, P, D), got {tuple(patches.shape)}")

        bank = self._banks[name]
        queries = l2_normalise(patches.float())
        b, p, _ = queries.shape

        # cosine similarity of every query patch against every stored prototype
        similarity = queries.reshape(b * p, -1) @ bank.T          # (B*P, M)
        k = min(self.top_k, similarity.shape[1])
        nearest = similarity.topk(k, dim=1).values.mean(dim=1)     # (B*P,)

        anomaly = (1.0 - nearest).reshape(b, p)
        h, w = self.grid
        return AnomalyScores(
            image=anomaly.max(dim=1).values,
            patches=anomaly.reshape(b, h, w),
        )

    # --- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dim": self.dim,
                "grid": self.grid,
                "top_k": self.top_k,
                "banks": self._banks,
            },
            path,
        )
        meta = path.with_suffix(".json")
        meta.write_text(
            json.dumps(
                {
                    "dim": self.dim,
                    "grid": list(self.grid),
                    "top_k": self.top_k,
                    "classes": {n: self.size(n) for n in self.classes},
                    "total_prototypes": self.total_prototypes(),
                    "memory_bytes": self.memory_bytes(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> PrototypeMemory:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        memory = cls(dim=state["dim"], grid=state["grid"], top_k=state["top_k"])
        memory._banks = state["banks"]
        return memory

    def __repr__(self) -> str:
        return (
            f"PrototypeMemory(classes={len(self)}, prototypes={self.total_prototypes()}, "
            f"dim={self.dim}, top_k={self.top_k}, {self.memory_bytes() / 1024:.1f} KiB)"
        )
