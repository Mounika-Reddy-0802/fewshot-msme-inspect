"""Tip-Adapter-style cache head over the prototype memory.

The prototype memory scores a patch by its single nearest stored neighbour. That
throws away most of the evidence: if a patch is a decent match to twenty stored
prototypes, that is a stronger signal of normality than one lucky match, and the
hard max cannot express it.

Tip-Adapter replaces the max with a similarity kernel over the whole cache

    A(s) = exp(-beta * (1 - s))          s = cosine similarity, A(1) = 1

and reads normality off the aggregated affinity. ``beta`` sets how sharply
attention falls away from a perfect match -- large beta approaches the plain
nearest-neighbour behaviour, small beta spreads credit across the cache.

The second half is open-vocabulary. Text prompts give an anomaly estimate that
needs no support images at all, and it is strongest exactly where the memory is
weakest -- at 0 and 1 shots, when the cache is nearly empty. ``alpha`` mixes them:

    anomaly = (1 - alpha) * cache_anomaly + alpha * text_anomaly

Both are training-free, which is the point: an MSME adds a product class by
storing photos, and nothing needs a GPU or a training run. The cache keys are
held as an ``nn.Parameter`` so the fine-tuned variant (Tip-Adapter-F, week 5) can
unfreeze them without any change to how scoring works.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.model.encoder import l2_normalise
from src.model.memory import AnomalyScores, PrototypeMemory


@dataclass(frozen=True)
class AdapterConfig:
    alpha: float = 0.3      # weight on the text branch
    beta: float = 5.5       # sharpness of the affinity kernel
    top_k: int = 1          # cache entries aggregated per query patch
    trainable: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


class TipAdapter(nn.Module):
    """Cache head for one class.

    Built from a class's stored prototypes, optionally blended with text
    prompts. Scoring returns the same :class:`AnomalyScores` the prototype
    memory returns, so the two are drop-in comparable in the benchmark.
    """

    def __init__(
        self,
        keys: torch.Tensor,
        grid: tuple[int, int],
        config: AdapterConfig | None = None,
        text_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if keys.ndim != 2:
            raise ValueError(f"expected cache keys (M, D), got {tuple(keys.shape)}")

        self.config = config or AdapterConfig()
        self.grid = (int(grid[0]), int(grid[1]))

        keys = l2_normalise(keys.detach().float())
        self.keys = nn.Parameter(keys, requires_grad=self.config.trainable)

        # (2, D): row 0 normal, row 1 anomalous. Absent -> pure cache scoring.
        if text_embeddings is not None:
            if text_embeddings.shape[0] != 2:
                raise ValueError(
                    f"expected 2 text embeddings (normal, anomalous), "
                    f"got {text_embeddings.shape[0]}"
                )
            self.register_buffer("text", l2_normalise(text_embeddings.detach().float()))
        else:
            self.text = None

    @classmethod
    def from_memory(
        cls,
        memory: PrototypeMemory,
        class_name: str,
        config: AdapterConfig | None = None,
        text_embeddings: torch.Tensor | None = None,
    ) -> TipAdapter:
        """Initialise the cache from an existing prototype memory.

        The adapter is a different *read* over the same prototypes, so a class
        already in memory needs no re-extraction to gain an adapter.
        """
        if class_name not in memory:
            raise KeyError(f"no class {class_name!r} in memory; have {memory.classes}")
        return cls(memory._banks[class_name], memory.grid, config, text_embeddings)

    @property
    def num_keys(self) -> int:
        return int(self.keys.shape[0])

    def cache_anomaly(self, patches: torch.Tensor) -> torch.Tensor:
        """Anomaly from the cache alone. (B, P) in [0, 1]."""
        queries = l2_normalise(patches.float())
        b, p, _ = queries.shape

        similarity = queries.reshape(b * p, -1) @ self.keys.T
        # exp(-beta(1 - s)): 1 at a perfect match, decaying with distance
        affinity = torch.exp(-self.config.beta * (1.0 - similarity))

        k = min(self.config.top_k, affinity.shape[1])
        normality = affinity.topk(k, dim=1).values.mean(dim=1)
        return (1.0 - normality).reshape(b, p)

    def text_anomaly(self, patches: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """Anomaly from the prompts alone. (B, P) in [0, 1]."""
        if self.text is None:
            raise RuntimeError("no text embeddings were provided")
        logits = (l2_normalise(patches.float()) @ self.text.T) / temperature
        return logits.softmax(dim=-1)[..., 1]

    def forward(self, patches: torch.Tensor) -> AnomalyScores:
        if patches.ndim != 3:
            raise ValueError(f"expected (B, P, D), got {tuple(patches.shape)}")

        anomaly = self.cache_anomaly(patches)
        if self.text is not None and self.config.alpha > 0:
            anomaly = (
                (1.0 - self.config.alpha) * anomaly
                + self.config.alpha * self.text_anomaly(patches)
            )

        h, w = self.grid
        return AnomalyScores(
            image=anomaly.max(dim=1).values,
            patches=anomaly.reshape(-1, h, w),
        )

    def score(self, patches: torch.Tensor) -> AnomalyScores:
        """Alias for :meth:`forward`, matching PrototypeMemory.score."""
        return self(patches)

    def extra_repr(self) -> str:
        return (
            f"keys={self.num_keys}, alpha={self.config.alpha}, beta={self.config.beta}, "
            f"top_k={self.config.top_k}, text={'yes' if self.text is not None else 'no'}, "
            f"trainable={self.config.trainable}"
        )
