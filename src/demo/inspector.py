"""The demo's engine: add a product from good photos, then inspect parts.

Kept free of Streamlit so it can be tested, and so the demo runs exactly the
method the benchmark measures -- same encoder, same prototype memory, same map
post-processing -- rather than a look-alike.

A product can also carry a **typed description** of what goes wrong with it
("cracked rim", "missing solder"). That becomes an extra "defective" prompt next
to the standard ensemble, and the image score is fused with the memory's:
``(1 - w) * memory + w * text``, w = 0.5 -- the setting the text-fusion ablation
measured (results/manifests/text_fusion.json). Text cannot move the heatmap;
768-dim patches have no text counterpart.

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

from src.eval.pipeline import ANOMALOUS_PROMPTS, NORMAL_PROMPTS
from src.eval.scoring import upsample_maps
from src.model.encoder import l2_normalise
from src.model.memory import PrototypeMemory
from src.model.windows import text_image_scores

TEXT_WEIGHT = 0.5


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
    memory_score: float
    text_score: float | None
    heatmap: torch.Tensor        # (H, W) at the encoder's input resolution
    encode_ms: float
    search_ms: float


class Inspector:
    def __init__(self, encoder, top_k: int = 1, sigma: float = 4.0) -> None:
        self.encoder = encoder
        self.sigma = sigma
        self.memory = PrototypeMemory(encoder.embed_dim, encoder.grid_size, top_k=top_k)
        self._support: dict[str, list[torch.Tensor]] = {}   # per image, for leave-one-out
        self._pooled: dict[str, list[torch.Tensor]] = {}
        self._text: dict[str, torch.Tensor] = {}              # (2, D) normal, defective
        self.descriptions: dict[str, str] = {}

    @property
    def products(self) -> list[str]:
        return self.memory.classes

    def _encode(self, images):
        batch = torch.stack([self.encoder.preprocess(img) for img in images])
        encoded = self.encoder.encode_images(batch)
        return encoded.patches, encoded.image

    def _prompts(self, name: str, description: str) -> torch.Tensor:
        label = name.replace("_", " ").replace("-", " ")
        normal = self.encoder.encode_text([p.format(object=label) for p in NORMAL_PROMPTS])
        defective = [p.format(object=label) for p in ANOMALOUS_PROMPTS]
        defective += [f"a photo of a {label} with {description}", description]
        anomalous = self.encoder.encode_text(defective)
        text = torch.stack([normal.mean(0), anomalous.mean(0)])
        return l2_normalise(text)

    def add_product(self, name: str, images, description: str | None = None) -> AddResult:
        """Append good photos (and optionally a defect description) for a product.
        Existing products are untouched."""
        if not images:
            raise ValueError("need at least one good photo")
        before = self.memory.memory_bytes()
        start = time.perf_counter()
        patches, pooled = self._encode(images)
        self.memory.add_class(name, patches)
        if description:
            self._text[name] = self._prompts(name, description)
            self.descriptions[name] = description
        seconds = time.perf_counter() - start
        self._support.setdefault(name, []).extend(p for p in patches)
        self._pooled.setdefault(name, []).extend(p for p in pooled)
        return AddResult(
            product=name,
            images=len(self._support[name]),
            prototypes=self.memory.size(name),
            bytes_added=self.memory.memory_bytes() - before,
            seconds=seconds,
        )

    def _fuse(self, name: str, memory_score: float, pooled: torch.Tensor) -> tuple[float, float | None]:
        text = self._text.get(name)
        if text is None:
            return memory_score, None
        text_score = float(text_image_scores(pooled.unsqueeze(0), text)[0])
        return (1 - TEXT_WEIGHT) * memory_score + TEXT_WEIGHT * text_score, text_score

    def inspect(self, name: str, image) -> Inspection:
        start = time.perf_counter()
        patches, pooled = self._encode([image])
        encoded = time.perf_counter()
        scores = self.memory.score(patches, name)
        searched = time.perf_counter()
        heatmap = upsample_maps(scores.patches, self.encoder.image_size, sigma=self.sigma)[0]
        memory_score = float(scores.image[0])
        score, text_score = self._fuse(name, memory_score, pooled[0])
        return Inspection(
            score=score,
            memory_score=memory_score,
            text_score=text_score,
            heatmap=heatmap,
            encode_ms=(encoded - start) * 1000,
            search_ms=(searched - encoded) * 1000,
        )

    def save(self, path) -> None:
        """Write every product to one file, so the app can be closed and
        reopened without adding them again. The file records which backbone made
        the features; loading it under another backbone is refused."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder": self.encoder.name,
            "dim": self.memory.dim,
            "grid": self.memory.grid,
            "top_k": self.memory.top_k,
            "support": {k: torch.stack(v) for k, v in self._support.items()},
            "pooled": {k: torch.stack(v) for k, v in self._pooled.items()},
            "text": dict(self._text),
            "descriptions": dict(self.descriptions),
        }, path)

    def load(self, path) -> list[str]:
        """Add every product saved in ``path``; returns their names.

        Products are rebuilt by appending the saved patches, exactly as
        add_product() stored them, so scores are bit-identical to before the
        save. A product already in memory is not loaded twice."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["encoder"] != self.encoder.name or state["dim"] != self.memory.dim:
            raise ValueError(
                f"saved with {state['encoder']} ({state['dim']}-dim); this inspector "
                f"uses {self.encoder.name} ({self.memory.dim}-dim)"
            )
        loaded = []
        for name, patches in state["support"].items():
            if name in self.memory:
                continue
            self.memory.add_class(name, patches)
            self._support[name] = list(patches)
            self._pooled[name] = list(state["pooled"][name])
            if name in state["text"]:
                self._text[name] = state["text"][name]
                self.descriptions[name] = state["descriptions"][name]
            loaded.append(name)
        return loaded

    def suggested_threshold(self, name: str) -> float | None:
        """Highest leave-one-out score among the good photos, or None if there
        are fewer than two (a photo scored against itself is always 0).
        Scored exactly as inspect() scores, text fusion included."""
        support = self._support.get(name, [])
        if len(support) < 2:
            return None
        worst = 0.0
        for i, held_out in enumerate(support):
            others = PrototypeMemory(self.memory.dim, self.memory.grid, self.memory.top_k)
            others.add_class("x", torch.stack(support[:i] + support[i + 1:]))
            memory_score = float(others.score(held_out.unsqueeze(0), "x").image[0])
            worst = max(worst, self._fuse(name, memory_score, self._pooled[name][i])[0])
        return worst
