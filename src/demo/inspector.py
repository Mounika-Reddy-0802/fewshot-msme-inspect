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
from src.model.windows import text_image_scores, windowed_scores

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
    worst_patch: tuple[int, int] = (0, 0)     # (row, col) of the most anomalous patch
    patch_map: torch.Tensor | None = None     # (h, w) raw patch scores before smoothing


class Inspector:
    def __init__(self, encoder, top_k: int = 1, sigma: float = 4.0,
                 scales: tuple[int, ...] = (1,)) -> None:
        """``scales`` (1,) is the single-pass memory; (1, 2, 3) adds the multi-scale
        windows of src/model/windows.py, as in configs/proposed.yaml."""
        self.encoder = encoder
        self.sigma = sigma
        self.scales = tuple(scales)
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

    def _score(self, name: str, patches: torch.Tensor, support=None):
        """Memory scores for (B, P, D) query patches against one product only."""
        if self.scales == (1,) and support is None:
            return self.memory.score(patches, name)
        bank = torch.stack(support if support is not None else self._support[name])
        if self.scales == (1,):
            m = PrototypeMemory(self.memory.dim, self.memory.grid, self.memory.top_k)
            m.add_class("x", bank)
            return m.score(patches, "x")
        return windowed_scores(bank, patches, self.memory.grid, self.scales, self.memory.top_k)

    def _inspection(self, name, patches, pooled, encode_ms, started) -> list[Inspection]:
        scores = self._score(name, patches)
        search_ms = (time.perf_counter() - started) * 1000
        maps = upsample_maps(scores.patches, self.encoder.image_size, sigma=self.sigma)
        out = []
        w = scores.patches.shape[-1]
        for i in range(patches.shape[0]):
            memory_score = float(scores.image[i])
            score, text_score = self._fuse(name, memory_score, pooled[i])
            flat = int(scores.patches[i].flatten().argmax())
            out.append(Inspection(
                score=score, memory_score=memory_score, text_score=text_score,
                heatmap=maps[i], encode_ms=encode_ms, search_ms=search_ms,
                worst_patch=(flat // w, flat % w), patch_map=scores.patches[i],
            ))
        return out

    def inspect(self, name: str, image) -> Inspection:
        start = time.perf_counter()
        patches, pooled = self._encode([image])
        encoded = time.perf_counter()
        return self._inspection(name, patches, pooled, (encoded - start) * 1000, encoded)[0]

    def inspect_many(self, name: str, images, batch_size: int = 16, progress=None) -> list[Inspection]:
        """Score a list of images; ``progress(done, total)`` is called per batch."""
        results = []
        for i in range(0, len(images), batch_size):
            start = time.perf_counter()
            patches, pooled = self._encode(images[i:i + batch_size])
            encoded = time.perf_counter()
            per_image_ms = (encoded - start) * 1000 / patches.shape[0]
            results += self._inspection(name, patches, pooled, per_image_ms, encoded)
            if progress:
                progress(len(results), len(images))
        return results

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

    def good_photo_scores(self, name: str) -> list[float]:
        """Each good photo scored against the others (leave-one-out), exactly as
        inspect() scores a part, text fusion included. Empty below two photos (a
        photo scored against itself is always 0)."""
        support = self._support.get(name, [])
        if len(support) < 2:
            return []
        scores = []
        for i, held_out in enumerate(support):
            s = self._score(name, held_out.unsqueeze(0), support=support[:i] + support[i + 1:])
            scores.append(self._fuse(name, float(s.image[0]), self._pooled[name][i])[0])
        return scores

    def suggested_threshold(self, name: str) -> float | None:
        """Highest leave-one-out score among the good photos: the lowest threshold
        that flags none of them. None if there are fewer than two."""
        scores = self.good_photo_scores(name)
        return max(scores) if scores else None
