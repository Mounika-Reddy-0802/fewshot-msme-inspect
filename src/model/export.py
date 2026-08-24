"""ONNX export and INT8 quantisation of the frozen visual encoder.

The efficiency claim has two axes: which backbone, and at what precision. This
covers the second. The backbone never trains, so quantising it is a pure
deployment decision -- no retraining, no calibration set, nothing an MSME would
need a data scientist for.

**Dynamic quantisation** is used rather than static. Static quantisation needs a
representative calibration set, which assumes someone has already collected
images of the product being inspected -- exactly what an MSME setting up a new
line does not have. Dynamic quantisation stores weights as INT8 and computes
activation ranges at runtime, so it works out of the box on a machine that has
never seen the product.

:class:`OnnxEncoder` mirrors :class:`~src.model.encoder.FrozenCLIPEncoder` closely
enough to drop into the same benchmark, so FP32 and INT8 numbers come from an
identical pipeline rather than two code paths that might differ in some other way.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.model.encoder import EncodedBatch, l2_normalise

# CLIP's own normalisation constants, needed to rebuild the transform without
# loading the torch weights again.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_OPSET = 17


class VisualWrapper(nn.Module):
    """Tensor-in, tensor-out view of the visual tower, for tracing.

    open_clip's visual module returns the pooled embedding and the *unprojected*
    patch tokens. The projection is folded in here so the exported graph emits
    both outputs already in the joint image-text space -- the same thing
    ``FrozenCLIPEncoder`` does, kept identical so precisions stay comparable.
    """

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, tokens = self.visual(images)
        return pooled, tokens @ self.visual.proj


def export_onnx(encoder, path: str | Path, opset: int = DEFAULT_OPSET) -> Path:
    """Trace the frozen visual tower to ONNX. Returns the written path.

    PyTorch routes ``nn.MultiheadAttention`` through a fused kernel
    (``aten::_native_multi_head_attention``) that the ONNX exporter cannot lower.
    The fast path is switched off for the duration of the trace so attention
    decomposes into exportable ops, then restored -- leaving it off would quietly
    slow down every later torch timing and make FP32 look worse than it is.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    encoder.model.visual.output_tokens = True
    wrapper = VisualWrapper(encoder.model.visual).eval()
    height, width = encoder.image_size
    example = torch.zeros(1, 3, height, width)

    has_toggle = hasattr(torch.backends, "mha")
    try:
        if has_toggle:
            torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                example,
                str(path),
                input_names=["images"],
                output_names=["pooled", "patches"],
                # batch must be dynamic or the graph only ever accepts one image
                dynamic_axes={
                    "images": {0: "batch"},
                    "pooled": {0: "batch"},
                    "patches": {0: "batch"},
                },
                opset_version=opset,
                do_constant_folding=True,
            )
    finally:
        if has_toggle:
            torch.backends.mha.set_fastpath_enabled(True)

    _write_sidecar(path, encoder)
    return path


def _write_sidecar(path: Path, encoder) -> None:
    """Record what the graph expects, so OnnxEncoder needs no torch weights."""
    height, width = encoder.image_size
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "backbone": encoder.name,
                "pretrained": encoder.pretrained,
                "grid": list(encoder.grid_size),
                "embed_dim": encoder.embed_dim,
                "image_size": [height, width],
                "mean": list(CLIP_MEAN),
                "std": list(CLIP_STD),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def quantize_int8(fp32_path: str | Path, int8_path: str | Path | None = None) -> Path:
    """Dynamically quantise an exported graph to INT8 weights."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path) if int8_path else fp32_path.with_name(
        fp32_path.stem + "-int8.onnx"
    )

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

    sidecar = fp32_path.with_suffix(".json")
    if sidecar.exists():
        int8_path.with_suffix(".json").write_text(
            sidecar.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return int8_path


def build_preprocess(image_size: tuple[int, int], mean=CLIP_MEAN, std=CLIP_STD):
    """Rebuild CLIP's eval transform without loading the torch model."""
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Normalize,
        Resize,
        ToTensor,
    )

    return Compose([
        Resize(image_size[0], interpolation=InterpolationMode.BICUBIC),
        CenterCrop(list(image_size)),
        _to_rgb,
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


def _to_rgb(image):
    return image.convert("RGB")


class OnnxEncoder:
    """An exported encoder, run through ONNX Runtime.

    Exposes the same surface the benchmark uses from ``FrozenCLIPEncoder``:
    ``name``, ``grid_size``, ``num_patches``, ``embed_dim``, ``image_size``,
    ``preprocess`` and ``encode_images``.

    Text encoding is deliberately absent. Only the visual tower is exported, so
    zero-shot prompting is not available at INT8 -- which costs nothing, since
    the alpha sweep showed the text branch never helps once there is at least one
    support image.
    """

    def __init__(self, onnx_path: str | Path, threads: int | None = None) -> None:
        import onnxruntime

        self.path = Path(onnx_path)
        meta = json.loads(self.path.with_suffix(".json").read_text(encoding="utf-8"))

        self.backbone = meta["backbone"]
        self.pretrained = meta["pretrained"]
        self._grid = (int(meta["grid"][0]), int(meta["grid"][1]))
        self._embed_dim = int(meta["embed_dim"])
        self._image_size = (int(meta["image_size"][0]), int(meta["image_size"][1]))
        self.precision = "int8" if "int8" in self.path.stem else "fp32"

        options = onnxruntime.SessionOptions()
        if threads:
            # pinned for the same reason torch threads are: comparability
            options.intra_op_num_threads = threads
        self.session = onnxruntime.InferenceSession(
            str(self.path), options, providers=["CPUExecutionProvider"]
        )
        self.preprocess = build_preprocess(self._image_size, meta["mean"], meta["std"])

    @property
    def name(self) -> str:
        """Precision is part of the identity: feature banks from FP32 and INT8
        must never land in the same cache file."""
        return f"{self.backbone}-{self.precision}"

    @property
    def grid_size(self) -> tuple[int, int]:
        return self._grid

    @property
    def num_patches(self) -> int:
        return self._grid[0] * self._grid[1]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def encode_images(self, images: torch.Tensor) -> EncodedBatch:
        pooled, patches = self.session.run(
            ["pooled", "patches"], {"images": images.detach().cpu().numpy()}
        )
        return EncodedBatch(
            image=l2_normalise(torch.from_numpy(pooled)),
            patches=l2_normalise(torch.from_numpy(patches)),
            grid=self._grid,
        )

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        raise NotImplementedError(
            "only the visual tower is exported; zero-shot prompting needs the "
            "torch encoder (see FrozenCLIPEncoder)"
        )

    def file_bytes(self) -> int:
        return self.path.stat().st_size

    def __repr__(self) -> str:
        h, w = self._grid
        return (
            f"OnnxEncoder({self.name}, grid={h}x{w}, dim={self._embed_dim}, "
            f"{self.file_bytes() / 1e6:.0f} MB)"
        )
