"""ONNX export plumbing that does not need the real weights."""

from __future__ import annotations

import torch
from torch import nn

from src.model.export import VisualWrapper, onnx_path


class FakeVisual(nn.Module):
    """Stands in for open_clip's visual tower: 768-wide tokens, 512-dim proj."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.randn(768, 512))

    def forward(self, images):
        batch = images.shape[0]
        return torch.randn(batch, 512), torch.randn(batch, 4, 768)


def test_wrapper_projects_only_when_asked():
    images = torch.zeros(2, 3, 8, 8)
    _, projected = VisualWrapper(FakeVisual(), project_patches=True)(images)
    _, raw = VisualWrapper(FakeVisual(), project_patches=False)(images)
    assert projected.shape[-1] == 512
    assert raw.shape[-1] == 768


def test_graph_names_carry_dim_and_precision(tmp_path):
    fp32 = onnx_path(tmp_path, "ViT-B-16-quickgelu", 768)
    int8 = onnx_path(tmp_path, "ViT-B-16-quickgelu", 768, "int8")
    d512 = onnx_path(tmp_path, "ViT-B-16-quickgelu", 512, "int8")
    assert fp32.name == "ViT-B-16-quickgelu-d768.onnx"
    assert int8.name == "ViT-B-16-quickgelu-d768-int8.onnx"
    # a 512-dim graph must never be picked up for a 768-dim run
    assert len({fp32, int8, d512}) == 3
    # OnnxEncoder infers precision from the stem
    assert "int8" in int8.stem and "int8" not in fp32.stem
