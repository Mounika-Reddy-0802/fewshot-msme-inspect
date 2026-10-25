"""State and helpers shared by the pages of the Streamlit app (app/demo.py).

Kept out of the page files so every page sees the same engine, the same saved
products and the same chart styling.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.data.splits import load_manifest, manifest_path, resolve_split, split_for
from src.demo.inspector import Inspector
from src.utils.paths import data_root, repo_root, results_root

# chart colours: the first slots of the validated reference palette (blue,
# orange, aqua pass colour-blind checks in every pairing); text stays in ink
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED = "#0b0b0b", "#52514e"

# colour follows the model on every chart, so a model keeps its colour across pages
MODEL_COLOURS = {"ViT-B/16": BLUE, "PatchCore (same photos)": ORANGE, "ViT-B/32": AQUA,
                 "ViT-L/14": YELLOW, "ViT-B/16 multi-layer": MAGENTA}


def colours_for(models) -> list[str]:
    spare = iter(c for c in (BLUE, ORANGE, AQUA, YELLOW, MAGENTA) if c not in
                 {MODEL_COLOURS.get(m) for m in models})
    return [MODEL_COLOURS.get(m) or next(spare, MUTED) for m in models]

ENGINES = {
    "b32": {"label": "ViT-B/32 (fastest, ~37 ms per part)", "backbone": "ViT-B-32-quickgelu",
            "kind": "torch", "store": "ViT-B-32-quickgelu.pt",
            "note": "Enough when defects cover more than ~2% of the photo (MVTec, NEU-DET)."},
    "b16": {"label": "ViT-B/16 (small defects, ~88 ms)", "backbone": "ViT-B-16-quickgelu",
            "kind": "torch", "store": "ViT-B-16-quickgelu.pt",
            "note": "Needed when defects are tiny, like VisA's (0.04-0.9% of the photo)."},
    "b16ml": {"label": "ViT-B/16 multi-layer (most accurate, ~90 ms)", "backbone": "ViT-B-16-quickgelu",
              "kind": "torch", "layers": (6, 9, 12), "store": "ViT-B-16-quickgelu-ml6-9-12.pt",
              "note": "Patch features from blocks 6, 9 and 12 instead of the last block only. "
                      "Best accuracy and localisation in the benchmark (release 1.1)."},
    "b32int8": {"label": "ViT-B/32 INT8 via ONNX (~33 ms, no descriptions)",
                "backbone": "ViT-B-32-quickgelu", "kind": "onnx",
                "store": "ViT-B-32-quickgelu-int8.pt",
                "note": "4x smaller model, 1.3x faster; costs about 0.013 AUROC. Text descriptions "
                        "need CLIP's text tower, which is not exported."},
}
DATASETS = {"MVTec AD": "mvtec_ad", "VisA": "visa", "NEU-DET": "neu_det", "MPDD": "mpdd"}


def onnx_graph(backbone: str) -> Path:
    from src.model.export import onnx_path
    return onnx_path(results_root() / "onnx", backbone, 768, "int8")


def available_engines() -> dict:
    return {k: v for k, v in ENGINES.items()
            if v["kind"] != "onnx" or onnx_graph(v["backbone"]).exists()}


@st.cache_resource(show_spinner="Loading the frozen CLIP model (first time only) ...")
def load_encoder(engine_id: str):
    import torch
    torch.set_num_threads(6)        # one per performance core; see docs section 7
    spec = ENGINES[engine_id]
    if spec["kind"] == "onnx":
        from src.model.export import OnnxEncoder
        return OnnxEncoder(onnx_graph(spec["backbone"]), threads=6)
    from src.model.encoder import FrozenCLIPEncoder
    return FrozenCLIPEncoder(spec["backbone"], "openai", device="cpu", project_patches=False,
                             layers=spec.get("layers"))


def engine_id() -> str:
    return st.session_state.get("engine", "b16ml")


def scales() -> tuple[int, ...]:
    return (1, 2, 3) if st.session_state.get("windows", True) else (1,)


def store_path(eid: str | None = None) -> Path:
    return results_root() / "demo" / ENGINES[eid or engine_id()]["store"]


def inspector() -> Inspector:
    """The inspector for the chosen engine, with its saved products loaded."""
    eid = engine_id()
    key = f"inspector::{eid}"
    if key not in st.session_state:
        insp = Inspector(load_encoder(eid))
        path = store_path(eid)
        if path.exists():
            try:
                st.session_state[f"loaded::{eid}"] = insp.load(path)
            except Exception as exc:          # a store from an older build: start clean
                st.session_state[f"loaded::{eid}"] = []
                st.warning(f"Could not reopen saved products ({type(exc).__name__}); starting empty.")
        st.session_state[key] = insp
        st.session_state.setdefault("history", {})
    insp = st.session_state[key]
    insp.scales = scales()
    return insp


def has_inspector() -> bool:
    return f"inspector::{engine_id()}" in st.session_state


def forget_products() -> None:
    eid = engine_id()
    store_path(eid).unlink(missing_ok=True)
    for k in (f"inspector::{eid}", f"loaded::{eid}", "history", "batch"):
        st.session_state.pop(k, None)


# ------------------------------------------------------------------ datasets

def splits_dir() -> Path:
    return repo_root() / "results" / "manifests" / "splits"


@st.cache_data
def dataset_categories(dataset: str) -> list[str]:
    path = manifest_path(splits_dir(), dataset)
    if not path.exists():
        return []
    cats = sorted(load_manifest(path)["categories"])
    from src.data.splits import category_root
    root = category_root(dataset, data_root())
    return [c for c in cats if (root / c).is_dir()]


@st.cache_data
def dataset_split(dataset: str, category: str, shots: int):
    manifest = load_manifest(manifest_path(splits_dir(), dataset))
    split = split_for(manifest, category, shots, data_root())
    return resolve_split(split, data_root())


def dataset_of(product: str) -> str | None:
    for ds in DATASETS.values():
        if product in dataset_categories(ds):
            return ds
    return None


# ------------------------------------------------------------------ images

def display_crop(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """The same resize + centre-crop the encoder sees, so maps line up."""
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize
    crop = Compose([Resize(size[0], interpolation=InterpolationMode.BICUBIC), CenterCrop(size)])
    return np.asarray(crop(image.convert("RGB")), dtype=np.float32) / 255.0


def mask_crop(mask_path, size: tuple[int, int]) -> np.ndarray | None:
    if mask_path is None:
        return None
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize
    crop = Compose([Resize(size[0], interpolation=InterpolationMode.NEAREST), CenterCrop(size)])
    return np.asarray(crop(Image.open(mask_path).convert("L"))) > 127


def overlay(image: np.ndarray, heat: np.ndarray, vmax: float) -> np.ndarray:
    """Red where the part looks unlike every good photo, fading to clear."""
    strength = np.clip(heat / max(vmax, 1e-6), 0, 1)[..., None]
    red = np.zeros_like(image)
    red[..., 0] = 1.0
    return np.clip(image * (1 - 0.6 * strength) + red * 0.6 * strength, 0, 1)


def draw_box(image: np.ndarray, row: int, col: int, grid: tuple[int, int],
             colour=(1.0, 0.8, 0.0), width: int = 2) -> np.ndarray:
    """Outline one patch cell, e.g. the most unusual one."""
    out = image.copy()
    h, w = out.shape[:2]
    ph, pw = h / grid[0], w / grid[1]
    y0, y1 = int(row * ph), min(h - 1, int((row + 1) * ph))
    x0, x1 = int(col * pw), min(w - 1, int((col + 1) * pw))
    out[y0:y0 + width, x0:x1] = colour
    out[y1 - width:y1, x0:x1] = colour
    out[y0:y1, x0:x0 + width] = colour
    out[y0:y1, x1 - width:x1] = colour
    return out


def mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    green = np.array([0.1, 0.75, 0.45])
    out[mask] = out[mask] * 0.45 + green * 0.55
    return out


# ------------------------------------------------------------------ charts

def line_chart(df: pd.DataFrame, x: str, y: str, color: str, colours: list[str],
               y_title: str | None = None, x_title: str | None = None, height: int = 300,
               y_zero: bool = False):
    order = list(dict.fromkeys(df[color]))
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:Q", title=x_title or x, axis=alt.Axis(tickMinStep=1)),
        y=alt.Y(f"{y}:Q", title=y_title or y, scale=alt.Scale(zero=y_zero)),
        color=alt.Color(f"{color}:N", sort=order, title=None,
                        scale=alt.Scale(domain=order, range=colours[:len(order)]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip(f"{color}:N"), alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q", format=".3f")],
    )
    return (base.mark_line(strokeWidth=2) + base.mark_point(size=60, filled=True)).properties(height=height)


def bar_chart(df: pd.DataFrame, x: str, y: str, color: str, colours: list[str],
              height: int = 300, y_title: str | None = None, domain=None):
    order = list(dict.fromkeys(df[color]))
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X(f"{x}:N", title=None, sort=list(dict.fromkeys(df[x]))),
        xOffset=alt.XOffset(f"{color}:N", sort=order),
        y=alt.Y(f"{y}:Q", title=y_title or y,
                scale=alt.Scale(domain=domain) if domain else alt.Scale(zero=False)),
        color=alt.Color(f"{color}:N", sort=order, title=None,
                        scale=alt.Scale(domain=order, range=colours[:len(order)]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip(f"{color}:N"), alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=".3f")],
    ).properties(height=height)


def show_data(df: pd.DataFrame, label: str = "Show the numbers") -> None:
    with st.expander(label):
        st.dataframe(df, hide_index=True, use_container_width=True)
