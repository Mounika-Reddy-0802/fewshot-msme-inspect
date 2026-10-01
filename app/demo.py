"""Few-shot defect inspection, as a workshop would use it.

    streamlit run app/demo.py

1. Add a product from a few photos of good parts -- no defect photos, no training --
   optionally with a typed description of what a defect looks like.
2. Inspect a part: anomaly score, heatmap, pass/fail against a threshold.
3. Add another product and watch the first one's scores stay exactly the same.

With FEWSHOT_DATA_ROOT set, any MVTec AD or VisA category can be loaded in one
click from the committed seed-42 split, so the demo shows the benchmarked
method on the benchmarked data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.splits import load_manifest, manifest_path, resolve_split, split_for  # noqa: E402
from src.demo.inspector import Inspector  # noqa: E402
from src.utils.paths import data_root, repo_root  # noqa: E402

BACKBONES = {
    "ViT-B/32 -- fast, fine for large defects": "ViT-B-32-quickgelu",
    "ViT-B/16 -- slower, needed for small defects": "ViT-B-16-quickgelu",
}
DATASETS = {"MVTec AD": "mvtec_ad", "VisA": "visa"}


@st.cache_resource(show_spinner="Loading the frozen CLIP backbone ...")
def load_encoder(name: str):
    import torch

    from src.model.encoder import FrozenCLIPEncoder

    torch.set_num_threads(6)      # one per performance core; see docs section 7
    return FrozenCLIPEncoder(name, "openai", device="cpu", project_patches=False)


def display_crop(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """The same resize + centre-crop the encoder sees, so the heatmap lines up."""
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

    crop = Compose([Resize(size[0], interpolation=InterpolationMode.BICUBIC), CenterCrop(size)])
    return np.asarray(crop(image.convert("RGB")), dtype=np.float32) / 255.0


def overlay(image: np.ndarray, heat: np.ndarray, vmax: float) -> np.ndarray:
    """Red where the part looks unlike every good photo, fading to transparent."""
    strength = np.clip(heat / max(vmax, 1e-6), 0, 1)[..., None]
    red = np.zeros_like(image)
    red[..., 0] = 1.0
    return np.clip(image * (1 - 0.6 * strength) + red * 0.6 * strength, 0, 1)


@st.cache_data
def dataset_split(dataset: str, category: str, shots: int):
    manifest = load_manifest(manifest_path(repo_root() / "results" / "manifests" / "splits", dataset))
    split = split_for(manifest, category, shots, data_root())
    return resolve_split(split, data_root())


@st.cache_data
def dataset_categories(dataset: str) -> list[str]:
    path = manifest_path(repo_root() / "results" / "manifests" / "splits", dataset)
    return sorted(load_manifest(path)["categories"]) if path.exists() else []


st.set_page_config(page_title="Few-shot inspection", layout="wide")
st.title("Few-shot defect inspection")
st.caption("Frozen CLIP + append-only prototype memory. CPU only. Nothing is trained.")

with st.sidebar:
    label = st.radio("Backbone", list(BACKBONES))
    backbone = BACKBONES[label]
    st.caption("Rule of thumb from the benchmark: if the defects you care about "
               "are smaller than ~2% of the photo, use ViT-B/16.")

encoder = load_encoder(backbone)
# one inspector per backbone: a memory built from B/32 features cannot score B/16 ones
key = f"inspector::{backbone}"
if key not in st.session_state:
    st.session_state[key] = Inspector(encoder)
    st.session_state["history"] = {}
inspector: Inspector = st.session_state[key]

# ---------------------------------------------------------------- add a product
st.header("1. Add a product from good photos")
tab_upload, tab_dataset = st.tabs(["Upload photos", "Load from a dataset"])

with tab_upload:
    name = st.text_input("Product name", placeholder="e.g. bracket-A")
    files = st.file_uploader("Photos of good parts (1-10)", type=["png", "jpg", "jpeg", "bmp"],
                             accept_multiple_files=True)
    description = st.text_input("What does a defect look like? (optional)",
                                placeholder="e.g. a cracked rim, missing screw")
    if st.button("Add product", disabled=not (name and files)):
        result = inspector.add_product(name, [Image.open(f) for f in files],
                                       description=description or None)
        st.success(f"Added {result.product}: {result.images} photo(s), {result.prototypes:,} "
                   f"prototypes, {result.bytes_added / 1e6:.1f} MB, {result.seconds:.2f} s.")

with tab_dataset:
    if not data_root().exists():
        st.info("Set FEWSHOT_DATA_ROOT to load MVTec AD or VisA categories here.")
    else:
        c1, c2, c3 = st.columns(3)
        dataset = DATASETS[c1.selectbox("Dataset", list(DATASETS))]
        category = c2.selectbox("Category", dataset_categories(dataset))
        shots = c3.selectbox("Good photos", [1, 5, 10], index=1)
        ds_description = st.text_input("Defect description (optional)", key="ds_description",
                                       placeholder="e.g. a broken edge")
        if category and st.button(f"Add {category} ({shots}-shot, seed 42 split)"):
            support, _ = dataset_split(dataset, category, shots)
            result = inspector.add_product(category, [Image.open(p) for p in support],
                                           description=ds_description or None)
            st.success(f"Added {category}: {result.prototypes:,} prototypes, "
                       f"{result.bytes_added / 1e6:.1f} MB, {result.seconds:.2f} s.")

if inspector.products:
    rows = [{"product": p, "prototypes": inspector.memory.size(p),
             "MB": round(inspector.memory.size(p) * inspector.memory.dim * 4 / 1e6, 1),
             "description": inspector.descriptions.get(p, "")}
            for p in inspector.products]
    st.dataframe(rows, hide_index=True)

# ---------------------------------------------------------------- inspect a part
st.header("2. Inspect a part")
if not inspector.products:
    st.info("Add a product first.")
    st.stop()

product = st.selectbox("Product", inspector.products)
source = st.radio("Part to inspect", ["Upload", "Pick from the dataset's test set"],
                  horizontal=True)

image, truth, ident = None, None, None
if source == "Upload":
    upload = st.file_uploader("Photo of the part", type=["png", "jpg", "jpeg", "bmp"])
    if upload:
        image, ident = Image.open(upload), upload.name
else:
    found = None
    for ds in DATASETS.values():
        if product in dataset_categories(ds):
            found = ds
            break
    if not found:
        st.info(f"{product} is not a dataset category; upload a photo instead.")
    else:
        _, test = dataset_split(found, product, 1)
        options = {f"{s.defect_type} / {s.image.name}": s for s in test}
        pick = st.selectbox("Test image", list(options))
        image, truth = Image.open(options[pick].image), options[pick]
        ident = str(truth.image)

if image is not None:
    result = inspector.inspect(product, image)
    suggested = inspector.suggested_threshold(product)
    threshold = st.slider(
        "Threshold (score above this = defective)", 0.0, 0.6,
        value=float(round(suggested, 3)) if suggested is not None else 0.2, step=0.005,
        help="Suggested value: the worst score any of your good photos gets when "
             "held out from the others. Tune it on your own parts.",
    )
    if suggested is None:
        st.caption("Add at least two good photos to get a suggested threshold.")

    defective = result.score > threshold
    crop = display_crop(image, encoder.image_size)
    heat = result.heatmap.numpy()

    left, right = st.columns(2)
    left.image(crop, caption="What the model sees", use_container_width=True)
    right.image(overlay(crop, heat, vmax=max(threshold * 1.5, float(heat.max()))),
                caption="Anomaly heatmap", use_container_width=True)

    verdict = "DEFECTIVE" if defective else "OK"
    (st.error if defective else st.success)(
        f"{verdict} -- score {result.score:.3f} vs threshold {threshold:.3f}")
    if truth is not None:
        st.caption(f"Ground truth: {truth.defect_type} "
                   f"({'defective' if truth.label else 'good'}).")
    if result.text_score is not None:
        st.caption(f"Score = 0.5 x memory {result.memory_score:.3f} + 0.5 x description "
                   f"{result.text_score:.3f}. The heatmap is the memory's alone.")
    st.caption(f"Encode {result.encode_ms:.0f} ms + memory search {result.search_ms:.1f} ms, CPU.")

    # append-only, shown rather than claimed: re-inspecting the same part after
    # adding other products must give a bit-identical score
    history = st.session_state["history"]
    fingerprint = (product, ident)
    seen = history.get(fingerprint)
    if seen is not None and seen[1] != len(inspector.products):
        same = seen[0] == result.score
        st.info(f"Scored {seen[0]:.6f} when memory held {seen[1]} product(s); now "
                f"{result.score:.6f} with {len(inspector.products)}. "
                + ("Bit-identical." if same else "Changed -- this should not happen."))
    history[fingerprint] = (result.score, len(inspector.products))
