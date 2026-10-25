import numpy as np
import streamlit as st
from PIL import Image

from src.demo import app_shared as S

st.title("How it works")
st.markdown("Five steps, none of which involves training a model.")

example = None
if S.dataset_categories("mvtec_ad"):
    _, test = S.dataset_split("mvtec_ad", "bottle", 1)
    example = next((t for t in test if t.label == 1 and t.mask), None)

st.header("1 · Cut the photo into patches")
c1, c2 = st.columns([1, 2])
with c1:
    if example is not None:
        crop = S.display_crop(Image.open(example.image), (224, 224))
        grid = crop.copy()
        for k in range(0, 224, 16):
            grid[k, :] = (1.0, 1.0, 1.0)
            grid[:, k] = (1.0, 1.0, 1.0)
        st.image(grid, caption="224 x 224 px, a 14 x 14 grid of 16 px patches (ViT-B/16)",
                 use_container_width=True)
with c2:
    st.markdown(
        "The photo is resized to 224 x 224 pixels and split into a grid of patches. A **frozen "
        "CLIP vision model** - trained by OpenAI on 400 million internet images and never "
        "changed here - turns each patch into a vector of 768 numbers that describes what that "
        "patch looks like.\n\n"
        "- ViT-B/32: 7 x 7 = 49 patches, each 2% of the image (fast)\n"
        "- ViT-B/16: 14 x 14 = 196 patches, 0.5% each\n"
        "- ViT-L/14: 16 x 16 = 256 patches, 0.4% each (slow)\n\n"
        "A defect smaller than one patch gets averaged away. That single fact explains when the "
        "cheap model is enough and when it is not.\n\n"
        "**Release 1.1: multi-layer features.** The model is a stack of 12 blocks. The last one "
        "summarises the whole image and blurs where things are; earlier blocks keep local "
        "detail. So each patch is now described by blocks 6, 9 and 12 together (2,304 numbers "
        "instead of 768). On MVTec AD this raised detection from 0.904 to 0.938 and "
        "localisation from 0.936 to 0.957, for about 7% more time per part.")

st.header("2 · Remember the good patches")
st.markdown(
    "For a new product, the patch vectors of its 1-10 good photos are stored in a **memory owned "
    "by that product alone**. Ten photos of a bottle become 1,960 stored vectors (about 6 MB). "
    "Because no two products share anything, adding product number 16 cannot alter how products "
    "1-15 are scored. The project checks this **bit for bit** rather than assuming it.")

st.header("3 · Score each patch of a new part")
st.markdown("Each patch *q* of the part to inspect is compared with every stored good patch *m*:")
st.latex(r"\text{patch score}(q) \;=\; 1 - \max_{m \in \text{memory}} \cos(q, m)")
st.markdown(
    "A patch that looks like *some* good patch scores near 0; a patch unlike all of them scores "
    "high. The part's score is its **worst patch**, and all patch scores together, smoothed and "
    "enlarged, form the **heatmap**.")
if example is not None:
    c1, c2 = st.columns(2)
    c1.image(S.display_crop(Image.open(example.image), (224, 224)), caption="A broken bottle",
             use_container_width=True)
    c2.image(S.mask_overlay(S.display_crop(Image.open(example.image), (224, 224)),
                            S.mask_crop(example.mask, (224, 224))),
             caption="Where the benchmark says the defect is", use_container_width=True)

st.header("4 · Two refinements")
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown("**Multi-scale windows**")
    st.caption("The same comparison is repeated on 2 x 2 and 3 x 3 groups of neighbouring "
               "patches, and the three maps are averaged. A defect is judged over its "
               "neighbourhood, not one patch at a time. It costs two extra memory searches "
               "(about 1% of the time) and improved detection on all three datasets.")
with c2, st.container(border=True):
    st.markdown("**Text description**")
    st.caption("CLIP also understands text. The whole photo is compared with prompts like "
               "'a photo of a damaged bottle' plus the worker's own description, and that score "
               "is averaged 50/50 with the memory score. Helps most with only one photo; it "
               "only changes the score, not the heatmap.")

st.header("5 · Decide OK or DEFECTIVE")
st.markdown(
    "Each good photo is scored against the *other* good photos. The highest of those scores is "
    "the **suggested threshold**: the strictest line that rejects none of the known-good parts. "
    "A part scoring above it is flagged. With only 3 photos this line can be too cautious, so "
    "the app lets the operator move it, and shows where each part sits relative to it.")

st.header("Why these design choices")
st.markdown(
    "| choice | reason |\n|---|---|\n"
    "| frozen model, no training | deployment is a file copy; no GPU, no data scientist |\n"
    "| one memory per product | a new product can never break existing inspections |\n"
    "| 768-number patches (not CLIP's 512) | +0.02 AUROC, measured at 4x its error bar |\n"
    "| CPU only, 6 threads | the laptop's 6 fast cores; 14 threads gave timings 15x apart |\n"
    "| INT8 option | 4x smaller, 1.3-1.8x faster, about 0.01-0.02 AUROC cost |\n"
    "| results over 3 seeds, paired | one lucky draw of photos can swing AUROC by 0.1 |")
