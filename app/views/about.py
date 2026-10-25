import subprocess

import streamlit as st

from src.utils.paths import repo_root

st.title("About & reproduce")


def version() -> str:
    try:
        out = subprocess.run(["git", "describe", "--tags", "--always"], cwd=repo_root(),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


st.markdown(f"**Version:** `{version()}`")

st.header("Team")
st.markdown(
    "| member | responsibility |\n|---|---|\n"
    "| Mounika Reddy (Member A) | model, prototype memory, windows and text scoring, ONNX/INT8 export, "
    "CPU profiling, PatchCore baseline, app |\n"
    "| Ashi Sharma (Member B) | datasets and loaders, seeded splits, NEU-DET protocol, split checks, "
    "licence log, independent rerun on a second machine, Colab reproduction |")

st.header("Data")
st.markdown(
    "| dataset | what | used for |\n|---|---|---|\n"
    "| MVTec AD | 15 industrial products, 5,354 images, pixel masks (CC BY-NC-SA 4.0) | main benchmark |\n"
    "| VisA | 12 products, ~10,800 images, tiny defects (CC BY 4.0) | small-defect benchmark |\n"
    "| NEU-DET | 6 steel-surface classes, 1,800 images (research use) | cross-domain test |\n\n"
    "Only download scripts are in the repository; no images are redistributed.")

st.header("Protocol")
st.markdown(
    "- 1, 5 or 10 good photos per product; the 1-photo set is inside the 5, which is inside the 10\n"
    "- the test images never change with the number of photos\n"
    "- three random draws (seeds 42, 1, 2), all committed in `results/manifests/splits/`\n"
    "- comparisons are paired on identical splits; a difference counts above twice its standard error\n"
    "- all timing on one laptop CPU (Intel i7-13700H), batch of one part, 6 threads")

st.header("Reproduce any number")
st.code("python scripts/reproduce.py --list        # one target per table and figure\n"
        "python scripts/reproduce.py mvtec         # e.g. the MVTec table\n"
        "pytest -q                                 # the test suite\n"
        "streamlit run app/demo.py                 # this app", language="bash")
st.markdown("Checked so far: a fresh install on a second laptop reproduces MVTec and VisA to three "
            "decimals; the Colab notebook reproduces the bottle numbers exactly; NEU-DET splits "
            "regenerate byte-identically on another machine.")

st.header("Limitations")
st.markdown(
    "- public benchmark photos, not a real shop floor\n"
    "- defects thinner than one patch (e.g. hairline scratches on screws) are still missed often\n"
    "- PatchCore, given the same photos, marks the exact defect location better\n"
    "- the suggested threshold is cautious with only three photos\n"
    "- WinCLIP and AnomalyCLIP are cited, not re-run")
