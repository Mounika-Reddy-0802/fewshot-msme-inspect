"""Few-shot defect inspection on a laptop CPU -- the project app.

    streamlit run app/demo.py

Project pages explain the work and show every result straight from the
committed records (no datasets needed). Live-demo pages run the method: add a
product from a few good photos, inspect a part, or check a whole batch.

Set FEWSHOT_DATA_ROOT to load MVTec AD / VisA / NEU-DET categories in one click,
and FEWSHOT_RESULTS_ROOT for where products are saved between sessions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.demo import app_shared as S  # noqa: E402

st.set_page_config(page_title="Few-shot defect inspection", page_icon=":material/search:",
                   layout="wide")

pages = {
    "Project": [
        st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page("views/how_it_works.py", title="How it works", icon=":material/account_tree:"),
        st.Page("views/results.py", title="Results", icon=":material/bar_chart:"),
        st.Page("views/about.py", title="About & reproduce", icon=":material/info:"),
    ],
    "Live demo": [
        st.Page("views/add_product.py", title="1 · Add a product", icon=":material/add_a_photo:"),
        st.Page("views/inspect_part.py", title="2 · Inspect a part", icon=":material/search:"),
        st.Page("views/batch.py", title="3 · Check a batch", icon=":material/fact_check:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("**Model for the live demo**")
    engines = S.available_engines()
    ids = list(engines)
    current = st.session_state.get("engine", "b32")
    st.session_state["engine"] = st.radio(
        "Model", ids, index=ids.index(current) if current in ids else 0,
        format_func=lambda k: engines[k]["label"], label_visibility="collapsed")
    st.caption(engines[S.engine_id()]["note"])
    st.session_state["windows"] = st.toggle(
        "Multi-scale windows", value=st.session_state.get("windows", True),
        help="Also compare 2x2 and 3x3 groups of patches. Raised detection AUROC on all three "
             "datasets in the benchmark at no extra model cost.")
    if S.has_inspector():
        insp = S.inspector()
        if insp.products:
            st.markdown(f"**Products in memory ({len(insp.products)})**")
            st.caption(", ".join(insp.products))
            if st.button("Forget all products", use_container_width=True):
                S.forget_products()
                st.rerun()
        loaded = st.session_state.get(f"loaded::{S.engine_id()}")
        if loaded:
            st.caption(f"Reopened from disk: {', '.join(loaded)}.")

nav.run()
