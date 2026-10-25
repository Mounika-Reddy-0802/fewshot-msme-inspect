import streamlit as st
from PIL import Image

from src.demo import app_shared as S
from src.utils.paths import data_root

st.title("1 · Add a product")
st.markdown(
    "Give the system a few photos of **good** parts. Each photo is cut into a grid of patches, "
    "every patch becomes a vector of numbers, and those vectors are stored in a memory that "
    "belongs to this product alone. **No training happens** - adding a product takes seconds "
    "and cannot change any other product.")

insp = S.inspector()
text_ok = hasattr(insp.encoder, "encode_text") and S.ENGINES[S.engine_id()]["kind"] != "onnx"


def report(result, name):
    S.store_path().parent.mkdir(parents=True, exist_ok=True)
    insp.save(S.store_path())
    st.session_state.pop("batch", None)
    st.success(
        f"**{name}** added from {result.images} photo(s): {result.prototypes:,} patch vectors "
        f"stored ({result.bytes_added / 1e6:.1f} MB) in {result.seconds:.2f} s. Saved to disk.")


tab_data, tab_upload = st.tabs(["From a benchmark dataset", "Upload your own photos"])

with tab_data:
    if not data_root().exists():
        st.info("Set FEWSHOT_DATA_ROOT to the datasets folder to load benchmark categories.")
    else:
        avail = {k: v for k, v in S.DATASETS.items() if S.dataset_categories(v)}
        if not avail:
            st.info("No benchmark categories found under FEWSHOT_DATA_ROOT.")
        else:
            c1, c2, c3 = st.columns([1, 1, 1])
            ds_name = c1.selectbox("Dataset", list(avail))
            ds = avail[ds_name]
            category = c2.selectbox("Category", S.dataset_categories(ds))
            shots = c3.selectbox("Good photos", [1, 5, 10], index=1,
                                 help="Taken from the committed seed-42 split, exactly as in the benchmark.")
            desc = st.text_input("What does a defect look like? (optional)", key="ds_desc",
                                 placeholder="e.g. a broken or cracked rim", disabled=not text_ok,
                                 help=None if text_ok else "Not available with the INT8 model.")
            support, _ = S.dataset_split(ds, category, shots)
            st.image([str(p) for p in support], width=110,
                     caption=[p.name for p in support])
            if st.button(f"Add {category}", type="primary"):
                with st.spinner("Encoding the good photos ..."):
                    result = insp.add_product(category, [Image.open(p) for p in support],
                                              description=desc or None)
                report(result, category)

with tab_upload:
    name = st.text_input("Product name", placeholder="e.g. bracket-A")
    files = st.file_uploader("Photos of good parts (1-10)", type=["png", "jpg", "jpeg", "bmp"],
                             accept_multiple_files=True)
    desc_u = st.text_input("What does a defect look like? (optional)", key="up_desc",
                           placeholder="e.g. a scratch, a missing screw", disabled=not text_ok)
    if files:
        st.image([Image.open(f) for f in files], width=110)
    st.caption("Use at least 3 photos (5 or more is better): the suggested threshold is computed "
               "from how different your good photos are from each other.")
    if st.button("Add product", type="primary", disabled=not (name and files)):
        with st.spinner("Encoding the good photos ..."):
            result = insp.add_product(name, [Image.open(f) for f in files],
                                      description=desc_u or None)
        report(result, name)

if insp.products:
    st.subheader("Products in memory")
    rows = [{"product": p, "good photos": len(insp._support.get(p, [])),
             "patch vectors": insp.memory.size(p),
             "MB": round(insp.memory.size(p) * insp.memory.dim * 4 / 1e6, 1),
             "defect description": insp.descriptions.get(p, "")} for p in insp.products]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.page_link("views/inspect_part.py", label="Next: inspect a part", icon=":material/arrow_forward:")
