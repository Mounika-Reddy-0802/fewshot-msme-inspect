import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from src.demo import app_shared as S

st.title("3 · Check a batch")
st.markdown("Inspect many parts at once, the way a production run would be checked. On a "
            "benchmark test set the true answers are known, so the app measures its own "
            "accuracy live.")
insp = S.inspector()
if not insp.products:
    st.info("No products yet. Add one first.")
    st.page_link("views/add_product.py", label="Add a product", icon=":material/add_a_photo:")
    st.stop()

product = st.selectbox("Product", insp.products)
ds = S.dataset_of(product)
sources = (["Its whole benchmark test set"] if ds else []) + ["Upload several photos"]
source = st.radio("Parts", sources, horizontal=True)

items = []                      # (name, PIL image, label or None)
if source == "Upload several photos":
    files = st.file_uploader("Photos of parts", type=["png", "jpg", "jpeg", "bmp"],
                             accept_multiple_files=True)
    items = [(f.name, Image.open(f), None) for f in files or []]
else:
    _, test = S.dataset_split(ds, product, 1)
    items = [(f"{s.defect_type}/{s.image.name}", s.image, s.label) for s in test]
    st.caption(f"{len(items)} test images: {sum(1 for i in items if i[2] == 0)} good, "
               f"{sum(1 for i in items if i[2] == 1)} defective.")

key = (S.engine_id(), insp.scales, product, len(insp._support.get(product, [])), source, len(items))
cache = st.session_state.setdefault("batch", {})
if items and st.button(f"Inspect {len(items)} parts", type="primary"):
    bar = st.progress(0.0, text="Inspecting ...")
    images = [Image.open(x) if not isinstance(x, Image.Image) else x for _, x, _ in items]
    results = insp.inspect_many(product, images, progress=lambda d, t: bar.progress(d / t, text=f"{d}/{t} parts"))
    bar.empty()
    cache[key] = [{"part": n, "score": r.score, "truth": lab, "ms": r.encode_ms}
                  for (n, _, lab), r in zip(items, results)]

if key not in cache:
    st.stop()

df = pd.DataFrame(cache[key])
good = insp.good_photo_scores(product)
suggested = max(good) if good else float(df["score"].median())
mode = st.radio("Threshold", ["Suggested from the good photos", "Set by hand"], horizontal=True)
threshold = suggested if mode.startswith("Suggested") else st.slider(
    "Score above this means DEFECTIVE", 0.0, 1.0, float(round(suggested, 3)), 0.005)
df["verdict"] = df["score"].map(lambda s: "DEFECTIVE" if s > threshold else "OK")

labelled = df["truth"].notna().all()
m = st.columns(5)
m[0].metric("Parts", len(df))
m[1].metric("Flagged defective", int((df["verdict"] == "DEFECTIVE").sum()))
m[2].metric("ms per part", f"{df['ms'].mean():.0f}")
if labelled:
    from sklearn.metrics import roc_auc_score
    y = df["truth"].astype(int)
    pred = (df["verdict"] == "DEFECTIVE").astype(int)
    auroc = roc_auc_score(y, df["score"]) if y.nunique() == 2 else float("nan")
    acc = float((pred == y).mean())
    m[3].metric("AUROC", f"{auroc:.3f}", help="Threshold-free: 1.0 means every defective part "
                                              "scored above every good one.")
    m[4].metric("Accuracy at this threshold", f"{acc:.1%}")
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    st.markdown(f"**Caught {tp} of {tp + fn} defective parts; {fp} of {tn + fp} good parts "
                f"wrongly rejected.**")
    st.dataframe(pd.DataFrame({"": ["truly defective", "truly good"],
                               "flagged DEFECTIVE": [tp, fp], "passed OK": [fn, tn]}),
                 hide_index=True)

    plot = df.assign(kind=df["truth"].map({1: "defective (truth)", 0: "good (truth)"}))
    hist = alt.Chart(plot).mark_bar(opacity=0.8, cornerRadiusEnd=2).encode(
        x=alt.X("score:Q", bin=alt.Bin(maxbins=30), title="anomaly score"),
        y=alt.Y("count():Q", title="parts", stack=None),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(
            domain=["good (truth)", "defective (truth)"], range=[S.AQUA, S.ORANGE]),
            legend=alt.Legend(orient="bottom")),
        tooltip=["kind:N", alt.Tooltip("count():Q", title="parts")])
    rule = alt.Chart(pd.DataFrame([{"t": threshold}])).mark_rule(
        color=S.INK, strokeDash=[6, 4], strokeWidth=2).encode(x="t:Q")
    st.altair_chart((hist + rule).properties(height=240), use_container_width=True)
    st.caption("Good parts (green) should sit left of the dashed threshold, defective parts "
               "(orange) right of it. Overlap is where mistakes happen.")

st.subheader("Every part")
show = df.drop(columns=["ms"]).copy()
if labelled:
    show["truth"] = show["truth"].map({1: "defective", 0: "good"})
    show["correct"] = (show["verdict"] == "DEFECTIVE") == (show["truth"] == "defective")
st.dataframe(show.sort_values("score", ascending=False), hide_index=True, use_container_width=True,
             column_config={"score": st.column_config.ProgressColumn(
                 "score", min_value=0.0, max_value=float(max(df["score"].max(), threshold)), format="%.3f")})

if labelled:
    wrong = show[~show["correct"]]
    if len(wrong):
        st.subheader(f"The {len(wrong)} mistakes")
        st.caption("Worth showing honestly: these are the defects too small or too subtle for "
                   "the patch grid, or good parts with unusual texture.")
        lookup = {n: x for n, x, _ in items}
        for chunk in range(0, min(len(wrong), 8), 4):
            cols = st.columns(4)
            for c, (_, r) in zip(cols, wrong.iloc[chunk:chunk + 4].iterrows()):
                c.image(Image.open(lookup[r["part"]]) if not isinstance(lookup[r["part"]], Image.Image)
                        else lookup[r["part"]], use_container_width=True,
                        caption=f"{r['part']} - {r['verdict']} ({r['score']:.3f})")
