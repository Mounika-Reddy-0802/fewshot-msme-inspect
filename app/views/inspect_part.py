import random

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from src.demo import app_shared as S

st.title("2 · Inspect a part")
insp = S.inspector()
if not insp.products:
    st.info("No products yet. Add one first.")
    st.page_link("views/add_product.py", label="Add a product", icon=":material/add_a_photo:")
    st.stop()

c1, c2 = st.columns([1, 2])
product = c1.selectbox("Product", insp.products)
ds = S.dataset_of(product)
sources = (["Pick from the benchmark test set"] if ds else []) + ["Upload a photo"]
source = c2.radio("Part to inspect", sources, horizontal=True)

image, truth, ident = None, None, None
if source == "Upload a photo":
    upload = st.file_uploader("Photo of the part", type=["png", "jpg", "jpeg", "bmp"])
    if upload:
        image, ident = Image.open(upload), upload.name
else:
    _, test = S.dataset_split(ds, product, 1)
    options = {f"{s.defect_type} / {s.image.name}": s for s in test}
    keys = list(options)
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Random defective part"):
        st.session_state["pick"] = random.choice([k for k in keys if options[k].label == 1])
    if b2.button("Random good part"):
        st.session_state["pick"] = random.choice([k for k in keys if options[k].label == 0])
    default = st.session_state.get("pick", keys[0])
    pick = st.selectbox("Test image (defect type / file)", keys,
                        index=keys.index(default) if default in keys else 0)
    truth = options[pick]
    image, ident = Image.open(truth.image), str(truth.image)

if image is None:
    st.stop()

with st.spinner("Inspecting ..."):
    result = insp.inspect(product, image)
    good = insp.good_photo_scores(product)
suggested = max(good) if good else None

mode = st.radio("Threshold", ["Suggested from the good photos", "Set by hand"], horizontal=True,
                help="Suggested = the highest score any good photo gets when compared with the "
                     "other good photos, i.e. the lowest threshold that would reject none of them.")
if mode == "Set by hand" or suggested is None:
    threshold = st.slider("Score above this means DEFECTIVE", 0.0, 1.0,
                          float(round(suggested, 3)) if suggested else 0.25, 0.005)
    if suggested is None:
        st.caption("Add at least two good photos to get a suggested threshold.")
else:
    threshold = suggested

defective = result.score > threshold
verdict = "DEFECTIVE" if defective else "OK"
(st.error if defective else st.success)(
    f"### {verdict}\nscore **{result.score:.3f}** {'>' if defective else '<='} threshold {threshold:.3f}")
if truth is not None:
    right = (truth.label == 1) == defective
    st.caption(f"Ground truth: **{'defective' if truth.label else 'good'}** ({truth.defect_type}) - "
               f"the verdict is {'correct' if right else 'wrong'}.")

enc = insp.encoder
crop = S.display_crop(image, enc.image_size)
heat = result.heatmap.numpy()
grid = enc.grid_size
row, col = result.worst_patch
boxed = S.draw_box(crop, row, col, grid)
mask = S.mask_crop(truth.mask, enc.image_size) if truth is not None and truth.mask else None

cols = st.columns(3 if mask is not None else 2)
cols[0].image(boxed, caption="What the model sees (yellow box = most unusual patch)",
              use_container_width=True)
cols[1].image(S.overlay(crop, heat, vmax=max(float(heat.max()), threshold)),
              caption="Heatmap: red = unlike every good photo", use_container_width=True)
if mask is not None:
    cols[2].image(S.mask_overlay(crop, mask), caption="Real defect (benchmark annotation, green)",
                  use_container_width=True)

st.subheader("Where this part sits")
points = pd.DataFrame([{"score": s, "what": "good photo (held out)"} for s in good]
                      + [{"score": result.score, "what": "this part"}])
dots = alt.Chart(points).mark_point(size=120, filled=True).encode(
    x=alt.X("score:Q", title="anomaly score (higher = less like the good photos)",
            scale=alt.Scale(zero=False, padding=12)),
    color=alt.Color("what:N", title=None, scale=alt.Scale(
        domain=["good photo (held out)", "this part"], range=[S.AQUA, S.BLUE]),
        legend=alt.Legend(orient="bottom")),
    shape=alt.Shape("what:N", scale=alt.Scale(domain=["good photo (held out)", "this part"],
                                              range=["circle", "diamond"]), legend=None),
    tooltip=[alt.Tooltip("what:N"), alt.Tooltip("score:Q", format=".3f")])
rule = alt.Chart(pd.DataFrame([{"t": threshold}])).mark_rule(
    color=S.ORANGE, strokeDash=[6, 4], strokeWidth=2).encode(x="t:Q", tooltip=[alt.Tooltip("t:Q", title="threshold", format=".3f")])
st.altair_chart((rule + dots).properties(height=110), use_container_width=True)
st.caption("Orange dashed line = threshold. Each green dot is one good photo scored against the "
           "other good photos; a defective part should land clearly to the right of all of them.")

with st.expander("How this verdict was reached", expanded=True):
    n_patches = grid[0] * grid[1]
    stored = insp.memory.size(product)
    steps = [
        f"The photo was resized to {enc.image_size[0]} px and cut into a {grid[0]} x {grid[1]} grid: "
        f"**{n_patches} patches**, each turned into a {enc.embed_dim}-number vector by the frozen model.",
        f"Every patch was compared with all **{stored:,} stored patches** of good *{product}* "
        f"({len(insp._support.get(product, []))} photos). A patch's score is 1 minus its similarity "
        f"to the closest good patch.",
    ]
    if insp.scales != (1,):
        steps.append("The same comparison was repeated for 2 x 2 and 3 x 3 groups of patches, and "
                     "the three maps were averaged (multi-scale windows).")
    steps.append(f"The most unusual patch is at row {row + 1}, column {col + 1} (yellow box). Its "
                 f"score is the part's memory score: **{result.memory_score:.3f}**.")
    if result.text_score is not None:
        steps.append(f"Your description *\"{insp.descriptions.get(product)}\"* was compared with "
                     f"the whole photo: {result.text_score:.3f}. Final score = 0.5 x "
                     f"{result.memory_score:.3f} + 0.5 x {result.text_score:.3f} = "
                     f"**{result.score:.3f}**. (Text changes the score, not the heatmap.)")
    steps.append(f"{result.score:.3f} {'is above' if defective else 'is not above'} the threshold "
                 f"{threshold:.3f}, so the part is **{verdict}**.")
    for i, s in enumerate(steps, 1):
        st.markdown(f"{i}. {s}")
    st.caption(f"Time on this CPU: encode {result.encode_ms:.0f} ms + memory search "
               f"{result.search_ms:.1f} ms.")

# append-only, shown rather than claimed
history = st.session_state.setdefault("history", {})
fingerprint = (S.engine_id(), insp.scales, product, ident)
seen = history.get(fingerprint)
if seen is not None and seen[1] != len(insp.products):
    same = seen[0] == result.score
    st.info(f"**Append-only check:** this part scored {seen[0]:.6f} when memory held {seen[1]} "
            f"product(s); now {result.score:.6f} with {len(insp.products)}. "
            + ("Bit-identical: adding products changed nothing." if same else
               "Changed - this should not happen."))
history[fingerprint] = (result.score, len(insp.products))
st.page_link("views/batch.py", label="Next: check a whole batch", icon=":material/arrow_forward:")
