import streamlit as st

from src.demo import results as R

h = R.headline()

st.title("Few-shot defect inspection on a laptop CPU")
st.markdown(
    "A small workshop photographs a few **good** parts. From then on, every new part is "
    "checked against those photos: the system says *OK* or *DEFECTIVE* and highlights where "
    "the part looks wrong. **Nothing is trained**, no defect photos are needed, and it runs on "
    "the laptop the workshop already owns.")

st.subheader("The problem")
c1, c2, c3 = st.columns(3)
with c1, st.container(border=True):
    st.markdown("**No defect photos**")
    st.caption("A new production line has good parts and no catalogue of failures yet.")
with c2, st.container(border=True):
    st.markdown("**No GPU, no ML team**")
    st.caption("The only computer is an ordinary laptop.")
with c3, st.container(border=True):
    st.markdown("**Products keep changing**")
    st.caption("A new part must be added without retraining or disturbing the others.")

st.subheader("The idea in one picture")
st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#2a78d6",
        fontname="Helvetica", fontsize=11, margin="0.15,0.08"];
  edge [color="#52514e", arrowsize=0.7];
  good [label="1-10 photos of\\ngood parts"];
  enc  [label="Frozen CLIP model\\n(never trained)"];
  mem  [label="Memory of good patches\\n(one per product)"];
  part [label="New part"];
  cmp  [label="Compare every patch\\nwith the closest good one"];
  out  [label="Score + heatmap\\nOK / DEFECTIVE", fillcolor="#fff3e8", color="#eb6834"];
  good -> enc -> mem; part -> enc; mem -> cmp; enc -> cmp -> out;
}
""", use_container_width=True)

st.subheader("What the measurements show")
m1, m2, m3, m4 = st.columns(4)
m1.metric("MVTec AD, 10 photos", f"{h['mvtec_proposed']:.3f}" if h["mvtec_proposed"] else "-",
          help="Image AUROC of ViT-B/16 with windows and text, mean over 3 seeds. "
               "1.0 = perfect separation of good and defective parts.")
m2.metric("Steel surfaces (NEU-DET)", f"{h['neu_windows']:.3f}" if h["neu_windows"] else "-",
          help="Image AUROC, ViT-B/16 with windows, 10 photos.")
m3.metric("Fastest check", f"{h['fastest_ms']:.0f} ms" if h["fastest_ms"] else "-",
          help="ViT-B/32 INT8 per part on a cool laptop CPU (Intel i7-13700H).")
m4.metric("Electricity per year", f"Rs {h['rs_per_year_b16']:.0f}" if h["rs_per_year_b16"] else "-",
          help="ViT-B/16, 10,000 parts a day, 300 days, 45 W, Rs 8 per kWh.")
m5, m6, m7, m8 = st.columns(4)
m5.metric("VisA (tiny defects)", f"{h['visa_proposed']:.3f}" if h["visa_proposed"] else "-",
          help="Image AUROC, ViT-B/16 with windows and text, 10 photos.")
m6.metric("Best MVTec model", f"{h['mvtec_best']:.3f}" if h["mvtec_best"] else "-",
          help="ViT-L/14, 10 photos. Slower: about 400 ms per part.")
m7.metric("Add a product", f"{h['append_seconds']:.1f} s" if h["append_seconds"] else "-",
          help="From 10 photos, end to end: read, encode, store. No retraining.")
m8.metric("Other products after an add", "unchanged" if h["append_bit_identical"] else "-",
          help="Checked bit-for-bit on every seed: adding a product never changes another one's scores.")

st.subheader("Honest comparison with the standard method")
st.markdown(
    "Given **the same 10 photos**, PatchCore (the strongest classical method) and this system "
    "**detect defects equally well**; PatchCore **marks the exact defect location** a little "
    "better. What this system adds: a new product in seconds without touching the others, a "
    "typed description of the defect, and measured cost on a laptop. See *Results*.")

st.subheader("Showing it in three minutes")
st.markdown(
    "1. **Add a product** - load `bottle` from MVTec AD with 5 good photos, type "
    "*a broken or cracked rim*.\n"
    "2. **Inspect a part** - pick a `broken_large` image (DEFECTIVE, heatmap on the break), "
    "then a `good` one (OK).\n"
    "3. **Check a batch** - run the whole bottle test set: the app computes the accuracy live.\n"
    "4. **Add a second product** (e.g. `screw`), inspect the same bottle again: the score is "
    "*bit-identical*.")
st.caption("Capstone project by Mounika Reddy (modelling and efficiency) and Ashi Sharma "
           "(data, evaluation and dissemination). Release 1.1.")
