import altair as alt
import pandas as pd
import streamlit as st

from src.demo import app_shared as S
from src.demo import results as R

st.title("Results")
st.markdown("Every number on this page is read from the committed result records "
            "(`results/manifests/*.json`) - the same files the paper cites. Means over three "
            "random draws of the good photos (seeds 42, 1, 2). **AUROC** = how well the score "
            "separates good from defective parts (1.0 perfect, 0.5 coin toss).")

ds_label = st.radio("Dataset", list(R.DATASETS.values()), horizontal=True)
ds = {v: k for k, v in R.DATASETS.items()}[ds_label or "MVTec AD"]
blurb = {
    "mvtec_ad": "15 products (bottles, screws, carpet, transistors...). Defects average 4.6% of the image.",
    "visa": "12 products incl. circuit boards and macaroni. Defects are tiny: 0.04-0.9% of the image.",
    "neu_det": "6 steel-surface textures; each takes a turn as 'normal'. Defects are large (~39%). "
               "No pixel masks, so detection only.",
    "mpdd": "6 metal parts (brackets, connectors, plates, tubes) photographed under varied light "
            "and orientation - the closest public stand-in for a shop floor.",
}[ds]
st.caption(blurb)

st.header("1 · More good photos help, and the model choice depends on defect size")
curves = R.shots_curves(ds)
if len(curves):
    curves = curves[~curves["model"].str.contains("INT8")]
else:
    st.info("No results recorded for this dataset yet.")
if len(curves):
    st.altair_chart(S.line_chart(curves, "shots", "image AUROC", "model", [S.BLUE, S.ORANGE, S.AQUA],
                                 x_title="good photos per product"), use_container_width=True)
    S.show_data(curves.pivot_table(index="model", columns="shots", values="image AUROC").round(3).reset_index())
st.markdown({
    "mvtec_ad": "On MVTec the cheap **ViT-B/32 equals ViT-B/16** at 10 photos (0.904 both): defects "
                "are large enough for its coarser 7x7 patch grid.",
    "visa": "On VisA **ViT-B/16 beats ViT-B/32 by 0.05-0.09**: the defects are smaller than one "
            "B/32 patch (2% of the image) and get averaged away.",
    "neu_det": "On steel the defects are large, so again **ViT-B/32 is as good as ViT-B/16**.",
}.get(ds, ""))

st.header("2 · What each idea adds (ViT-B/16)")
abl = R.ablation(ds)
if len(abl):
    order = [s for s in ["memory only", "+ windows", "+ text", "+ windows + text", "multi-layer",
                         "multi-layer + windows", "text only (zero-shot)"] if s in set(abl["scoring"])]
    abl = abl[abl["shots"].isin([1, 10])].assign(shots=lambda d: d["shots"].map(lambda k: f"{k} photo(s)"))
    chart = alt.Chart(abl).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("scoring:N", sort=order, title=None),
        x=alt.X("image AUROC:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("shots:N", title=None, scale=alt.Scale(range=[S.BLUE, S.ORANGE]),
                        legend=alt.Legend(orient="bottom")),
        yOffset="shots:N",
        tooltip=["scoring:N", "shots:N", alt.Tooltip("image AUROC:Q", format=".3f")],
    ).properties(height=46 * len(order))
    st.altair_chart(chart, use_container_width=True)
    S.show_data(abl.pivot_table(index="scoring", columns="shots", values="image AUROC").round(3).reset_index())
st.markdown({
    "mvtec_ad": "**Windows** (comparing 2x2 and 3x3 groups of patches) and **text** (the photo "
                "compared with 'a photo of a damaged bottle' prompts) each help; together they lift "
                "1-photo detection from 0.856 to 0.881. Text alone is much worse.",
    "visa": "Same pattern on VisA: 0.773 to 0.815 at one photo.",
    "neu_det": "Windows help most on textures (+0.049). Text **hurts** here, and should: the "
               "'normal' class is itself a defect type, so a prompt like 'a crazing with a defect' "
               "asks the wrong question.",
}.get(ds, "Multi-layer features with windows against the last block alone."))

if ds != "neu_det":
    st.header("3 · Against PatchCore given the same photos")
    vs = R.versus_patchcore(ds)
    if len(vs):
        for metric, title in (("image", "Detection (is the part defective?)"),
                              ("pixel", "Localisation (where is the defect?)")):
            sub = vs[(vs["metric"] == metric) & (vs["shots"] == 10)]
            st.markdown(f"**{title}, 10 photos**")
            st.altair_chart(alt.Chart(sub).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("model:N", sort=list(sub["model"]), title=None),
                x=alt.X("AUROC:Q", scale=alt.Scale(zero=False)),
                color=alt.condition(alt.datum.model == "PatchCore (same photos)",
                                    alt.value(S.ORANGE), alt.value(S.BLUE)),
                tooltip=["model:N", alt.Tooltip("AUROC:Q", format=".3f"),
                         alt.Tooltip("delta vs PatchCore:Q", format="+.3f"),
                         alt.Tooltip("x SE:Q", title="x paired SE")],
            ).properties(height=40 * len(sub)), use_container_width=True)
        S.show_data(vs.round(3))
        st.markdown("Blue = this project, orange = PatchCore. A difference counts only when it is "
                    "more than twice its paired standard error ('x SE' in the table). **Detection "
                    "is a tie** once windows are on; **PatchCore localises better** - its feature "
                    "map is twice as fine as CLIP B/16's.")

st.header("Accuracy against CPU cost")
fr = R.frontier(ds)
if len(fr):
    models = list(dict.fromkeys(fr["model"]))
    colours = [S.BLUE, S.ORANGE, S.AQUA, S.YELLOW, S.MAGENTA][:len(models)]
    st.altair_chart(alt.Chart(fr).mark_point(size=110, filled=True).encode(
        x=alt.X("ms per part:Q", title="CPU time per part (ms, cool laptop)"),
        y=alt.Y("image AUROC:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", title=None, scale=alt.Scale(domain=models, range=colours),
                        legend=alt.Legend(orient="bottom")),
        shape=alt.Shape("shots:O", title="photos"),
        tooltip=["model:N", "shots:O", alt.Tooltip("ms per part:Q", format=".0f"),
                 alt.Tooltip("image AUROC:Q", format=".3f")],
    ).properties(height=320), use_container_width=True)
    S.show_data(fr.round(3))
    st.caption("Up and to the left is better. INT8 = the same model compressed 4x; it is faster "
               "and loses about 0.01-0.02 AUROC.")

st.header("Cost on a laptop (all datasets)")
c1, c2 = st.columns([3, 2])
with c1:
    sus = R.sustained()
    if len(sus):
        st.markdown("**The laptop slows down under continuous use**")
        st.altair_chart(S.line_chart(sus, "minute", "ms per part", "model", [S.BLUE, S.ORANGE, S.AQUA],
                                     x_title="minutes of non-stop inspection", height=260),
                        use_container_width=True)
        st.caption("ViT-B/32 holds 40 ms for four minutes, then the CPU throttles to ~230 ms "
                   "(5.8x). Plan capacity on the slow figure.")
with c2:
    st.markdown("**Inspecting 10,000 parts a day**")
    cost = R.cost_table()
    st.dataframe(cost[["configuration", "ms per part (cool)", "Rs per year", "busy hours / day"]],
                 hide_index=True, use_container_width=True)
    st.caption("45 W CPU, Rs 8 per kWh, 300 days a year. Hardware cost: Rs 0 (existing laptop).")
S.show_data(cost, "Full cost table (RAM, sustained speed)")

st.header("Compression to INT8")
i8 = R.int8_table()
if len(i8):
    st.dataframe(i8.round(3), hide_index=True, use_container_width=True)
    st.caption("10-photo image AUROC. INT8 makes the model 4x smaller and 1.3-1.8x faster for "
               "0.012-0.019 AUROC, and never changes which model is the right choice.")

st.header("Adding a product never disturbs the others")
hold = R.record("incremental_holdout_10shot.json")
if hold:
    st.markdown(
        f"Twelve MVTec products were in memory; three new ones (*{', '.join(hold['held_out'])}*) "
        f"were added from 10 photos each, on three seeds.\n\n"
        f"- the 12 existing products scored **bit-identically** before and after: "
        f"**{'yes' if hold['no_regression_all_seeds'] else 'no'}**\n"
        f"- each added product scored exactly as if it had been there from the start: "
        f"**{'yes' if hold['appended_equals_scratch_all_seeds'] else 'no'}**\n"
        f"- time to add one: **{min(v['median'] for v in hold['append_seconds_end_to_end'].values()):.1f}-"
        f"{max(v['median'] for v in hold['append_seconds_end_to_end'].values()):.1f} s** "
        f"(reading and encoding the photos; storing them takes ~2 ms)")
