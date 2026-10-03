"""app.py - Streamlit front-end of the Crop Suitability Ranking & Agronomic Recommendation Decision System.

Run:   streamlit run app.py

Designed for low-resource use: the trained model ships as a 2 MB file (loads in ~1 s), heavy results are cached,
SHAP is imported only when the "Why?" tab is opened (and the app still works without it), no GPU, no internet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from cropds import rank as rk, whatif as wi
from cropds.data import (CONTROLLABLE, EXAMPLE_FIELDS, FEATURES, LABEL, SLIDERS, UNITS, field_series, load_profit_index)
from cropds.explain import Explainer, radar_figure
from cropds.model import load_or_train
from cropds.report import build_csv, build_pdf
from cropds.uncertainty import trust_report

st.set_page_config(page_title="Crop Decision System", page_icon="🌾", layout="wide")


# ----------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading model ...")
def get_model():
    return load_or_train()


@st.cache_resource
def get_explainer():
    return Explainer(get_model())


@st.cache_data(show_spinner="Analysing field ...", max_entries=64)
def analyse(values: tuple, irrigation: float, profit_items: tuple):
    """Everything that depends only on the field (not on the what-if sliders): trust table + decision matrix."""
    m = get_model()
    x = pd.Series(values, index=FEATURES).to_frame().T
    tbl, info = trust_report(m, x)
    D = rk.build_matrix(m, x, irrigation_mm=irrigation, profit_index=dict(profit_items))
    return tbl, info, D


@st.cache_data(show_spinner="Searching the cheapest changes ...", max_entries=32)
def whatif_advice(values: tuple, crops: tuple, target: float, season_items: tuple):
    m = get_model()
    f = pd.Series(values, index=FEATURES)
    return wi.recommend(m, f, list(crops), target=target, season_scenarios=dict(season_items) or None)


def pct_col(label, hi=100.0):
    return st.column_config.ProgressColumn(label, min_value=0.0, max_value=hi, format="%.0f")


# ----------------------------------------------------------------------------- sidebar: inputs
model = get_model()
_first = list(EXAMPLE_FIELDS)[0]
st.session_state.setdefault("example", _first)
st.session_state.setdefault("have_soil", True)
st.session_state.setdefault("have_ph", True)               # start on a meaningful example field
for f in FEATURES:                                           # initial slider values
    st.session_state.setdefault(f"in_{f}", float(EXAMPLE_FIELDS[_first][f]))


def load_example():
    name = st.session_state.example
    if name in EXAMPLE_FIELDS:
        for f in FEATURES:
            st.session_state[f"in_{f}"] = float(EXAMPLE_FIELDS[name][f])
        st.session_state.have_soil = True
        st.session_state.have_ph = True


with st.sidebar:
    st.title("🌾 Your field")
    st.selectbox("Start from an example", ["Custom"] + list(EXAMPLE_FIELDS), key="example", on_change=load_example)
    st.caption("Climate (needed)")
    for f in ["temperature", "humidity", "rainfall"]:
        lo, hi, _, step = SLIDERS[f]
        st.slider(f"{LABEL[f].capitalize()} {UNITS[f]}".strip(), float(lo), float(hi), step=float(step), key=f"in_{f}")
    irrigation = st.slider("Irrigation water available (mm per season)", 0.0, 200.0, 0.0, 5.0,
                           help="Added to rainfall when judging water fit.")
    st.caption("Soil (optional - untick if you have no report)")
    have_soil = st.checkbox("I have a soil test (N, P, K)", key="have_soil")
    for f in ["N", "P", "K"]:
        lo, hi, _, step = SLIDERS[f]
        st.slider(f"{LABEL[f].capitalize()} (relative level)", float(lo), float(hi), step=float(step), key=f"in_{f}",
                  disabled=not have_soil)
    have_ph = st.checkbox("I know my soil pH", key="have_ph")
    lo, hi, _, step = SLIDERS["ph"]
    st.slider("Soil pH", float(lo), float(hi), step=float(step), key="in_ph", disabled=not have_ph)

    st.divider()
    st.subheader("What matters to you?")
    choice = st.radio("Priority", list(rk.PRESETS) + ["custom"], format_func=lambda s: s.replace("_", " ").capitalize(),
                      help="Changes how crops are traded off in the priority ranking.")
    if choice == "custom":
        weights = {c: st.slider(c, 0.0, 1.0, float(rk.PRESETS["balanced"][c]), 0.05, key=f"w_{c}") for c in rk.CRITERIA}
        if sum(weights.values()) == 0:
            weights = dict(rk.PRESETS["balanced"])
    else:
        weights = rk.PRESETS[choice]
    st.caption("Profit data (optional)")
    up = st.file_uploader("CSV: crop, profit_index_0_to_1", type="csv", label_visibility="collapsed")
    demo_profit = st.checkbox("Use ILLUSTRATIVE demo profit values (not real prices)", value=False)

profit = {}
if up is not None:
    try:
        d = pd.read_csv(up).dropna(subset=["profit_index_0_to_1"]); profit = dict(zip(d["crop"], d["profit_index_0_to_1"].astype(float)))
    except Exception:
        st.sidebar.error("Could not read the profit CSV.")
elif demo_profit:
    profit = load_profit_index()

values = {f: float(st.session_state[f"in_{f}"]) for f in FEATURES}
if not have_soil:
    for f in ["N", "P", "K"]: values[f] = np.nan
if not have_ph:
    values["ph"] = np.nan
field = field_series(values)
vt = tuple(float(field[f]) for f in FEATURES)

tbl, info, Dm = analyse(vt, float(irrigation), tuple(sorted(profit.items())))
ranked = rk.rank(Dm, weights)

# ----------------------------------------------------------------------------- header
st.title("Crop Suitability Ranking & Agronomic Recommendation")
st.caption("Hybrid decision system: machine learning + agronomic rules  ·  ranks crops, says how sure it is, explains why, "
           "and shows what to change.")
(st.warning if (info["band"] == "LOW" or info["borderline"]) else st.success)(
    f"**{tbl.crop[0].capitalize()}** is the top match.  {info['message']}")
if not have_soil or not have_ph:
    st.info("Some soil inputs are missing - the system ignores them and uses a model trained on what you provided. "
            "Accuracy is lower than with a full soil test (see the evaluation report).")

tab_rank, tab_why, tab_if, tab_rep = st.tabs(["🏆 Ranking", "🔍 Why?", "🧪 What-if", "📄 Report"])

# ----------------------------------------------------------------------------- tab 1 : ranking
with tab_rank:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Top 5 by Suitability Index")
        show = tbl.assign(range=lambda t: t.SI_p10.round(0).astype(int).astype(str) + " - " + t.SI_p90.round(0).astype(int).astype(str))
        st.dataframe(show[["rank", "crop", "SI", "prob", "range"]], hide_index=True, width="stretch",
                     column_config={"rank": "Rank", "crop": "Crop", "SI": pct_col("Suitability (0-100)"),
                                    "prob": st.column_config.ProgressColumn("Model confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                                    "range": st.column_config.TextColumn("Range if inputs are off by ~10%")})
        st.subheader("Priority ranking")
        st.caption(f"Priority = **{choice.replace('_', ' ')}**. Crops that fail the agronomic gates are never shown.")
        pr = ranked.reset_index().rename(columns={"index": "crop"})
        cols = ["rank", "crop", "TOPSIS", "SI", "water", "risk"] + (["profit"] if Dm["profit"].notna().all() else [])
        st.dataframe(pr[cols], hide_index=True, width="stretch",
                     column_config={"TOPSIS": st.column_config.ProgressColumn("Priority score", min_value=0.0, max_value=1.0, format="%.2f"),
                                    "SI": st.column_config.NumberColumn("Suitability", format="%.0f"),
                                    "water": st.column_config.NumberColumn("Water fit", format="%.2f"),
                                    "risk": st.column_config.NumberColumn("Risk", format="%.2f"),
                                    "profit": st.column_config.NumberColumn("Profit idx", format="%.2f")})
        if len(Dm) > 1:
            sens = rk.weight_sensitivity(Dm, weights, n=400)
            st.caption(f"Stability: **{sens.index[0]}** stays #1 in {sens.iloc[0]:.0%} of random re-weightings of your priorities.")
        if not Dm["profit"].notna().all():
            st.caption("Profit is ignored because no profit data was provided (the dataset has no prices).")
    with c2:
        st.subheader("Fit to ideal ranges")
        fig = radar_figure(model, field, list(tbl.crop[:3]))
        st.pyplot(fig, clear_figure=True); plt.close(fig)

# ----------------------------------------------------------------------------- tab 2 : why
with tab_why:
    crop = st.selectbox("Explain which crop?", list(tbl.crop), index=0)
    with st.spinner("Computing explanation ..."):
        fig, r = get_explainer().figure(field, crop)
    st.markdown(r["text"])
    st.pyplot(fig, clear_figure=True); plt.close(fig)
    t = r["table"].copy()
    t["Ideal range"] = t.ideal.map(lambda x: f"{x[0]:g} - {x[1]:g}")
    t["Parameter"] = t.feature.map(lambda f: LABEL[f].capitalize())
    st.dataframe(t[["Parameter", "value", "Ideal range", "status"]].rename(columns={"value": "Your value", "status": "Status"}),
                 hide_index=True, width="stretch")
    st.caption("Left chart: what pushes the machine-learning model towards this crop (SHAP; green = towards). "
               "Right chart: how well each factor fits the crop's agronomic range.")

# ----------------------------------------------------------------------------- tab 3 : what-if
with tab_if:
    st.subheader("Move the sliders - the ranking updates live")
    known = {f: not np.isnan(field[f]) for f in CONTROLLABLE}
    a, b, c, d = st.columns(4)
    deltas = {
        "N": a.slider("Add nitrogen (kg/ha)", 0, 60, 0, 1, disabled=not known["N"]),
        "P": b.slider("Add phosphorus (kg/ha)", 0, 40, 0, 1, disabled=not known["P"]),
        "K": c.slider("Add potassium (kg/ha)", 0, 40, 0, 1, disabled=not known["K"]),
        "ph": d.slider("Change soil pH", -1.0, 1.0, 0.0, 0.1, disabled=not known["ph"]),
    }
    lr = wi.live_rerank(model, field, deltas)
    lr["move"] = lr.change.map(lambda v: f"▲ {v}" if v > 0 else f"▼ {-v}" if v < 0 else "–")
    st.dataframe(lr[["crop", "rank_after", "move", "SI_before", "SI_after", "delta_SI"]], hide_index=True, width="stretch",
                 column_config={"crop": "Crop", "rank_after": "New rank", "move": "Move",
                                "SI_before": st.column_config.NumberColumn("Before", format="%.0f"),
                                "SI_after": pct_col("After"),
                                "delta_SI": st.column_config.NumberColumn("Change", format="%+.1f")})
    st.caption("Nutrients can only be added, not removed. Climate cannot be changed by the farmer.")

    st.divider()
    st.subheader("Find the minimum change that works")
    target = st.slider("Raise suitability by (points)", 5, 30, 15, 5)
    with st.expander("Optional: compare another sowing window (you enter its climate)"):
        use_season = st.checkbox("Include an alternative sowing window", value=False)
        s_name = st.text_input("Name", "Later sowing")
        s1, s2, s3 = st.columns(3)
        s_t = s1.number_input("Temperature °C", 5.0, 45.0, float(np.nan_to_num(field["temperature"], nan=25.0)), 0.5)
        s_h = s2.number_input("Humidity %", 10.0, 100.0, float(np.nan_to_num(field["humidity"], nan=70.0)), 1.0)
        s_r = s3.number_input("Rainfall mm", 10.0, 300.0, float(np.nan_to_num(field["rainfall"], nan=100.0)), 5.0)
    season = ((s_name, tuple(sorted(dict(temperature=s_t, humidity=s_h, rainfall=s_r).items()))),) if use_season else ()
    if st.button("Find changes for my top 2 crops", type="primary"):
        st.session_state["advice_key"] = (vt, tuple(tbl.crop[:2]), float(target), season)
    key = st.session_state.get("advice_key")
    if key:
        res = whatif_advice(key[0], key[1], key[2], tuple((n, dict(v)) for n, v in key[3]) and tuple((n, tuple(v)) for n, v in key[3]))
        for crop, v in res.items():
            st.markdown(f"**{crop.capitalize()}** - suitability {v['base_SI']:.0f}, goal {v['goal']:.0f}")
            if v["reachable"]:
                for p in v["plans"]:
                    st.success(" ".join(p["advice"]) + f"  → suitability {p['new_SI']:.0f}" + ("" if p["robust"] else "  (fragile: small input changes may undo it)"))
            elif v["limiting"] is None:
                st.info("Already near the ceiling - every provided factor is ideal; further gains are marginal.")
            else:
                msg = f"Target not reachable with controllable changes. Limiting factor: **{LABEL[v['limiting']]}**" + \
                      ("." if v["limiting_controllable"] else " (climate - cannot be changed).")
                st.warning(msg)
                if v["best_effort"]:
                    st.caption("Best achievable: " + " ".join(v["best_effort"]["advice"]) + f" → suitability {v['best_effort']['new_SI']:.0f}")
        st.caption(wi.DISCLAIMER)

# ----------------------------------------------------------------------------- tab 4 : report
with tab_rep:
    st.subheader("Download a report")
    st.caption("PDF for printing / sharing, CSV for spreadsheets.")
    if st.button("Prepare report"):
        crop1 = tbl.crop[0]
        fig, r = get_explainer().figure(field, crop1)
        import io
        bio = io.BytesIO(); fig.savefig(bio, format="png", dpi=110); plt.close(fig)
        res = whatif_advice(vt, tuple(tbl.crop[:2]), 15.0, ())
        adv = {c: (v["plans"][0]["advice"] if v["reachable"] else []) for c, v in res.items()}
        rank_out = tbl.merge(ranked.reset_index().rename(columns={"index": "crop"})[["crop", "TOPSIS", "water", "risk"]], on="crop", how="left")
        st.session_state["pdf"] = build_pdf(field, tbl, info, ranked, choice, r["text"], adv, wi.DISCLAIMER, bio.getvalue())
        st.session_state["csv"] = build_csv(field, rank_out)
    if "pdf" in st.session_state:
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download PDF", st.session_state["pdf"], "crop_report.pdf", "application/pdf")
        d2.download_button("⬇️ Download CSV", st.session_state["csv"], "crop_report.csv", "text/csv")
