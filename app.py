
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from src.transformer_numpy import TransformerNumpyEngine, FEATURES, POLLUTANTS
from src.data_service import load_data, station_rows, feature_row, historical_quantiles, relative_risk

st.set_page_config(
    page_title="JAK-AIR AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Style ----------------
st.markdown("""
<style>
:root { --brand:#123b5d; --brand2:#0f6c7a; --soft:#f4f8fb; --ink:#14202b; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b2940,#123b5d); }
[data-testid="stSidebar"] * { color: white; }
.hero {
  border-radius: 18px; padding: 22px 26px; margin-bottom: 14px;
  background: linear-gradient(135deg,#0b2940,#0f6c7a);
  color: white; box-shadow: 0 8px 24px rgba(0,0,0,.10);
}
.hero h1 { margin:0; font-size:2.0rem; }
.hero p { margin:.4rem 0 0 0; opacity:.9; }
.card {
  background:white; border:1px solid #dfe8ef; border-radius:14px;
  padding:15px 17px; box-shadow:0 3px 12px rgba(0,0,0,.04);
}
.badge {display:inline-block;padding:4px 9px;border-radius:99px;background:#e7f5ef;color:#176b4c;font-weight:700;font-size:.82rem;}
.small-note {font-size:.84rem;color:#5d6b76;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_data(BASE/"data"/"AQI_Jakarta_2024_operational.csv")

@st.cache_resource
def get_engine():
    return TransformerNumpyEngine(BASE/"model"/"best_model_Transformer.weights.h5")

df = get_data()
engine = get_engine()
q = historical_quantiles(df)

# Header
st.markdown("""
<div class="hero">
  <div class="badge">TKT 6 PROTOTYPE • v0.6</div>
  <h1>JAK-AIR AI</h1>
  <p>Explainable Urban Air Quality Forecasting & Decision Support for Jakarta</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## JAK-AIR AI")
role = st.sidebar.selectbox("Mode pengguna", ["Executive", "DLH DKI Jakarta", "Dishub DKI Jakarta", "Validasi TKT 6"])
page = st.sidebar.radio("Navigasi", ["Dashboard", "Forecast & Monitoring", "Explainable AI", "Causal / Lag Analysis", "System & Evidence"])
stations = sorted(df["Station_Name"].dropna().unique().tolist())
station = st.sidebar.selectbox("Stasiun AQMS", stations)

sdf = station_rows(df, station)
dt_opts = sdf["datetime"].dropna().sort_values().tolist()
selected_dt = st.sidebar.selectbox(
    "Waktu demonstrasi",
    dt_opts,
    index=len(dt_opts)-1,
    format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y • %H:%M")
)
row = sdf[sdf["datetime"] == selected_dt].iloc[-1]
X = feature_row(row)
pred = engine.predict_frame(X).iloc[0].to_dict()

# helper
def risk_color(r):
    return {"Low":"#1f8a70","Moderate":"#d9a400","High":"#e76f51","Very High":"#b42318"}.get(r,"#567")

def metric_grid(title="Model estimate"):
    st.subheader(title)
    cols = st.columns(6)
    for col,p in zip(cols,POLLUTANTS):
        r = relative_risk(pred[p], q[p])
        with col:
            st.metric(p, f"{pred[p]:.1f}", r)
    st.caption("Risk = relative screening based on historical distribution in the prototype dataset; not an official ISPU classification.")

# ---------------- Dashboard ----------------
if page == "Dashboard":
    metric_grid("Ringkasan estimasi enam polutan")

    c1,c2 = st.columns([1.1,0.9])
    with c1:
        st.markdown("### Peta jaringan AQMS")
        smap = (
            df[["Station_Name","Latitude","Longitude"]]
            .drop_duplicates("Station_Name")
            .dropna()
            .rename(columns={"Latitude":"lat","Longitude":"lon"})
        )
        st.map(smap, latitude="lat", longitude="lon", size=110)

    with c2:
        st.markdown("### Kondisi input")
        a,b = st.columns(2)
        a.metric("Vehicle speed", f"{float(row['Speed']):.1f}")
        b.metric("Congestion", f"{float(row['Congestion Level']):.1f}")
        c,d = st.columns(2)
        c.metric("Temperature", f"{float(row['Temperature']):.1f} °C")
        d.metric("Humidity", f"{float(row['Humidity']):.1f} %")
        e,f = st.columns(2)
        e.metric("Wind speed", f"{float(row['Wind Speed']):.1f}")
        f.metric("Pressure", f"{float(row['Pressure']):.1f}")
        st.info(f"Demonstration record: **{station}**, {pd.Timestamp(selected_dt).strftime('%d %B %Y %H:%M')}")

    st.markdown("### Decision-support summary")
    high = [p for p in POLLUTANTS if relative_risk(pred[p],q[p]) in ("High","Very High")]
    if high:
        st.warning("Polutan dengan relative-risk tinggi pada estimasi model: **" + ", ".join(high) + "**.")
    else:
        st.success("Tidak ada output model yang berada pada kategori relative-risk tinggi untuk record demonstrasi ini.")

# ---------------- Forecast/monitor ----------------
elif page == "Forecast & Monitoring":
    metric_grid("Model output at selected target timestamp")
    st.info(
        "Model riset menggunakan meteorologi pada target timestamp serta fitur polutan/meteorologi/traffic lag-1. "
        "Karena notebook menyiapkan input sebagai 1 timestep, prototype ini tidak menyebutnya sebagai Transformer sequence 24 jam."
    )

    pollutant = st.selectbox("Polutan untuk dianalisis", POLLUTANTS)
    recent = sdf[sdf["datetime"] <= selected_dt].tail(168).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["datetime"], y=recent[pollutant], name="Observed", mode="lines"))
    fig.add_trace(go.Scatter(x=[selected_dt], y=[pred[pollutant]], name="Model estimate", mode="markers", marker_size=12))
    fig.update_layout(title=f"{pollutant}: observed history and selected model estimate", height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Input audit trail")
    display = X.T.reset_index()
    display.columns = ["Feature","Value"]
    st.dataframe(display, use_container_width=True, hide_index=True)

# ---------------- XAI ----------------
elif page == "Explainable AI":
    st.subheader("Global SHAP explanation from the research notebook")
    p = st.selectbox("Target pollutant", POLLUTANTS)
    shap_file = BASE/"data"/f"shap_importance_{p.replace('.','_')}.csv"
    imp = pd.read_csv(shap_file)
    top = imp.head(15).sort_values("Mean_Abs_SHAP")
    fig = px.bar(top, x="Mean_Abs_SHAP", y="Feature", orientation="h",
                 title=f"Top SHAP features — {p}")
    st.plotly_chart(fig, use_container_width=True)

    total = imp["Mean_Abs_SHAP"].sum()
    traffic = imp[imp["Feature"].isin(["Speed_lag_1","Congestion Level_lag_1"])]["Mean_Abs_SHAP"].sum()
    spatial = imp[imp["Feature"].isin(["x_km","y_km"])]["Mean_Abs_SHAP"].sum()
    auto = imp[imp["Feature"]==f"{p}_lag_1"]["Mean_Abs_SHAP"].sum()
    c1,c2,c3 = st.columns(3)
    c1.metric("Target lag contribution", f"{100*auto/total:.1f}%")
    c2.metric("Traffic contribution*", f"{100*traffic/total:.1f}%")
    c3.metric("Spatial contribution*", f"{100*spatial/total:.1f}%")
    st.caption("*Share is calculated within the displayed research SHAP table, not a causal effect.")
    st.warning("SHAP explains model use of features. It does not by itself establish physical causality.")

# ---------------- PCMCI ----------------
elif page == "Causal / Lag Analysis":
    st.subheader("PCMCI-derived lagged conditional relationships")
    pc = pd.read_csv(BASE/"data"/"PCMCI_AQI_Jakarta_2024_Global_Summary.csv")
    target = st.selectbox("Target", POLLUTANTS)
    sub = pc[pc["Target"]==target].sort_values(["Stations_Significant","Mean_p_value"], ascending=[False,True])

    if sub.empty:
        st.info("No displayed aggregated PCMCI relationship for this target in the notebook summary.")
    else:
        fig = px.bar(
            sub, x="Mean_Strength", y="Source", orientation="h",
            hover_data=["Lag_h","Stations_Significant","Mean_p_value"],
            title=f"Aggregated lag-1 PCMCI relationships — {target}"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sub.round(4), use_container_width=True, hide_index=True)
    st.caption(
        "Interpretation: statistically supported lagged conditional association in the research workflow. "
        "Do not automatically interpret each edge as direct physical causation."
    )

# ---------------- Evidence ----------------
elif page == "System & Evidence":
    st.subheader("TKT 6 readiness & evidence")
    model_ok = (BASE/"model"/"best_model_Transformer.weights.h5").exists()
    checks = [
        ["Trained Transformer weights packaged", model_ok],
        ["22 research features implemented", len(FEATURES)==22],
        ["Six pollutant outputs", len(pred)==6],
        ["Five AQMS represented", df["Station_Name"].nunique()==5],
        ["Traffic variables integrated", {"Speed","Congestion Level"}.issubset(df.columns)],
        ["Meteorological variables integrated", {"Temperature","Humidity","Wind Speed","Pressure"}.issubset(df.columns)],
        ["SHAP research outputs integrated", (BASE/"data"/"shap_importance_PM2_5.csv").exists()],
        ["PCMCI research summary integrated", (BASE/"data"/"PCMCI_AQI_Jakarta_2024_Global_Summary.csv").exists()],
        ["Functional inference successful", bool(np.isfinite(list(pred.values())).all())],
    ]
    ck = pd.DataFrame(checks, columns=["Evidence item","Passed"])
    st.dataframe(ck, use_container_width=True, hide_index=True)

    st.markdown("### Model validation")
    nbm = pd.read_csv(BASE/"data"/"notebook_reported_metrics.csv")
    st.markdown("**Metrics reported by the uploaded research notebook**")
    st.dataframe(nbm.round(4), use_container_width=True, hide_index=True)

    rt = pd.read_csv(BASE/"data"/"runtime_validation_summary.csv")
    st.markdown("**Independent runtime check using packaged H5 weights + pure NumPy inference**")
    st.dataframe(rt.round(4), use_container_width=True, hide_index=True)
    st.caption(
        "Small differences versus notebook-reported metrics can arise because the prototype reimplements "
        "Keras inference without TensorFlow. The uploaded H5 weights are used directly."
    )

    st.markdown("### Evidence required before claiming TKT 6 in the proposal")
    st.markdown("""
1. Run this prototype in a relevant Jakarta air-quality/traffic use-case.
2. Record screenshot/video of the demonstration.
3. Complete the functional-test report and UAT form in `docs/`.
4. Obtain validation/sign-off from a relevant domain user/expert (preferably DLH/Dishub).
5. Attach self-assessment TKT, KI evidence, and partner support letter.
""")

# ---------------- Dishub scenario panel on every suitable page ----------------
if role == "Dishub DKI Jakarta" and page in ["Dashboard","Forecast & Monitoring"]:
    st.divider()
    st.subheader("Traffic scenario simulator")
    st.caption("Model-based sensitivity only — not a guaranteed causal effect.")
    c1,c2 = st.columns(2)
    speed0 = float(row["Speed_lag_1"])
    cong0 = float(row["Congestion Level_lag_1"])
    speed = c1.slider("Scenario vehicle speed (lag-1)", 0.0, max(100.0, speed0*2), speed0, 1.0)
    cong = c2.slider("Scenario congestion (lag-1)", 0.0, max(300.0, cong0*2), cong0, 1.0)

    Xs = X.copy()
    Xs.loc[0,"Speed_lag_1"] = speed
    Xs.loc[0,"Congestion Level_lag_1"] = cong
    scen = engine.predict_frame(Xs).iloc[0]
    cmp = pd.DataFrame({
        "Pollutant":POLLUTANTS,
        "Baseline":[pred[p] for p in POLLUTANTS],
        "Scenario":[float(scen[p]) for p in POLLUTANTS],
    })
    cmp["Difference"] = cmp["Scenario"]-cmp["Baseline"]
    st.dataframe(cmp.round(2), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "JAK-AIR AI v0.6 • TKT-6 demonstrator based on the uploaded Jakarta 2024 dataset, "
    "Transformer weights, SHAP outputs and PCMCI research workflow. Decision support only."
)
