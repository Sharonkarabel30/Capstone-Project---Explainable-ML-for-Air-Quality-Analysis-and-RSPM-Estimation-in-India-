"""
India Air Quality — RSPM Prediction Dashboard
AAI-590 Capstone Project (MSc Applied Artificial Intelligence)

Interactive dashboard for exploring India's RSPM (PM10) monitoring data and the
predictions of the two final models: a tuned XGBoost model and a tuned
entity-embedding neural network.

Run locally:   streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
MODEL_DIR = BASE / "models"
DATA_DIR = BASE / "data"

NAVY = "#0F243E"
COBALT = "#0047AB"
AMBER = "#F5B942"
SKY = "#5B8DEF"
LIGHT = "#F3F6FB"
GRID = "#E6ECF5"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# India National Air Quality Index (NAQI) bands for 24-hour PM10 (µg/m³), CPCB
NAQI_BANDS = [
    ("Good", 0, 50, "#00B050"),
    ("Satisfactory", 51, 100, "#92D050"),
    ("Moderate", 101, 250, "#F2C500"),
    ("Poor", 251, 350, "#FF9900"),
    ("Very Poor", 351, 430, "#FF0000"),
    ("Severe", 431, 10_000, "#C00000"),
]
NAQI_ORDER = [b[0] for b in NAQI_BANDS]
NAQI_COLORS = {b[0]: b[3] for b in NAQI_BANDS}

# Approximate geographic centre of each state / union territory (for the map)
STATE_COORDS = {
    "Andhra Pradesh": (15.9, 79.7), "Arunachal Pradesh": (28.2, 94.7), "Assam": (26.2, 92.9),
    "Bihar": (25.1, 85.3), "Chandigarh": (30.73, 76.78), "Chhattisgarh": (21.3, 81.9),
    "Dadra & Nagar Haveli": (20.2, 73.0), "Daman & Diu": (20.4, 72.8), "Delhi": (28.7, 77.1),
    "Goa": (15.3, 74.1), "Gujarat": (22.3, 71.2), "Haryana": (29.1, 76.1),
    "Himachal Pradesh": (31.1, 77.2), "Jammu & Kashmir": (33.7, 75.5), "Jharkhand": (23.6, 85.3),
    "Karnataka": (15.3, 75.7), "Kerala": (10.5, 76.3), "Madhya Pradesh": (23.5, 78.7),
    "Maharashtra": (19.7, 75.7), "Manipur": (24.7, 93.9), "Meghalaya": (25.5, 91.4),
    "Mizoram": (23.2, 92.9), "Nagaland": (26.2, 94.6), "Odisha": (20.9, 85.1),
    "Puducherry": (11.9, 79.8), "Punjab": (31.1, 75.3), "Rajasthan": (27.0, 74.2),
    "Sikkim": (27.5, 88.5), "Tamil Nadu": (11.1, 78.7), "Telangana": (18.1, 79.0),
    "Tripura": (23.9, 91.9), "Uttar Pradesh": (26.8, 80.9), "Uttarakhand": (30.1, 79.0),
    "West Bengal": (23.0, 87.9),
}

FEATURE_GROUP_LABELS = {
    "no2": "NO₂ level", "so2": "SO₂ level", "year": "Year",
    "season": "Season (month)", "state": "State", "location": "City / station",
    "type": "Area type",
}

# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="India Air Quality · RSPM Prediction",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, li, label {{
    font-family: 'DM Sans', sans-serif;
}}
h1, h2, h3, h4 {{ font-family: 'Syne', sans-serif !important; color: {NAVY}; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}

.hero {{
    background: {NAVY}; color: white; border-radius: 14px;
    padding: 1.5rem 1.8rem; margin-bottom: 1.2rem;
    border-left: 8px solid {AMBER};
}}
.hero h1 {{ color: white !important; font-size: 1.9rem; margin: 0 0 .3rem 0; }}
.hero p {{ color: #D7E1F0; margin: 0; font-size: 1.0rem; }}

.kpi {{
    background: white; border: 1px solid {GRID}; border-top: 4px solid {COBALT};
    border-radius: 12px; padding: .9rem 1.1rem; height: 100%;
}}
.kpi .label {{ color: #5A6B85; font-size: .82rem; text-transform: uppercase; letter-spacing: .05em; }}
.kpi .value {{ font-family: 'IBM Plex Mono', monospace; color: {NAVY}; font-size: 1.7rem; font-weight: 600; }}
.kpi .sub {{ color: {COBALT}; font-size: .85rem; }}

.badge {{ display: inline-block; padding: .35rem .9rem; border-radius: 999px;
          font-weight: 700; color: white; font-size: 1rem; }}
.note {{ background: {LIGHT}; border-left: 4px solid {COBALT}; border-radius: 8px;
         padding: .8rem 1rem; color: {NAVY}; font-size: .93rem; margin: .4rem 0 1rem 0; }}

section[data-testid="stSidebar"] {{ background: {LIGHT}; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ color: {NAVY}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
.stTabs [aria-selected="true"] {{ color: {COBALT} !important; }}
</style>
""",
    unsafe_allow_html=True,
)

pio.templates["capstone"] = go.layout.Template(
    layout=dict(
        font=dict(family="DM Sans, sans-serif", color=NAVY, size=13),
        title=dict(font=dict(family="Syne, sans-serif", size=17, color=NAVY), x=0.01),
        colorway=[COBALT, AMBER, NAVY, SKY, "#9DB4E0", "#E07A5F", "#2A9D8F"],
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        margin=dict(l=10, r=10, t=55, b=10),
        hoverlabel=dict(font=dict(family="DM Sans, sans-serif")),
    )
)
pio.templates.default = "capstone"


def show_chart(fig, height=None):
    if height:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def show_table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def kpi(label, value, sub=""):
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def naqi_category(value):
    for name, low, high, _ in NAQI_BANDS:
        if value <= high:
            return name
    return "Severe"


# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading monitoring data…")
def load_data():
    df = pd.read_csv(DATA_DIR / "dashboard_data.csv.gz", parse_dates=["date"])
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["naqi"] = pd.cut(
        df["rspm"],
        bins=[-np.inf, 50, 100, 250, 350, 430, np.inf],
        labels=NAQI_ORDER,
    )
    return df


@st.cache_data
def load_table(name):
    path = DATA_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(MODEL_DIR / "xgboost_rspm.json")
    with open(MODEL_DIR / "nn_metadata.json") as f:
        meta = json.load(f)
    weights = dict(np.load(MODEL_DIR / "nn_weights.npz"))
    feature_names = list(preprocessor.get_feature_names_out())
    return {"pre": preprocessor, "xgb": xgb_model, "meta": meta, "w": weights, "features": feature_names}


def feature_frame(rows):
    """rows: list of dicts with state, location, type, year, month, so2, no2."""
    df = pd.DataFrame(rows)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["so2"] = pd.to_numeric(df["so2"], errors="coerce")
    df["no2"] = pd.to_numeric(df["no2"], errors="coerce")
    return df


def predict_xgb(models, frame):
    X = models["pre"].transform(frame[models["meta"]["features"]])
    return np.maximum(models["xgb"].predict(X), 0), X


def xgb_contributions(models, X):
    """Per-feature contributions (µg/m³) of the XGBoost prediction, grouped by original feature."""
    contrib = models["xgb"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
    names = models["features"] + ["bias"]
    grouped = {}
    for name, value in zip(names, contrib):
        if name == "bias":
            key = "bias"
        elif name.startswith("numeric__"):
            raw = name.replace("numeric__", "")
            key = "season" if raw.startswith("month") else raw
        else:
            raw = name.replace("categorical__", "")
            key = raw.split("_", 1)[0]
        grouped[key] = grouped.get(key, 0.0) + float(value)
    return grouped


def predict_nn(models, frame):
    meta, w = models["meta"], models["w"]
    num = pd.DataFrame(index=frame.index)
    for c in ["so2", "no2"]:
        num[f"{c}_missing"] = frame[c].isna().astype(float)
        num[f"{c}_log"] = np.log1p(frame[c].fillna(meta["nn_medians"][c]).clip(0, meta["nn_caps"][c]))
    num["year"] = frame["year"].astype(float)
    num["month_sin"] = frame["month_sin"]
    num["month_cos"] = frame["month_cos"]
    cols = meta["numeric_columns"]
    mean = pd.Series(meta["num_mean"])[cols]
    std = pd.Series(meta["num_std"])[cols]
    numeric = ((num[cols] - mean) / std).to_numpy(dtype=float)

    parts = [numeric]
    for col in meta["cat_cols"]:
        ids = frame[col].map(meta["cat_maps"][col]).fillna(0).astype(int).to_numpy()
        parts.append(w[f"emb_{col}"][ids])
    x = np.concatenate(parts, axis=1)

    n_hidden = meta["n_hidden_layers"]
    for i in range(n_hidden):
        x = x @ w[f"dense_{i}_W"] + w[f"dense_{i}_b"]
        x = (w[f"bn_{i}_gamma"] * (x - w[f"bn_{i}_mean"])
             / np.sqrt(w[f"bn_{i}_var"] + meta["bn_epsilons"][i]) + w[f"bn_{i}_beta"])
        x = np.maximum(x, 0)
    out = (x @ w[f"dense_{n_hidden}_W"] + w[f"dense_{n_hidden}_b"]).ravel()
    log_pred = out * meta["y_log_std"] + meta["y_log_mean"]
    return np.expm1(np.clip(log_pred, 0, meta["log_cap"]))


def metrics(actual, predicted):
    err = np.asarray(actual) - np.asarray(predicted)
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    r2 = 1 - np.sum(err ** 2) / np.sum((actual - np.mean(actual)) ** 2)
    return mae, rmse, r2


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
try:
    data = load_data()
except FileNotFoundError:
    st.error("`data/dashboard_data.csv.gz` was not found. Copy the `data` folder from "
             "`dashboard_assets.zip` (exported by the notebook) next to `app.py`.")
    st.stop()

try:
    models = load_models()
    models_ok = True
except Exception as exc:  # the data pages still work without the models
    models, models_ok = None, False
    model_error = str(exc)

# ---------------------------------------------------------------------------
# Sidebar: navigation and filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<h2 style='margin-bottom:0'>🌫️ RSPM India</h2>"
                f"<p style='color:{COBALT};margin-top:.2rem'>Air-quality analytics &amp; prediction</p>",
                unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Overview", "Explore the data", "Model performance", "Predict RSPM", "About the project"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Filters")
    st.caption("Apply to *Overview* and *Explore the data*.")
    year_min, year_max = int(data["year"].min()), int(data["year"].max())
    years = st.slider("Years", year_min, year_max, (year_min, year_max))
    all_states = sorted(data["state"].unique())
    states = st.multiselect("States", all_states, placeholder="All states")
    types = st.multiselect("Area type", sorted(data["type"].unique()), placeholder="All area types")

    st.markdown("---")
    st.caption("AAI-590 Capstone · MSc Applied Artificial Intelligence")

filtered = data[(data["year"] >= years[0]) & (data["year"] <= years[1])]
if states:
    filtered = filtered[filtered["state"].isin(states)]
if types:
    filtered = filtered[filtered["type"].isin(types)]


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
def page_overview():
    hero("India Air Quality — RSPM Monitoring & Prediction",
         "Respirable Suspended Particulate Matter (PM10) across India's national monitoring network, "
         f"{years[0]}–{years[1]}, with machine-learning and deep-learning predictions.")

    if filtered.empty:
        st.warning("No records match the current filters.")
        return

    above = (filtered["rspm"] > 100).mean() * 100
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Readings", f"{len(filtered):,}", "daily RSPM measurements")
    with c2:
        kpi("Cities / stations", f"{filtered['location'].nunique():,}", f"in {filtered['state'].nunique()} states")
    with c3:
        kpi("Mean RSPM", f"{filtered['rspm'].mean():.0f}", "µg/m³")
    with c4:
        kpi("Median RSPM", f"{filtered['rspm'].median():.0f}", "µg/m³")
    with c5:
        kpi("Above NAQI 'Satisfactory'", f"{above:.0f}%", "readings > 100 µg/m³")

    st.write("")
    left, right = st.columns([1.15, 1])

    with left:
        by_state = (filtered.groupby("state")
                    .agg(mean_rspm=("rspm", "mean"), readings=("rspm", "size"))
                    .reset_index())
        by_state["lat"] = by_state["state"].map(lambda s: STATE_COORDS.get(s, (None, None))[0])
        by_state["lon"] = by_state["state"].map(lambda s: STATE_COORDS.get(s, (None, None))[1])
        by_state = by_state.dropna(subset=["lat"])
        fig = px.scatter_geo(
            by_state, lat="lat", lon="lon", size="readings", color="mean_rspm",
            hover_name="state", size_max=38,
            color_continuous_scale=["#9DB4E0", COBALT, AMBER, "#C00000"],
            hover_data={"lat": False, "lon": False, "mean_rspm": ":.1f", "readings": ":,"},
            labels={"mean_rspm": "Mean RSPM", "readings": "Readings"},
            title="Mean RSPM by state (bubble size = number of readings)",
        )
        fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#B8C4D6",
                        showland=True, landcolor="#F7F9FC", showocean=True, oceancolor="#EAF1FB")
        show_chart(fig, 520)

    with right:
        cat_counts = filtered["naqi"].value_counts().reindex(NAQI_ORDER).fillna(0).reset_index()
        cat_counts.columns = ["Category", "Readings"]
        fig = px.pie(cat_counts, names="Category", values="Readings", hole=0.55,
                     color="Category", color_discrete_map=NAQI_COLORS,
                     category_orders={"Category": NAQI_ORDER},
                     title="Readings by NAQI category (PM10)")
        fig.update_traces(textposition="outside", textinfo="percent+label", sort=False)
        fig.update_layout(showlegend=False)
        show_chart(fig, 520)

    c1, c2 = st.columns(2)
    with c1:
        yearly = (filtered.groupby("year")["rspm"].agg(["mean", "median", "size"]).reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["mean"], name="Mean", mode="lines+markers",
                                 line=dict(color=COBALT, width=3)))
        fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["median"], name="Median", mode="lines+markers",
                                 line=dict(color=AMBER, width=3, dash="dot")))
        fig.update_layout(title="Yearly trend", yaxis_title="RSPM (µg/m³)", xaxis_title="Year",
                          legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
        show_chart(fig, 380)
    with c2:
        monthly = filtered.groupby("month")["rspm"].mean().reindex(range(1, 13)).reset_index()
        monthly["Month"] = MONTHS
        fig = px.bar(monthly, x="Month", y="rspm", color="rspm",
                     color_continuous_scale=["#9DB4E0", COBALT, AMBER],
                     title="Seasonal pattern — mean RSPM by month",
                     labels={"rspm": "RSPM (µg/m³)"})
        fig.update_layout(coloraxis_showscale=False)
        show_chart(fig, 380)

    note("<b>Key patterns.</b> Pollution is highest in the Indo-Gangetic Plain (Delhi, Uttar Pradesh, Punjab) "
         "and in winter (December–January), and lowest in the monsoon months (July–August), when rain washes "
         "particles out of the air. Use the filters on the left to focus on particular years, states or area types.")


# ===========================================================================
# PAGE 2 — EXPLORE
# ===========================================================================
def page_explore():
    hero("Explore the Data", "Interactive analysis of where, when and with what RSPM pollution occurs.")
    if filtered.empty:
        st.warning("No records match the current filters.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["📍 Geography", "📅 Seasonality", "🧪 Pollutants", "🏭 Area type"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            top_n = st.slider("Number of states", 5, 35, 15)
            s = (filtered.groupby("state")["rspm"].mean().sort_values(ascending=False)
                 .head(top_n).sort_values().reset_index())
            fig = px.bar(s, x="rspm", y="state", orientation="h", color="rspm",
                         color_continuous_scale=["#9DB4E0", COBALT, AMBER],
                         labels={"rspm": "Mean RSPM (µg/m³)", "state": ""},
                         title=f"Most polluted states (top {top_n})")
            fig.update_layout(coloraxis_showscale=False)
            show_chart(fig, 30 * top_n + 120)
        with c2:
            min_obs = st.slider("Minimum readings per city", 10, 1000, 100, step=10)
            loc = filtered.groupby(["location", "state"])["rspm"].agg(["mean", "size"]).reset_index()
            loc = (loc[loc["size"] >= min_obs].sort_values("mean", ascending=False).head(top_n)
                   .sort_values("mean"))
            fig = px.bar(loc, x="mean", y="location", orientation="h", color="state",
                         hover_data={"size": ":,"},
                         labels={"mean": "Mean RSPM (µg/m³)", "location": "", "size": "Readings"},
                         title=f"Most polluted cities (top {top_n})")
            show_chart(fig, 30 * top_n + 120)

    with tab2:
        heat = (filtered.pivot_table(index="year", columns="month", values="rspm", aggfunc="mean")
                .reindex(columns=range(1, 13)))
        heat.columns = MONTHS
        fig = px.imshow(heat, aspect="auto", color_continuous_scale=["#EAF1FB", COBALT, AMBER, "#C00000"],
                        labels=dict(color="Mean RSPM"), title="Mean RSPM by year and month")
        fig.update_yaxes(dtick=1)
        show_chart(fig, 460)

        default = [s for s in ["Delhi", "Uttar Pradesh", "Maharashtra", "Kerala"] if s in filtered["state"].unique()]
        compare = st.multiselect("Compare the seasonal profile of states",
                                 sorted(filtered["state"].unique()), default=default, max_selections=6)
        if compare:
            prof = (filtered[filtered["state"].isin(compare)]
                    .groupby(["state", "month"])["rspm"].mean().reset_index())
            prof["Month"] = prof["month"].map(lambda m: MONTHS[m - 1])
            fig = px.line(prof, x="Month", y="rspm", color="state", markers=True,
                          category_orders={"Month": MONTHS},
                          labels={"rspm": "Mean RSPM (µg/m³)", "state": "State"},
                          title="Seasonal profile by state")
            show_chart(fig, 420)

    with tab3:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            gas = st.radio("Gas", ["no2", "so2"], horizontal=True,
                           format_func=lambda g: "NO₂" if g == "no2" else "SO₂")
            sample = filtered.dropna(subset=[gas]).sample(min(8000, len(filtered)), random_state=1)
            fig = px.scatter(sample, x=gas, y="rspm", color="type", opacity=0.45,
                             labels={gas: f"{'NO₂' if gas == 'no2' else 'SO₂'} (µg/m³)",
                                     "rspm": "RSPM (µg/m³)", "type": "Area type"},
                             title=f"RSPM vs {'NO₂' if gas == 'no2' else 'SO₂'} (random sample of 8,000 readings)")
            fig.update_xaxes(range=[0, sample[gas].quantile(0.995)])
            fig.update_yaxes(range=[0, sample["rspm"].quantile(0.995)])
            show_chart(fig, 460)
        with c2:
            corr = filtered[["rspm", "no2", "so2"]].corr().round(2)
            corr.index = corr.columns = ["RSPM", "NO₂", "SO₂"]
            fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1,
                            color_continuous_scale=["#C00000", "white", COBALT],
                            title="Correlation between pollutants")
            show_chart(fig, 460)

        clip = filtered["rspm"].quantile(0.995)
        fig = px.histogram(filtered[filtered["rspm"] <= clip], x="rspm", nbins=80,
                           labels={"rspm": "RSPM (µg/m³)"},
                           title="Distribution of RSPM readings with NAQI category limits")
        fig.update_traces(marker_color=COBALT)
        for name, low, high, colour in NAQI_BANDS[:-1]:
            fig.add_vline(x=high, line_dash="dot", line_color=colour,
                          annotation_text=name, annotation_position="top")
        show_chart(fig, 400)

    with tab4:
        sample = filtered.sample(min(40000, len(filtered)), random_state=2)
        fig = px.box(sample, x="type", y="rspm", color="type", points=False,
                     labels={"type": "Area type", "rspm": "RSPM (µg/m³)"},
                     title="RSPM by area type")
        fig.update_yaxes(range=[0, sample["rspm"].quantile(0.99)])
        fig.update_layout(showlegend=False)
        show_chart(fig, 440)
        summary = (filtered.groupby("type")["rspm"]
                   .agg(Readings="size", Mean="mean", Median="median").round(1).reset_index()
                   .rename(columns={"type": "Area type"}))
        show_table(summary)


def prediction_analysis(sub, model_col, st_choice):
    mae, rmse, r2 = metrics(sub["rspm"].to_numpy(), sub[model_col].to_numpy())
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Readings", f"{len(sub):,}")
    with k2:
        kpi("MAE", f"{mae:.1f}", "µg/m³")
    with k3:
        kpi("RMSE", f"{rmse:.1f}", "µg/m³")
    with k4:
        kpi("R²", f"{r2:.3f}")
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        s = sub.sample(min(6000, len(sub)), random_state=3)
        lim = float(max(s["rspm"].max(), s[model_col].max()))
        fig = px.scatter(s, x="rspm", y=model_col, opacity=0.35, color_discrete_sequence=[COBALT],
                         hover_data=["state", "location", "date"],
                         labels={"rspm": "Actual RSPM", model_col: "Predicted RSPM"},
                         title="Actual vs predicted (sample of up to 6,000 readings)")
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect prediction",
                                 line=dict(color=AMBER, dash="dash", width=2)))
        fig.update_layout(legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
        show_chart(fig, 440)
    with c2:
        bands = pd.cut(sub["rspm"], [-np.inf, 50, 100, 150, 200, np.inf],
                       labels=["< 50", "50–100", "100–150", "150–200", "> 200"])
        err = (sub.assign(band=bands, resid=sub["rspm"] - sub[model_col])
               .groupby("band", observed=False)["resid"].agg(["mean", "size"]).reset_index())
        fig = px.bar(err, x="band", y="mean", color="mean",
                     color_continuous_scale=[COBALT, "#D7E1F0", AMBER], color_continuous_midpoint=0,
                     hover_data={"size": ":,"},
                     labels={"band": "Actual RSPM band (µg/m³)", "mean": "Mean residual",
                             "size": "Readings"},
                     title="Bias by pollution level (positive = under-predicted)")
        fig.add_hline(y=0, line_color=NAVY)
        fig.update_layout(coloraxis_showscale=False)
        show_chart(fig, 440)

    locs = sub["location"].value_counts()
    loc_choice = st.selectbox("Time series for a city", locs.index.tolist(),
                              help="Cities sorted by number of readings in the selected split.")
    ts = (sub[sub["location"] == loc_choice]
          .assign(period=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
          .groupby("period")[["rspm", "pred_xgb", "pred_nn"]].mean().reset_index())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["period"], y=ts["rspm"], name="Actual", mode="lines+markers",
                             line=dict(color=NAVY, width=3)))
    fig.add_trace(go.Scatter(x=ts["period"], y=ts["pred_xgb"], name="XGBoost", mode="lines",
                             line=dict(color=COBALT, width=2)))
    fig.add_trace(go.Scatter(x=ts["period"], y=ts["pred_nn"], name="Neural network", mode="lines",
                             line=dict(color=AMBER, width=2, dash="dot")))
    fig.update_layout(title=f"Monthly mean RSPM — {loc_choice}", yaxis_title="RSPM (µg/m³)",
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    show_chart(fig, 400)

    by_state = (sub.assign(abs_err=(sub["rspm"] - sub[model_col]).abs())
                .groupby("state")["abs_err"].agg(["mean", "size"]).reset_index())
    by_state = by_state[by_state["size"] >= 100].sort_values("mean")
    if not by_state.empty and st_choice == "All states":
        fig = px.bar(by_state, x="mean", y="state", orientation="h", color="mean",
                     color_continuous_scale=["#9DB4E0", COBALT, AMBER],
                     hover_data={"size": ":,"},
                     labels={"mean": "MAE (µg/m³)", "state": "", "size": "Readings"},
                     title="Error by state (states with at least 100 readings)")
        fig.update_layout(coloraxis_showscale=False)
        show_chart(fig, 26 * len(by_state) + 120)



# ===========================================================================
# PAGE 3 — MODEL PERFORMANCE
# ===========================================================================
def page_performance():
    hero("Model Performance",
         "Six models trained on 2003–2012, tuned on 2013–2014 and tested once on the untouched year 2015.")

    test_res = load_table("test_results.csv")
    val_res = load_table("validation_results.csv")

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Leaderboard", "🔍 Prediction analysis", "⚙️ Tuning", "💡 Explainability"])

    with tab1:
        if test_res is not None:
            best = test_res.sort_values("RMSE").iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                kpi("Best model (2015 test)", best["Model"].split(" (")[0], "lowest RMSE")
            with c2:
                kpi("Mean absolute error", f"{best['MAE']:.1f}", "µg/m³ on 2015")
            with c3:
                kpi("R²", f"{best['R2']:.3f}", "share of variation explained")
            st.write("")
            metric = st.radio("Metric", ["MAE", "RMSE", "R2"], horizontal=True,
                              format_func=lambda m: {"MAE": "MAE (lower is better)",
                                                     "RMSE": "RMSE (lower is better)",
                                                     "R2": "R² (higher is better)"}[m])
            frames = []
            if val_res is not None:
                frames.append(val_res.assign(Split="Validation 2013–14"))
            frames.append(test_res.assign(Split="Test 2015"))
            both = pd.concat(frames)
            order = test_res.sort_values(metric, ascending=(metric != "R2"))["Model"].tolist()
            both = both[both["Model"].isin(order)]
            fig = px.bar(both, x=metric, y="Model", color="Split", barmode="group", orientation="h",
                         category_orders={"Model": order},
                         color_discrete_sequence=["#9DB4E0", COBALT],
                         title=f"{metric} by model — validation vs test")
            fig.update_layout(legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
            show_chart(fig, 460)
            st.markdown("##### Test results (2015)")
            show_table(test_res)
            if val_res is not None:
                with st.expander("Validation results (2013–2014) — all models and variants"):
                    show_table(val_res)
        note("<b>Reading the leaderboard.</b> MAE is the average error in µg/m³. RMSE penalises large errors more. "
             "R² is the share of day-to-day variation the model explains. XGBoost and the entity-embedding neural "
             "network are practically tied, and both clearly beat the median baseline.")

    with tab2:
        holdout = data[data["split"] != "Train"].copy()
        c1, c2, c3 = st.columns(3)
        with c1:
            split = st.selectbox("Data split", ["Test", "Validation"],
                                 format_func=lambda s: "Test (2015)" if s == "Test" else "Validation (2013–14)")
        with c2:
            model_col = st.selectbox("Model", ["pred_xgb", "pred_nn"],
                                     format_func=lambda m: "XGBoost (tuned)" if m == "pred_xgb"
                                     else "Neural network (entity embedding)")
        with c3:
            st_choice = st.selectbox("State", ["All states"] + sorted(holdout["state"].unique()))
        sub = holdout[holdout["split"] == split]
        if st_choice != "All states":
            sub = sub[sub["state"] == st_choice]
        sub = sub.dropna(subset=[model_col])
        if sub.empty:
            st.info("No predictions for this selection.")
        else:
            prediction_analysis(sub, model_col, st_choice)

    with tab3:
        specs = [("tuning_ridge.csv", "Ridge Regression — regularisation strength (α)"),
                 ("tuning_xgboost.csv", "XGBoost — tree depth and regularisation"),
                 ("tuning_neural_network.csv", "Entity-embedding neural network — architecture")]
        for file, title in specs:
            t = load_table(file)
            if t is None:
                continue
            c1, c2 = st.columns([1.2, 1])
            with c1:
                fig = px.bar(t.sort_values("RMSE", ascending=False), x="RMSE", y="Model", orientation="h",
                             color="RMSE", color_continuous_scale=[COBALT, "#9DB4E0"],
                             title=title, labels={"Model": ""})
                fig.update_layout(coloraxis_showscale=False)
                show_chart(fig, 70 * len(t) + 120)
            with c2:
                st.write("")
                st.write("")
                show_table(t)
        note("Each configuration was trained on 2003–2012 and compared on 2013–2014. The 2015 test set was "
             "never used to choose settings.")

    with tab4:
        shap_df = load_table("shap_importance.csv")
        if shap_df is not None:
            def pretty(name):
                name = name.replace("numeric__", "").replace("categorical__", "")
                for key, label in [("no2", "NO₂"), ("so2", "SO₂"), ("month_sin", "Month (sine)"),
                                   ("month_cos", "Month (cosine)"), ("year", "Year")]:
                    if name == key:
                        return label
                for prefix, label in [("state_", "State: "), ("location_", "City: "), ("type_", "Area type: ")]:
                    if name.startswith(prefix):
                        return label + name[len(prefix):]
                return name
            top = shap_df.head(20).copy()
            top["Feature"] = top["Feature"].map(pretty)
            fig = px.bar(top.sort_values("Mean_Absolute_SHAP"), x="Mean_Absolute_SHAP", y="Feature",
                         orientation="h", color="Mean_Absolute_SHAP",
                         color_continuous_scale=["#9DB4E0", COBALT, AMBER],
                         labels={"Mean_Absolute_SHAP": "Mean |SHAP| (µg/m³)", "Feature": ""},
                         title="What drives the XGBoost predictions? (top 20 features, SHAP)")
            fig.update_layout(coloraxis_showscale=False)
            show_chart(fig, 620)
        note("<b>How to read this.</b> SHAP measures how many µg/m³ each feature moves a prediction, on average. "
             "NO₂ (a traffic and combustion marker) is the strongest driver, followed by location (Uttar Pradesh, "
             "Punjab), season and SO₂. You can see the explanation of an individual prediction on the "
             "<b>Predict RSPM</b> page.")


# ===========================================================================
# PAGE 4 — PREDICT
# ===========================================================================
def page_predict():
    hero("Predict RSPM", "Enter monitoring conditions and get live predictions from both final models, "
                         "with an explanation of what drives the result.")
    if not models_ok:
        st.error("The models could not be loaded, so live prediction is unavailable. "
                 f"Check that the `models` folder is next to `app.py`. Details: {model_error}")
        return

    meta = models["meta"]
    known_locations = set(meta["cat_maps"]["location"])

    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", sorted(data["state"].unique()),
                             index=sorted(data["state"].unique()).index("Delhi")
                             if "Delhi" in data["state"].unique() else 0)
        locs = sorted(data.loc[data["state"] == state, "location"].unique())
        location = st.selectbox("City / station", locs)
    with c2:
        area_types = [t for t in ["Residential", "Industrial", "Sensitive"] if t in meta["cat_maps"]["type"]] \
            or sorted(meta["cat_maps"]["type"])
        typical_type = data.loc[data["location"] == location, "type"].mode()
        default_type = typical_type.iloc[0] if not typical_type.empty and typical_type.iloc[0] in area_types else area_types[0]
        area = st.selectbox("Area type", area_types, index=area_types.index(default_type))
        month = st.selectbox("Month", list(range(1, 13)), index=0, format_func=lambda m: MONTHS[m - 1])
    with c3:
        year = st.slider("Year", 2003, 2020, 2015,
                         help="The models were trained on 2003–2012 and tested on 2015. "
                              "Years after 2015 are an extrapolation.")
        hist_loc = data[data["location"] == location]
        so2_default = float(np.nan_to_num(hist_loc["so2"].median(), nan=8.0))
        no2_default = float(np.nan_to_num(hist_loc["no2"].median(), nan=25.0))

    g1, g2 = st.columns(2)
    with g1:
        so2_missing = st.checkbox("SO₂ reading not available", value=False)
        so2 = st.slider("SO₂ (µg/m³)", 0.0, 120.0, min(round(so2_default, 1), 120.0), 0.5,
                        disabled=so2_missing, help="Default: this city's historical median.")
    with g2:
        no2_missing = st.checkbox("NO₂ reading not available", value=False)
        no2 = st.slider("NO₂ (µg/m³)", 0.0, 200.0, min(round(no2_default, 1), 200.0), 0.5,
                        disabled=no2_missing, help="Default: this city's historical median.")

    row = {"state": state, "location": location, "type": area, "year": year, "month": month,
           "so2": np.nan if so2_missing else so2, "no2": np.nan if no2_missing else no2}
    frame = feature_frame([row])
    pred_x, X = predict_xgb(models, frame)
    pred_n = predict_nn(models, frame)
    pred_x, pred_n = float(pred_x[0]), float(pred_n[0])
    ensemble = (pred_x + pred_n) / 2
    category = naqi_category(ensemble)

    if location not in known_locations:
        note(f"<b>{location}</b> does not appear in the 2003–2012 training data, so the models treat it as a "
             "new station: they rely on its state, area type, gas levels and season.")

    st.write("")
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1.2])
    with k1:
        kpi("XGBoost", f"{pred_x:.0f}", "µg/m³")
    with k2:
        kpi("Neural network", f"{pred_n:.0f}", "µg/m³")
    with k3:
        kpi("Average of both", f"{ensemble:.0f}", "µg/m³")
    with k4:
        st.markdown(
            f'<div class="kpi"><div class="label">NAQI category (PM10)</div>'
            f'<div style="margin-top:.45rem"><span class="badge" style="background:{NAQI_COLORS[category]};'
            f'color:{"#0F243E" if category in ("Satisfactory", "Moderate") else "white"}">{category}</span></div>'
            f'<div class="sub" style="margin-top:.4rem">based on the average prediction</div></div>',
            unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=ensemble,
            number=dict(suffix=" µg/m³", font=dict(family="IBM Plex Mono", size=34, color=NAVY)),
            title=dict(text="Predicted RSPM (average of both models)", font=dict(family="Syne", size=16)),
            gauge=dict(
                axis=dict(range=[0, 500], tickwidth=1),
                bar=dict(color=NAVY, thickness=0.25),
                steps=[dict(range=[low, min(high, 500)], color=colour) for _, low, high, colour in NAQI_BANDS],
            ),
        ))
        show_chart(fig, 340)
    with c2:
        contrib = xgb_contributions(models, X)
        bias = contrib.pop("bias")
        items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
        labels = ["Average prediction"] + [FEATURE_GROUP_LABELS.get(k, k) for k, _ in items] + ["XGBoost prediction"]
        values = [bias] + [v for _, v in items] + [0]
        fig = go.Figure(go.Waterfall(
            orientation="v", x=labels, y=values,
            measure=["absolute"] + ["relative"] * len(items) + ["total"],
            text=[f"{v:+.0f}" if i not in (0, len(values) - 1) else f"{(bias if i == 0 else pred_x):.0f}"
                  for i, v in enumerate(values)],
            textposition="outside",
            increasing=dict(marker=dict(color=AMBER)),
            decreasing=dict(marker=dict(color=COBALT)),
            totals=dict(marker=dict(color=NAVY)),
            connector=dict(line=dict(color="#B8C4D6")),
        ))
        fig.update_layout(title="Why this XGBoost prediction? (SHAP contributions, µg/m³)",
                          yaxis_title="RSPM (µg/m³)", showlegend=False)
        show_chart(fig, 340)

    # Seasonal profile for the chosen inputs
    rows = [dict(row, month=m) for m in range(1, 13)]
    prof = feature_frame(rows)
    prof_x, _ = predict_xgb(models, prof)
    prof_n = predict_nn(models, prof)
    hist = hist_loc.groupby("month")["rspm"].mean().reindex(range(1, 13))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=MONTHS, y=hist.values, name=f"Historical mean at {location}",
                         marker_color="#D7E1F0"))
    fig.add_trace(go.Scatter(x=MONTHS, y=prof_x, name="XGBoost", mode="lines+markers",
                             line=dict(color=COBALT, width=3)))
    fig.add_trace(go.Scatter(x=MONTHS, y=prof_n, name="Neural network", mode="lines+markers",
                             line=dict(color=AMBER, width=3, dash="dot")))
    fig.update_layout(title=f"Predicted seasonal profile for these conditions ({year})",
                      yaxis_title="RSPM (µg/m³)",
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    show_chart(fig, 400)

    note("<b>How the prediction works.</b> XGBoost combines hundreds of decision trees. The neural network uses "
         "learned embeddings for state, city and area type. The waterfall chart breaks the XGBoost prediction "
         "down into the contribution of each input: amber bars push the prediction up, blue bars pull it down. "
         "Both models tend to under-predict extreme pollution days, which depend on weather that is not "
         "in the data.")


# ===========================================================================
# PAGE 5 — ABOUT
# ===========================================================================
def page_about():
    hero("About the Project", "AAI-590 Capstone · Master of Science in Applied Artificial Intelligence")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("""
#### Objective
Predict the daily **RSPM (PM10)** concentration at Indian air-quality monitoring stations from the
co-measured gases (SO₂, NO₂), the station's state, city and area type, and the time of year.

#### Data
*India Air Quality Data* (Kaggle) — 435,742 records from the Central Pollution Control Board, 1987–2015.
After cleaning, 395,029 records with an RSPM reading (2003–2015) were used.

#### Method
1. **Data cleaning:** duplicates, missing targets and invalid dates removed; duplicate category labels merged.
2. **Exploratory data analysis:** geographic, seasonal and pollutant patterns.
3. **Chronological split:** train 2003–2012 · validate 2013–2014 · test 2015.
4. **Models:** median baseline, Ridge Regression, Random Forest, XGBoost, a feed-forward MLP and an
   entity-embedding neural network.
5. **Optimisation:** hyperparameter tuning of Ridge, XGBoost and the neural network on the validation years.
6. **Explainable AI:** SHAP analysis of the XGBoost model.
7. **Final evaluation** on the untouched 2015 data.
""")
    with c2:
        st.markdown("#### Final models")
        test_res = load_table("test_results.csv")
        if test_res is not None:
            show_table(test_res)
        st.markdown("""
#### Limitations
- Severe-pollution days are under-predicted; they depend on weather (wind, rain, temperature
  inversions) and local events that are not in the dataset.
- Stations opened after 2012 have no history of their own in the models.

#### NAQI categories (PM10, 24-hour)
Good ≤ 50 · Satisfactory 51–100 · Moderate 101–250 · Poor 251–350 · Very Poor 351–430 · Severe > 430 µg/m³
""")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
{
    "Overview": page_overview,
    "Explore the data": page_explore,
    "Model performance": page_performance,
    "Predict RSPM": page_predict,
    "About the project": page_about,
}[page]()
