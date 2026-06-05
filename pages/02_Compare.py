import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import engineer_base_features, engineer_extended_features
from src.pipeline import impute_semantic


# Page config

ICON_DIR = Path(__file__).parent.parent / "icons" / "scale.svg"
with open(ICON_DIR) as f:
    svg_content = f.read()

svg_content = svg_content.replace('currentColor', '#c8a96e')
svg_content = svg_content.replace('#000000', '#c8a96e')
svg_content = svg_content.replace('black', '#c8a96e')


st.set_page_config(
    page_title="Compare",
    page_icon=svg_content,
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
    h1, h2, h3, h4 { font-family: 'DM Serif Display', serif !important; letter-spacing: -0.02em; }
    .stApp { background-color: #0f0f0f; color: #e8e3d9; }
    .section-divider { border: none; border-top: 1px solid #2a2a2a; margin: 24px 0; }
    .price-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #141414 100%);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 28px;
        text-align: center;
    }
    .price-card-a { border-top: 3px solid #c8a96e; }
    .price-card-b { border-top: 3px solid #8fb8c8; }
    .price-value-a {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #c8a96e;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .price-value-b {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #8fb8c8;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .price-label {
        font-size: 0.7rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-top: 6px;
    }
    .col-header-a {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #c8a96e;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    .col-header-b {
        font-family: 'DM Serif Display', serif;
        font-size: 1.3rem;
        color: #8fb8c8;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    .hint {
        font-size: 0.75rem;
        color: #555;
        margin-top: -12px;
        margin-bottom: 16px;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label {
        color: #999 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
</style>
""", unsafe_allow_html=True)


# Constants

SQM_PER_SQFT = 0.0929

NEIGHBOURHOOD_NAMES: dict[str, str] = {
    "Blmngtn": "Bloomington Heights",
    "Blueste":  "Bluestem",
    "BrDale":   "Briardale",
    "BrkSide":  "Brookside",
    "ClearCr":  "Clear Creek",
    "CollgCr":  "College Creek",
    "Crawfor":  "Crawford",
    "Edwards":  "Edwards",
    "Gilbert":  "Gilbert",
    "IDOTRR":   "Iowa DOT & Rail Road",
    "MeadowV":  "Meadow Village",
    "Mitchel":  "Mitchell",
    "NAmes":    "North Ames",
    "NoRidge":  "Northridge",
    "NPkVill":  "Northpark Villa",
    "NridgHt":  "Northridge Heights",
    "NWAmes":   "Northwest Ames",
    "OldTown":  "Old Town",
    "SWISU":    "South & West of Iowa State University",
    "Sawyer":   "Sawyer",
    "SawyerW":  "Sawyer West",
    "Somerst":  "Somerset",
    "StoneBr":  "Stone Brook",
    "Timber":   "Timberland",
    "Veenker":  "Veenker",
}


# Load bundle

@st.cache_resource
def load_bundle():
    path = Path(__file__).parent.parent / "models" / "model_blend.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


bundle = load_bundle()


# Neighbourhood options

@st.cache_data
def build_neighbourhood_options(_bundle) -> dict[str, str]:
    items = []
    for code in _bundle["neighbourhood_list"]:
        name   = NEIGHBOURHOOD_NAMES.get(code, code)
        median = _bundle["neighborhood_medians"].get(code, 0)
        items.append((f"{name}  (median ${median:,.0f})", code, median))
    items.sort(key=lambda x: x[2], reverse=True)
    return {label: code for label, code, _ in items}


neighbourhood_options = build_neighbourhood_options(bundle)


# Predict

def predict(inputs: dict) -> float:
    row = bundle["feature_defaults"].copy()
    row.update(inputs)
    df  = pd.DataFrame([row])
    df  = engineer_base_features(df)
    df  = engineer_extended_features(df, bundle["neighborhood_medians"])
    df  = impute_semantic(df)
    df  = df.drop(columns=["SalePrice", "SalePrice_log", "Id"], errors="ignore")
    X_proc = bundle["preprocessor"].transform(df)
    w_r = bundle["blend_weights"]["ridge"]
    w_x = bundle["blend_weights"]["xgb"]
    y_log = (
        w_r * bundle["model_ridge"].predict(X_proc) +
        w_x * bundle["model_xgb"].predict(X_proc)
    )
    return float(np.expm1(y_log)[0])


# Property inputs

def property_inputs(suffix: str, use_sqm: bool) -> dict:
    unit_label  = "m²" if use_sqm else "sqft"
    to_display  = (lambda v: round(v * SQM_PER_SQFT)) if use_sqm else (lambda v: v)
    to_internal = (lambda v: int(v / SQM_PER_SQFT)) if use_sqm else (lambda v: v)

    default_idx = 0 if suffix == "a" else 5

    neighbourhood_label = st.selectbox(
        "Neighbourhood",
        options=list(neighbourhood_options.keys()),
        index=default_idx,
        key=f"neighbourhood_{suffix}",
    )
    neighbourhood   = neighbourhood_options[neighbourhood_label]
    selected_median = bundle["neighborhood_medians"].get(neighbourhood, 0)
    st.markdown(
        f"<p class='hint'>Neighbourhood median: ${selected_median:,.0f}</p>",
        unsafe_allow_html=True,
    )

    overall_qual = st.slider(
        "Overall Quality (1-10)",
        min_value=1, max_value=10, value=6,
        key=f"qual_{suffix}",
        help="1 = Very Poor · 5 = Average · 10 = Excellent",
    )

    gr_liv_area = st.number_input(
        f"Above Ground Living Area ({unit_label})",
        min_value=to_display(300),
        max_value=to_display(6000),
        value=to_display(1500),
        step=to_display(50),
        key=f"living_{suffix}",
    )

    total_bsmt_sf = st.number_input(
        f"Basement Area ({unit_label}) — 0 if none",
        min_value=0,
        max_value=to_display(3000),
        value=to_display(800),
        step=to_display(50),
        key=f"bsmt_{suffix}",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        year_built     = st.number_input(
            "Year Built",
            min_value=1870, max_value=2010,
            value=1990, step=1,
            key=f"year_{suffix}",
        )
        garage_options = ["No garage", "1 car", "2 cars", "3 cars", "4 cars"]
        garage_label   = st.selectbox(
            "Garage",
            options=garage_options,
            index=2,
            key=f"garage_{suffix}",
        )
        garage_cars = garage_options.index(garage_label)

    with col_b:
        full_bath    = st.selectbox(
            "Full Bathrooms",
            options=["0", "1", "2", "3", "4"],
            index=2,
            key=f"bath_{suffix}",
        )
        overall_cond = st.slider(
            "Overall Condition (1-10)",
            min_value=1, max_value=10, value=5,
            key=f"cond_{suffix}",
            help="1 = Very Poor · 5 = Average · 10 = Excellent",
        )

    return {
        "Neighborhood":   neighbourhood,
        "OverallQual":    overall_qual,
        "OverallCond":    overall_cond,
        "GrLivArea":      to_internal(gr_liv_area),
        "TotalBsmtSF":    to_internal(total_bsmt_sf),
        "YearBuilt":      year_built,
        "YrSold":         2010,
        "GarageCars":     garage_cars,
        "FullBath":       int(full_bath),
        "SaleCondition":  "Normal",
        "Functional":     "Typ",
        "_gr_liv_area":   gr_liv_area,
        "_total_bsmt_sf": total_bsmt_sf,
        "_unit_label":    unit_label,
    }


# Header

st.markdown("# Compare Properties")
st.markdown(
    "<p style='color:#666; font-size:0.9rem; margin-top:-12px'>"
    "Configure two properties side by side and compare estimated prices."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

use_sqm = st.toggle("Use m² instead of sqft", value=False)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

#  Inputs

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("<div class='col-header-a'>Property A</div>", unsafe_allow_html=True)
    inputs_a = property_inputs("a", use_sqm)

with col_b:
    st.markdown("<div class='col-header-b'>Property B</div>", unsafe_allow_html=True)
    inputs_b = property_inputs("b", use_sqm)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
compare_btn = st.button("Compare Prices", type="primary", use_container_width=True)

# Results

if compare_btn:
    with st.spinner("Estimating..."):
        price_a = predict({k: v for k, v in inputs_a.items() if not k.startswith("_")})
        price_b = predict({k: v for k, v in inputs_b.items() if not k.startswith("_")})

    diff      = price_b - price_a
    diff_pct  = (diff / price_a) * 100 if price_a > 0 else 0
    sign      = "+" if diff >= 0 else ""
    diff_color = "#6a9a6a" if diff >= 0 else "#9a6a6a"

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("## Results")

    res_a, res_mid, res_b = st.columns([5, 2, 5], gap="large")

    with res_a:
        st.markdown(f"""
        <div class='price-card price-card-a'>
            <div style='font-size:0.7rem; color:#666; text-transform:uppercase;
                        letter-spacing:0.15em; margin-bottom:8px'>Property A</div>
            <div class='price-value-a'>${price_a:,.0f}</div>
            <div class='price-label'>Estimated Sale Price</div>
        </div>
        """, unsafe_allow_html=True)

    with res_mid:
        st.markdown(
            f"<div style='display:flex; flex-direction:column; align-items:center; "
            f"justify-content:center; height:100%; text-align:center; padding-top:16px'>"
            f"<div style='font-size:1rem; color:#333; letter-spacing:0.1em'>vs</div>"
            f"<div style='color:{diff_color}; font-size:1rem; margin-top:10px'>"
            f"{sign}${abs(diff):,.0f}</div>"
            f"<div style='color:{diff_color}; font-size:0.85rem; margin-top:2px'>"
            f"{sign}{diff_pct:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with res_b:
        st.markdown(f"""
        <div class='price-card price-card-b'>
            <div style='font-size:0.7rem; color:#666; text-transform:uppercase;
                        letter-spacing:0.15em; margin-bottom:8px'>Property B</div>
            <div class='price-value-b'>${price_b:,.0f}</div>
            <div class='price-label'>Estimated Sale Price</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Feature Comparison")

    unit_label = inputs_a["_unit_label"]
    categories = [
        "Overall Quality",
        "Overall Condition",
        f"Living Area ({unit_label})",
        f"Basement ({unit_label})",
        "Garage (cars)",
        "Full Bathrooms",
    ]
    vals_a = [
        inputs_a["OverallQual"],
        inputs_a["OverallCond"],
        inputs_a["_gr_liv_area"],
        inputs_a["_total_bsmt_sf"],
        inputs_a["GarageCars"],
        inputs_a["FullBath"],
    ]
    vals_b = [
        inputs_b["OverallQual"],
        inputs_b["OverallCond"],
        inputs_b["_gr_liv_area"],
        inputs_b["_total_bsmt_sf"],
        inputs_b["GarageCars"],
        inputs_b["FullBath"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Property A", y=categories, x=vals_a,
        orientation="h", marker_color="#c8a96e", opacity=0.85,
        text=[str(v) for v in vals_a],
        textposition="outside", textfont=dict(color="#888", size=10),
    ))
    fig.add_trace(go.Bar(
        name="Property B", y=categories, x=vals_b,
        orientation="h", marker_color="#8fb8c8", opacity=0.85,
        text=[str(v) for v in vals_b],
        textposition="outside", textfont=dict(color="#888", size=10),
    ))
    fig.update_layout(
        barmode="group", height=340,
        margin=dict(l=0, r=60, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#666", size=11),
        legend=dict(
            orientation="h", font=dict(color="#888", size=11),
            bgcolor="rgba(0,0,0,0)", y=-0.12,
        ),
        xaxis=dict(gridcolor="#2a2a2a", showticklabels=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<p style='color:#444; font-size:0.75rem'>"
        "Model trained on Ames, Iowa 2006-2010. "
        "Not generalisable to other markets or time periods.</p>",
        unsafe_allow_html=True,
    )
