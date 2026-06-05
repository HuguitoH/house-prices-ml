import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import engineer_base_features, engineer_extended_features
from src.pipeline import impute_semantic


# Page config
ICON_DIR = Path(__file__).parent.parent / "icons" / "chart-no-axes-combined.svg"
with open(ICON_DIR) as f:
    svg_content = f.read()

svg_content = svg_content.replace('currentColor', '#c8a96e')
svg_content = svg_content.replace('#000000', '#c8a96e')
svg_content = svg_content.replace('black', '#c8a96e')



st.set_page_config(
    page_title="Sensitivity",
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
    .base-price {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-left: 4px solid #c8a96e;
        border-radius: 6px;
        padding: 16px 20px;
        font-size: 0.85rem;
        color: #666;
    }
    .base-price-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #c8a96e;
        letter-spacing: -0.02em;
    }
    .insight-box {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 16px;
        font-size: 0.82rem;
        color: #666;
        margin-top: 16px;
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
neighbourhood_codes   = list(neighbourhood_options.values())
neighbourhood_labels  = list(neighbourhood_options.keys())


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


@st.cache_data
def compute_sensitivity_curve(
    base_inputs: tuple,
    variable: str,
    values: tuple,
) -> list[float]:
    """
    Compute predicted price for each value of the variable.
    base_inputs is a tuple of (key, value) pairs for cache compatibility.
    """
    base = dict(base_inputs)
    prices = []
    for v in values:
        inputs = {**base, variable: v}
        prices.append(predict(inputs))
    return prices


# Header

st.markdown("# Sensitivity Analysis")
st.markdown(
    "<p style='color:#666; font-size:0.9rem; margin-top:-12px'>"
    "Fix a base property and explore how one variable drives the estimated price."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

use_sqm     = st.toggle("Use m² instead of sqft", value=False)
unit_label  = "m²" if use_sqm else "sqft"
to_display  = (lambda v: round(v * SQM_PER_SQFT)) if use_sqm else (lambda v: v)
to_internal = (lambda v: int(v / SQM_PER_SQFT)) if use_sqm else (lambda v: v)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

#  Layout

col_inputs, col_chart = st.columns([1, 2], gap="large")

with col_inputs:

    st.markdown("### Base Property")

    neighbourhood_label = st.selectbox(
        "Neighbourhood",
        options=neighbourhood_labels,
        index=neighbourhood_labels.index(
            next(k for k, v in neighbourhood_options.items() if v == "NAmes")
        ),
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
        help="1 = Very Poor · 5 = Average · 10 = Excellent",
    )

    gr_liv_area = st.number_input(
        f"Above Ground Living Area ({unit_label})",
        min_value=to_display(300),
        max_value=to_display(6000),
        value=to_display(1500),
        step=to_display(50),
    )

    total_bsmt_sf = st.number_input(
        f"Basement Area ({unit_label})",
        min_value=0,
        max_value=to_display(3000),
        value=to_display(800),
        step=to_display(50),
    )

    col_a, col_b = st.columns(2)
    with col_a:
        year_built     = st.number_input(
            "Year Built",
            min_value=1870, max_value=2010,
            value=1990, step=1,
        )
        garage_options = ["No garage", "1 car", "2 cars", "3 cars", "4 cars"]
        garage_label   = st.selectbox(
            "Garage", options=garage_options, index=2,
        )
        garage_cars = garage_options.index(garage_label)

    with col_b:
        full_bath    = st.selectbox(
            "Full Bathrooms",
            options=["0", "1", "2", "3", "4"], index=2,
        )
        overall_cond = st.slider(
            "Overall Condition (1-10)",
            min_value=1, max_value=10, value=5,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Explore Variable")

    variable = st.selectbox(
        "Variable to explore",
        options=[
            "Overall Quality",
            "Overall Condition",
            f"Living Area ({unit_label})",
            "Year Built",
            "Garage (cars)",
            "Full Bathrooms",
            "Neighbourhood",
        ],
    )


# Base inputs dict

base_inputs = {
    "Neighborhood":  neighbourhood,
    "OverallQual":   overall_qual,
    "OverallCond":   overall_cond,
    "GrLivArea":     to_internal(gr_liv_area),
    "TotalBsmtSF":   to_internal(total_bsmt_sf),
    "YearBuilt":     year_built,
    "YrSold":        2010,
    "GarageCars":    garage_cars,
    "FullBath":      int(full_bath),
    "SaleCondition": "Normal",
    "Functional":    "Typ",
}

base_price = predict(base_inputs)


# Variable range + curve

VARIABLE_CONFIG = {
    "Overall Quality": {
        "key":    "OverallQual",
        "values": list(range(1, 11)),
        "labels": [str(v) for v in range(1, 11)],
        "base":   overall_qual,
        "xaxis":  "Overall Quality (1-10)",
    },
    "Overall Condition": {
        "key":    "OverallCond",
        "values": list(range(1, 11)),
        "labels": [str(v) for v in range(1, 11)],
        "base":   overall_cond,
        "xaxis":  "Overall Condition (1-10)",
    },
    f"Living Area ({unit_label})": {
        "key":    "GrLivArea",
        "values": [to_internal(v) for v in range(to_display(300), to_display(4001), to_display(100))],
        "labels": [str(to_display(v)) for v in range(to_internal(300), to_internal(4001), to_internal(100))],
        "base":   to_internal(gr_liv_area),
        "xaxis":  f"Living Area ({unit_label})",
    },
    "Year Built": {
        "key":    "YearBuilt",
        "values": list(range(1900, 2011, 5)),
        "labels": [str(v) for v in range(1900, 2011, 5)],
        "base":   year_built,
        "xaxis":  "Year Built",
    },
    "Garage (cars)": {
        "key":    "GarageCars",
        "values": [0, 1, 2, 3, 4],
        "labels": ["None", "1", "2", "3", "4"],
        "base":   garage_cars,
        "xaxis":  "Garage Capacity (cars)",
    },
    "Full Bathrooms": {
        "key":    "FullBath",
        "values": [0, 1, 2, 3, 4],
        "labels": ["0", "1", "2", "3", "4"],
        "base":   int(full_bath),
        "xaxis":  "Full Bathrooms",
    },
    "Neighbourhood": {
        "key":    "Neighborhood",
        "values": neighbourhood_codes,
        "labels": [NEIGHBOURHOOD_NAMES.get(c, c) for c in neighbourhood_codes],
        "base":   neighbourhood,
        "xaxis":  "Neighbourhood (sorted by median price)",
    },
}

cfg    = VARIABLE_CONFIG[variable]
values = cfg["values"]
labels = cfg["labels"]

prices = compute_sensitivity_curve(
    base_inputs=tuple(sorted(base_inputs.items())),
    variable=cfg["key"],
    values=tuple(values),
)

# Find base value index
try:
    base_idx = values.index(cfg["base"])
except ValueError:
    base_idx = len(values) // 2


# Chart

with col_chart:

    # Base price display
    st.markdown(
        f"<div class='base-price'>"
        f"<div style='margin-bottom:4px'>Base property estimate</div>"
        f"<div class='base-price-value'>${base_price:,.0f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(f"### Price vs {variable}")

    # Colour each point: gold for current base, blue for others
    colors = ["#c8a96e" if i == base_idx else "#8fb8c8"
              for i in range(len(values))]

    fig = go.Figure()

    # Area fill under curve
    fig.add_trace(go.Scatter(
        x=labels, y=prices,
        fill="tozeroy",
        fillcolor="rgba(143, 184, 200, 0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Line
    fig.add_trace(go.Scatter(
        x=labels, y=prices,
        mode="lines",
        line=dict(color="#8fb8c8", width=2),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Points
    fig.add_trace(go.Scatter(
        x=labels, y=prices,
        mode="markers",
        marker=dict(color=colors, size=8, line=dict(color="#0f0f0f", width=1)),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
    ))

    # Base price reference line
    fig.add_hline(
        y=base_price,
        line=dict(color="#c8a96e", width=1, dash="dot"),
        annotation_text=f"Base  ${base_price:,.0f}",
        annotation_font=dict(color="#c8a96e", size=11),
        annotation_position="top right",
    )

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=16, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#666", size=11),
        xaxis=dict(
            gridcolor="#1a1a1a",
            tickfont=dict(size=10),
            tickangle=-45 if variable == "Neighbourhood" else 0,
        ),
        yaxis=dict(
            gridcolor="#2a2a2a",
            tickformat="$,.0f",
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Insight box
    price_min  = min(prices)
    price_max  = max(prices)
    price_range = price_max - price_min
    pct_range  = (price_range / price_min) * 100 if price_min > 0 else 0

    label_min = labels[prices.index(price_min)]
    label_max = labels[prices.index(price_max)]

    st.markdown(
        f"<div class='insight-box'>"
        f"<b style='color:#aaa'>{variable}</b> drives a "
        f"<b style='color:#c8a96e'>${price_range:,.0f} ({pct_range:.0f}%)</b> "
        f"price range across all values — "
        f"from <b style='color:#9a6a6a'>${price_min:,.0f}</b> ({label_min}) "
        f"to <b style='color:#6a9a6a'>${price_max:,.0f}</b> ({label_max})."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='color:#333; font-size:0.7rem; margin-top:16px'>"
        "All other variables held constant at base property values. "
        "Model trained on Ames, Iowa 2006-2010."
        "</p>",
        unsafe_allow_html=True,
    )
