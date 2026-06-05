import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import engineer_base_features, engineer_extended_features
from src.pipeline import impute_semantic
from PIL import Image

# Constants

SQM_PER_SQFT = 0.0929

NEIGHBOURHOOD_NAMES: dict[str, str] = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale":  "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "IDOTRR":  "Iowa DOT & Rail Road",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "NAmes":   "North Ames",
    "NoRidge": "Northridge",
    "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights",
    "NWAmes":  "Northwest Ames",
    "OldTown": "Old Town",
    "SWISU":   "South & West of Iowa State University",
    "Sawyer":  "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber":  "Timberland",
    "Veenker": "Veenker",
}

SALE_CONDITION_MAP: dict[str, str] = {
    "Normal sale":               "Normal",
    "Foreclosure / distressed":  "Abnorml",
    "New construction":          "Partial",
    "Family / estate sale":      "Family",
    "Other":                     "Alloca",
}

FUNCTIONAL_MAP: dict[str, str] = {
    "Fully functional": "Typ",
    "Minor issues":     "Min1",
    "Moderate issues":  "Mod",
    "Major issues":     "Maj1",
    "Severely damaged": "Sev",
}


#  Page config
favicon_path = Path(__file__).parent / "house-plus.svg"
with open(favicon_path) as f:
    svg_content = f.read()

svg_content = svg_content.replace('currentColor', '#c8a96e')
svg_content = svg_content.replace('#000000', '#c8a96e')
svg_content = svg_content.replace('black', '#c8a96e')

st.set_page_config(
    page_title="House Price Estimator",
    page_icon=svg_content,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

    h1, h2, h3, h4 {
        font-family: 'DM Serif Display', serif !important;
        letter-spacing: -0.02em;
    }

    .stApp { background-color: #0f0f0f; color: #e8e3d9; }

    .price-display {
        background: linear-gradient(135deg, #1a1a1a 0%, #141414 100%);
        border: 1px solid #333;
        border-left: 4px solid #c8a96e;
        border-radius: 8px;
        padding: 32px;
        text-align: center;
        margin: 24px 0;
    }

    .price-value {
        font-family: 'DM Serif Display', serif;
        font-size: 3.5rem;
        color: #c8a96e;
        letter-spacing: -0.02em;
        line-height: 1;
    }

    .price-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-top: 8px;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #2a2a2a;
        margin: 32px 0;
    }

    .limitation-box {
        background: #1a1209;
        border: 1px solid #3d2b0a;
        border-radius: 6px;
        padding: 16px;
        font-size: 0.85rem;
        color: #a08040;
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

    div[data-testid="metric-container"] {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)


#  Load model

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "model_blend.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


bundle = load_model()


#  Neighbourhood options

@st.cache_data
def build_neighbourhood_options(_bundle) -> dict[str, str]:
    """Display label -> neighbourhood code, sorted by median price descending."""
    items = []
    for code in _bundle["neighbourhood_list"]:
        name   = NEIGHBOURHOOD_NAMES.get(code, code)
        median = _bundle["neighborhood_medians"].get(code, 0)
        items.append((f"{name}  (median ${median:,.0f})", code, median))
    items.sort(key=lambda x: x[2], reverse=True)
    return {label: code for label, code, _ in items}


neighbourhood_options = build_neighbourhood_options(bundle)


#  Predict

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


#  Header

st.markdown("# House Price Estimator")
st.markdown(
    "<p style='color:#666; font-size:0.9rem; margin-top:-12px'>"
    "Ames, Iowa · Blend Ridge + XGBoost · Kaggle score $12,337 RMSE"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

tab_predict, tab_model = st.tabs(["Predict", "Model Info"])


#  Tab 1: Predict

with tab_predict:

    col_inputs, col_result = st.columns([1, 1], gap="large")

    with col_inputs:

        use_sqm     = st.toggle("Use m² instead of sqft", value=False)
        unit_label  = "m²" if use_sqm else "sqft"
        to_internal = (lambda v: int(v / SQM_PER_SQFT)) if use_sqm else (lambda v: v)
        to_display  = (lambda v: round(v * SQM_PER_SQFT)) if use_sqm else (lambda v: v)

        st.markdown("### Property Details")

        neighbourhood_label = st.selectbox(
            "Neighbourhood",
            options=list(neighbourhood_options.keys()),
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
            f"Basement Area ({unit_label}) — 0 if none",
            min_value=0,
            max_value=to_display(3000),
            value=to_display(800),
            step=to_display(50),
        )

        total_area = to_internal(gr_liv_area) + to_internal(total_bsmt_sf)
        st.markdown(
            f"<p class='hint'>Total area: {gr_liv_area + total_bsmt_sf:,} {unit_label}"
            f"{'  (above ground + basement)' if total_bsmt_sf > 0 else '  (above ground only)'}"
            f"</p>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### Structure")

        col_a, col_b = st.columns(2)

        with col_a:
            year_built = st.number_input(
                "Year Built",
                min_value=1870, max_value=2010, value=1980, step=1,
            )
            garage_cars = st.selectbox(
                "Garage Capacity",
                options=["No garage", "1 car", "2 cars", "3 cars", "4 cars"],
                index=2,
            )
            garage_cars_val = ["No garage", "1 car", "2 cars", "3 cars", "4 cars"].index(garage_cars)

        with col_b:
            full_bath = st.selectbox(
                "Full Bathrooms",
                options=["0", "1", "2", "3", "4"],
                index=2,
            )
            overall_cond = st.slider(
                "Overall Condition (1-10)",
                min_value=1, max_value=10, value=5,
                help="1 = Very Poor · 5 = Average · 10 = Excellent",
            )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### Sale Details")

        col_c, col_d = st.columns(2)

        with col_c:
            sale_condition_label = st.selectbox(
                "Sale Type",
                options=list(SALE_CONDITION_MAP.keys()),
                index=0,
            )
            sale_condition = SALE_CONDITION_MAP[sale_condition_label]

        with col_d:
            functional_label = st.selectbox(
                "Home Functionality",
                options=list(FUNCTIONAL_MAP.keys()),
                index=0,
            )
            functional = FUNCTIONAL_MAP[functional_label]

        predict_btn = st.button(
            "Estimate Price", type="primary", use_container_width=True
        )

    with col_result:
        st.markdown("### Estimated Price")

        if predict_btn:
            inputs = {
                "Neighborhood":  neighbourhood,
                "OverallQual":   overall_qual,
                "OverallCond":   overall_cond,
                "GrLivArea":     to_internal(gr_liv_area),
                "TotalBsmtSF":   to_internal(total_bsmt_sf),
                "YearBuilt":     year_built,
                "YrSold":        2010,
                "GarageCars":    garage_cars_val,
                "FullBath":      int(full_bath),
                "SaleCondition": sale_condition,
                "Functional":    functional,
            }

            with st.spinner("Estimating..."):
                price = predict(inputs)

            st.markdown(f"""
            <div class='price-display'>
                <div class='price-value'>${price:,.0f}</div>
                <div class='price-label'>Estimated Sale Price</div>
            </div>
            """, unsafe_allow_html=True)

            low  = price * 0.88
            high = price * 1.12
            st.markdown(
                f"<p style='color:#666; font-size:0.85rem; text-align:center'>"
                f"Typical range: ${low:,.0f} – ${high:,.0f} "
                f"<span style='color:#444'>(±12% based on model MAPE)</span></p>",
                unsafe_allow_html=True,
            )

            st.markdown("#### Key Inputs")

            fig = go.Figure(go.Bar(
                x=[
                    overall_qual,
                    int(full_bath),
                    garage_cars_val,
                    overall_cond,
                ],
                y=["Overall Quality", "Full Bathrooms", "Garage Cars", "Overall Condition"],
                orientation="h",
                marker_color=["#c8a96e", "#8fb8c8", "#8fb8c8", "#8fb8c8"],
                text=[
                    f"{overall_qual}/10",
                    str(int(full_bath)),
                    garage_cars,
                    f"{overall_cond}/10",
                ],
                textposition="outside",
                textfont=dict(color="#888", size=11),
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=60, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#666", size=11),
                xaxis=dict(gridcolor="#2a2a2a", showgrid=True, range=[0, 12]),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("#### Area Breakdown")

            area_above = gr_liv_area
            area_bsmt  = total_bsmt_sf

            fig_area = go.Figure()
            if area_bsmt > 0:
                fig_area.add_trace(go.Bar(
                    name="Basement",
                    x=[area_bsmt], y=[""],
                    orientation="h",
                    marker_color="#3a3a4a",
                    text=[f"{area_bsmt:,} {unit_label}"],
                    textposition="inside",
                    textfont=dict(color="#888", size=11),
                ))
            fig_area.add_trace(go.Bar(
                name="Above Ground",
                x=[area_above], y=[""],
                orientation="h",
                marker_color="#c8a96e",
                text=[f"{area_above:,} {unit_label}"],
                textposition="inside",
                textfont=dict(color="#1a1a1a", size=11),
            ))
            fig_area.update_layout(
                barmode="stack", height=80,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    font=dict(color="#666", size=10),
                    bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False),
            )
            st.plotly_chart(fig_area, use_container_width=True)

            st.markdown(f"""
            <div class='limitation-box'>
                Model trained on Ames, Iowa 2006-2010.
                Systematic -22% error on houses below $100k.
                Not generalisable to other markets.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='
                height:300px; display:flex; align-items:center;
                justify-content:center; border:1px dashed #2a2a2a;
                border-radius:8px; color:#444; font-size:0.85rem;
                letter-spacing:0.1em; text-transform:uppercase;
            '>Fill in the details and click Estimate Price</div>
            """, unsafe_allow_html=True)


#  Tab 2: Model Info

with tab_model:

    st.markdown("### Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Kaggle Score", "$12,337", delta="-716 vs baseline", delta_color="normal")
    with col2:
        st.metric("CV RMSE (Ridge)", "0.1118")
    with col3:
        st.metric("R2 Test", "0.9371")
    with col4:
        st.metric("Features", str(bundle["n_features"]))

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Results by Model")

    results = pd.DataFrame([
        {"Model": "Blend Ridge 50% + XGBoost 50%", "R2": 0.9371, "RMSE": 18647, "CV RMSE": None,   "Overfit Gap": None,  "Kaggle": 12337},
        {"Model": "XGBoost (baseline)",             "R2": 0.9392, "RMSE": 18326, "CV RMSE": None,   "Overfit Gap": 0.041, "Kaggle": 13053},
        {"Model": "Ridge (GridSearch)",             "R2": 0.9306, "RMSE": 19585, "CV RMSE": 0.1118, "Overfit Gap": 0.021, "Kaggle": 12883},
        {"Model": "XGBoost (GridSearch)",           "R2": 0.9301, "RMSE": 19652, "CV RMSE": 0.1150, "Overfit Gap": 0.043, "Kaggle": None},
        {"Model": "Random Forest (GridSearch)",     "R2": 0.9137, "RMSE": 21833, "CV RMSE": 0.1273, "Overfit Gap": 0.064, "Kaggle": None},
        {"Model": "SVR (GridSearch)",               "R2": 0.8152, "RMSE": 31948, "CV RMSE": 0.4903, "Overfit Gap": None,  "Kaggle": None},
    ])
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Kaggle Submission Progression")

    submissions = pd.DataFrame([
        {"Submission": "XGBoost baseline",   "Score": 13053},
        {"Submission": "Ridge (GridSearch)", "Score": 12883},
        {"Submission": "Blend 50/50",        "Score": 12337},
    ])
    fig = go.Figure(go.Bar(
        x=submissions["Score"],
        y=submissions["Submission"],
        orientation="h",
        marker_color=["#3a3a3a", "#555", "#c8a96e"],
        text=[f"${s:,}" for s in submissions["Score"]],
        textposition="outside",
        textfont=dict(color="#999", size=12),
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=80, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#666"),
        xaxis=dict(gridcolor="#2a2a2a", range=[12000, 13500]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Known Limitations")

    for limitation in bundle["known_limitations"]:
        st.markdown(f"- {limitation}")

    st.markdown(
        f"<p style='color:#444; font-size:0.75rem; margin-top:32px'>"
        f"Model trained: {bundle['trained_on']} · "
        f"Features: {bundle['n_features']} · "
        f"Pipeline: Ridge 50% + XGBoost 50%</p>",
        unsafe_allow_html=True,
    )
