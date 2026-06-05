"""
pages/01_Map.py
---------------
Neighbourhood map - Voronoi polygons from neighbourhood centroids.
Click a polygon to highlight it. Toggle tiers to filter.
"""

import pickle
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import Voronoi
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon
from streamlit_folium import st_folium


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neighbourhood Map",
    page_icon="🗺️",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif !important; letter-spacing: -0.02em; }
    .stApp { background-color: #0f0f0f; color: #e8e3d9; }
    .section-divider { border: none; border-top: 1px solid #2a2a2a; margin: 24px 0; }
    .selected-card {
        background: #1a1a1a;
        border: 1px solid #333;
        border-left: 4px solid #c8a96e;
        border-radius: 8px;
        padding: 16px;
        margin-top: 8px;
    }
    div[data-testid="stButton"] button {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        min-height: 0 !important;
        height: 24px !important;
        width: 24px !important;
        box-shadow: none !important;
    }
    div[data-testid="stButton"] button:hover {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

NEIGHBOURHOOD_COORDS: dict[str, tuple[float, float]] = {
    "Blmngtn": (42.0614, -93.6200),
    "Blueste":  (42.0300, -93.6300),
    "BrDale":   (42.0521, -93.6450),
    "BrkSide":  (42.0270, -93.6150),
    "ClearCr":  (42.0650, -93.6800),
    "CollgCr":  (42.0200, -93.6900),
    "Crawfor":  (42.0240, -93.6200),
    "Edwards":  (42.0150, -93.6700),
    "Gilbert":  (42.1050, -93.6500),
    "IDOTRR":   (42.0180, -93.6100),
    "MeadowV":  (41.9990, -93.6800),
    "Mitchel":  (41.9900, -93.6200),
    "NAmes":    (42.0450, -93.6500),
    "NoRidge":  (42.0650, -93.6600),
    "NPkVill":  (42.0500, -93.6350),
    "NridgHt":  (42.0580, -93.6700),
    "NWAmes":   (42.0500, -93.6800),
    "OldTown":  (42.0270, -93.6130),
    "SWISU":    (42.0180, -93.6500),
    "Sawyer":   (42.0350, -93.6700),
    "SawyerW":  (42.0300, -93.6850),
    "Somerst":  (42.0620, -93.6550),
    "StoneBr":  (42.0600, -93.6400),
    "Timber":   (41.9980, -93.7000),
    "Veenker":  (42.0430, -93.6600),
}

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

TIERS = [
    ("Top tier",      "> $200k",     "#c8a96e"),
    ("Above average", "$160k–$200k", "#8fb8c8"),
    ("Average",       "$130k–$160k", "#6a9a6a"),
    ("Below average", "< $130k",     "#9a6a6a"),
]


# ── Load bundle ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_bundle():
    path = Path(__file__).parent.parent / "models" / "model_blend.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


bundle = load_bundle()


# ── Build neighbourhood dataframe ─────────────────────────────────────────────

@st.cache_data
def build_df(_bundle) -> pd.DataFrame:
    rows = []
    for code, (lat, lng) in NEIGHBOURHOOD_COORDS.items():
        median = _bundle["neighborhood_medians"].get(code, 0)
        rows.append({
            "code":   code,
            "name":   NEIGHBOURHOOD_NAMES.get(code, code),
            "lat":    lat,
            "lng":    lng,
            "median": median,
        })
    return pd.DataFrame(rows).sort_values("median", ascending=False).reset_index(drop=True)


df         = build_df(bundle)
median_min = int(df["median"].min())
median_max = int(df["median"].max())


def price_to_color(median: float) -> str:
    t = (median - median_min) / (median_max - median_min)
    if t > 0.75:   return "#c8a96e"
    elif t > 0.50: return "#8fb8c8"
    elif t > 0.25: return "#6a9a6a"
    else:          return "#9a6a6a"


def get_tier_label(median: float) -> str:
    t = (median - median_min) / (median_max - median_min)
    if t > 0.75:   return "Top tier"
    elif t > 0.50: return "Above average"
    elif t > 0.25: return "Average"
    else:          return "Below average"


# ── Voronoi + Sutherland-Hodgman clipping ─────────────────────────────────────

def clip_polygon_to_bbox(polygon, min_lat, max_lat, min_lng, max_lng):
    def inside(p, edge):
        if edge == "left":   return p[1] >= min_lng
        if edge == "right":  return p[1] <= max_lng
        if edge == "bottom": return p[0] >= min_lat
        if edge == "top":    return p[0] <= max_lat

    def intersection(p1, p2, edge):
        lat1, lng1 = p1
        lat2, lng2 = p2
        if edge in ("left", "right"):
            lng = min_lng if edge == "left" else max_lng
            if lng2 == lng1: return p1
            t = (lng - lng1) / (lng2 - lng1)
            return [lat1 + t * (lat2 - lat1), lng]
        else:
            lat = min_lat if edge == "bottom" else max_lat
            if lat2 == lat1: return p1
            t = (lat - lat1) / (lat2 - lat1)
            return [lat, lng1 + t * (lng2 - lng1)]

    output = polygon[:]
    for edge in ("left", "right", "bottom", "top"):
        if not output:
            break
        inp, output = output, []
        for i in range(len(inp)):
            curr, prev = inp[i], inp[i - 1]
            if inside(curr, edge):
                if not inside(prev, edge):
                    output.append(intersection(prev, curr, edge))
                output.append(curr)
            elif inside(prev, edge):
                output.append(intersection(prev, curr, edge))
    return output


@st.cache_data
def compute_voronoi(df: pd.DataFrame):
    points = df[["lng", "lat"]].values
    BBOX   = {"min_lat": 41.985, "max_lat": 42.120,
               "min_lng": -93.720, "max_lng": -93.590}
    cx = (BBOX["min_lng"] + BBOX["max_lng"]) / 2
    cy = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
    mirror = np.array([
        [BBOX["min_lng"] - 0.5, cy],
        [BBOX["max_lng"] + 0.5, cy],
        [cx, BBOX["min_lat"] - 0.5],
        [cx, BBOX["max_lat"] + 0.5],
    ])
    vor = Voronoi(np.vstack([points, mirror]))

    polygons = []
    for i, code in enumerate(df["code"]):
        region = vor.regions[vor.point_region[i]]
        if -1 in region or len(region) < 3:
            continue
        verts   = [[float(vor.vertices[v][1]), float(vor.vertices[v][0])]
                   for v in region]
        clipped = clip_polygon_to_bbox(verts, **BBOX)
        if len(clipped) >= 3:
            polygons.append((code, clipped))
    return polygons


polygons = compute_voronoi(df)


# ── Point-in-polygon click detection ─────────────────────────────────────────

def find_clicked_neighbourhood(lat: float, lng: float) -> str | None:
    """Return the neighbourhood code whose Voronoi polygon contains the point."""
    point = Point(lat, lng)
    for code, poly_coords in polygons:
        # poly_coords is [[lat, lng], ...] — Shapely expects (x, y) = (lng, lat)
        shapely_poly = ShapelyPolygon([(c[1], c[0]) for c in poly_coords])
        if shapely_poly.contains(point):
            return code
    return None


# ── Session state ─────────────────────────────────────────────────────────────

if "selected_code" not in st.session_state:
    st.session_state["selected_code"] = None

for label, _, _ in TIERS:
    if f"tier_{label}" not in st.session_state:
        st.session_state[f"tier_{label}"] = True


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Neighbourhood Map")
st.markdown(
    "<p style='color:#666; font-size:0.9rem; margin-top:-12px'>"
    "Ames, Iowa · 25 neighbourhoods · click a zone to select · "
    "zones via Voronoi tessellation"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────────────

col_map, col_info = st.columns([2, 1], gap="large")

with col_info:

    st.markdown("### Price Tiers")
    st.markdown(
        "<p style='color:#555; font-size:0.75rem; margin-top:-8px; margin-bottom:12px'>"
        "Click to show / hide</p>",
        unsafe_allow_html=True,
    )

    tier_active = {}
    for label, price_range_label, color in TIERS:
        key    = f"tier_{label}"
        active = st.session_state[key]

        col_tier, col_btn = st.columns([10, 1])
        with col_tier:
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:10px; "
                f"margin-bottom:6px; font-size:0.85rem; "
                f"opacity:{'1.0' if active else '0.3'}'>"
                f"<div style='width:14px; height:14px; border-radius:3px; "
                f"background:{color}; flex-shrink:0'></div>"
                f"<span style='color:#aaa'>{label}</span>"
                f"<span style='color:#555; margin-left:auto'>{price_range_label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button("·", key=f"btn_{label}", help=f"Toggle {label}"):
                st.session_state[key] = not st.session_state[key]
                st.rerun()

        tier_active[label] = st.session_state[key]

    show_labels = st.toggle("Show labels", value=True)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Filtered codes
    filtered_codes = set(
        df.loc[
            df["median"].apply(get_tier_label).apply(
                lambda x: tier_active.get(x, True)
            ),
            "code",
        ]
    )

    # Selected neighbourhood card
    selected = st.session_state["selected_code"]
    if selected and selected in df.set_index("code").index:
        row   = df.set_index("code").loc[selected]
        color = price_to_color(row["median"])
        tier  = get_tier_label(row["median"])
        st.markdown(
            f"<div class='selected-card'>"
            f"<div style='font-size:0.7rem; color:#666; text-transform:uppercase; "
            f"letter-spacing:0.1em; margin-bottom:4px'>Selected</div>"
            f"<div style='font-size:1.1rem; color:{color}; font-weight:600'>"
            f"{row['name']}</div>"
            f"<div style='font-size:1.4rem; color:#e8e3d9; margin-top:4px'>"
            f"${row['median']:,.0f}</div>"
            f"<div style='font-size:0.75rem; color:#555; margin-top:4px'>"
            f"Median sale price · {tier}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Neighbourhood ranking
    active_df = df[df["code"].isin(filtered_codes)]
    st.markdown(f"### {len(active_df)} / {len(df)} neighbourhoods")

    for _, row in active_df.iterrows():
        color      = price_to_color(row["median"])
        is_selected = row["code"] == selected
        bg         = "background:#1a1a1a; border-radius:4px; padding:2px 4px;" if is_selected else ""
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px; "
            f"margin-bottom:5px; font-size:0.78rem; {bg}'>"
            f"<div style='width:8px; height:8px; border-radius:2px; "
            f"background:{color}; flex-shrink:0'></div>"
            f"<span style='color:{'#e8e3d9' if is_selected else '#aaa'}; flex:1'>"
            f"{row['name']}</span>"
            f"<span style='color:#666'>${row['median']:,.0f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<p style='color:#333; font-size:0.7rem; margin-top:24px'>"
        "Zones are Voronoi tessellations from neighbourhood centroids, "
        "not official city boundaries."
        "</p>",
        unsafe_allow_html=True,
    )


# ── Map ───────────────────────────────────────────────────────────────────────

with col_map:

    selected    = st.session_state["selected_code"]
    m           = folium.Map(location=[42.034, -93.660], zoom_start=12,
                             tiles="CartoDB dark_matter")
    code_to_row = df.set_index("code")

    for code, poly_coords in polygons:
        row        = code_to_row.loc[code]
        color      = price_to_color(row["median"])
        active     = code in filtered_codes
        is_selected = code == selected

        if is_selected:
            # Selected: bright border, higher opacity
            border_color   = "#ffffff"
            border_weight  = 3
            fill_opacity   = 0.70
        elif active:
            border_color   = color
            border_weight  = 1.5
            fill_opacity   = 0.40
        else:
            border_color   = "#2a2a2a"
            border_weight  = 0.5
            fill_opacity   = 0.08

        folium.Polygon(
            locations=poly_coords,
            color=border_color,
            weight=border_weight,
            fill=True,
            fill_color=color if active else "#1a1a1a",
            fill_opacity=fill_opacity,
            tooltip=f"{row['name']}  •  ${row['median']:,.0f}" if active else None,
        ).add_to(m)

        if active:
            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=4 if not is_selected else 6,
                color="#ffffff" if is_selected else color,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
            ).add_to(m)

            if show_labels:
                folium.Marker(
                    location=[row["lat"], row["lng"]],
                    icon=folium.DivIcon(
                        html=(
                            f"<div style='font-family:monospace; font-size:8px; "
                            f"color:{'#fff' if is_selected else '#ddd'}; "
                            f"white-space:nowrap; font-weight:{'bold' if is_selected else 'normal'}; "
                            f"text-shadow:0 0 3px #000,0 0 3px #000,0 0 3px #000; "
                            f"margin-top:-5px; margin-left:7px'>"
                            f"{row['name'].split()[0]}</div>"
                        ),
                        icon_size=(120, 16), icon_anchor=(0, 0),
                    ),
                ).add_to(m)

    map_data = st_folium(m, width="100%", height=540)

    # Detect click and find which neighbourhood was clicked
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]
        clicked_code = find_clicked_neighbourhood(clicked_lat, clicked_lng)
        if clicked_code and clicked_code != st.session_state["selected_code"]:
            st.session_state["selected_code"] = clicked_code
            st.rerun()
        elif clicked_code == st.session_state["selected_code"]:
            # Click on same → deselect
            st.session_state["selected_code"] = None
            st.rerun()
