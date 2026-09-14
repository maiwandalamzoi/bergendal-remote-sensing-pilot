"""
Streamlit dashboard for the Berg en Dal pilot — built as something to put in
front of the gemeente (and, on the nitrogen/crop pages, farmers) rather than
just a developer's map viewer. A real English/Dutch language switch (not
both languages inline at once) since the audience is Dutch: every string a
person reads goes through t(en, nl) below, keyed off st.session_state.lang.
Genuinely Dutch data values (BRP crop/category names, place names) stay
Dutch regardless of the switch — those are the actual registry records, not
UI chrome to translate.

    streamlit run dashboard.py

Needs run_pipeline.py plus src/fetch_cbs.py, src/fetch_brp.py and
src/fetch_air_quality.py to have been run at least once.
"""
import datetime
import json
import sys
from pathlib import Path

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
import streamlit as st
from rasterio.warp import transform as warp_transform
from shapely.geometry import Point
from streamlit_folium import st_folium

# Chart palette -- validated (CVD/contrast, see dataviz skill) for exactly
# this use: two data-provenance categories (Landsat vs Sentinel-2) on the
# NDVI/NDWI trends, distinct from the site's own UI colours (--forest/
# --river/--loess below) so a chart legend is never mistaken for site chrome.
COLOR_LANDSAT = "#eb6834"
COLOR_SENTINEL2 = "#2a78d6"
COLOR_WATER = "#3B6E8A"    # matches --river, reused for single-series water/flood charts
COLOR_DROUGHT = "#C03B2E"  # matches the flood "newly flooded" red -- reused here for "stress"
COLOR_TEMP = "#eb6834"
COLOR_SUN = "#eda100"
COLOR_SELECTED = "#4a3aa7"  # Year Explorer highlight ring -- distinct from the fixed 2018-drought callout

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from statsutil import load_stats
from visualize import (make_map, field_explorer_map, FIELD_COLOR_MODES, CROP_FAMILIES, classify_crop,
                        LANDCOVER_COLORS, landcover_label, landcover_icon_svg, BRP_COLORS)
from report import build_report_pdf, REPORT_LAYER_CHOICES
from crop_rotation import normalize_crop
from methodology import ENTRIES as METHOD_ENTRIES

BRP_PATH = Path(__file__).resolve().parent / "data" / "raw" / "brp_parcels.geojson"

st.set_page_config(page_title="Berg en Dal — remote sensing pilot", layout="wide", page_icon="🛰️")

if "lang" not in st.session_state:
    # Dutch by default -- the primary audience (Berg en Dal municipal
    # staff, local farmers) is Dutch; English stays one click away via
    # the sidebar toggle for anyone who needs it.
    st.session_state["lang"] = "nl"


def t(en: str, nl: str) -> str:
    """The one language switch point: every user-facing string in this file
    routes through here. Registry data (crop names, BRP categories) is
    never passed through t() -- it stays in its real, authoritative Dutch
    regardless of the UI language."""
    return en if st.session_state.get("lang", "en") == "en" else nl


# Small, hand-drawn line icons (Feather/Lucide-style: single stroke colour,
# rounded caps, plain geometric shapes) for the headline stat cards -- the
# deliberate alternative to emoji after the icon-reduction pass: a real,
# restrained icon system instead of either colourful emoji or no icon at
# all. Built as plain inline SVG (no icon font/CDN dependency) so they
# always render, in print too.
_ICON_PATHS = {
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="8" y1="3" x2="8" y2="7"/><line x1="16" y1="3" x2="16" y2="7"/>',
    "grid": '<rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/><rect x="13" y="13" width="8" height="8" rx="1"/>',
    "satellite": '<circle cx="12" cy="9" r="5"/><path d="M12 14 L12 21"/><path d="M8 21 L16 21"/><path d="M6 5 L3 2"/><path d="M18 5 L21 2"/>',
    "wave": '<path d="M2 14c2-3 4-3 6 0s4 3 6 0 4-3 6 0"/><path d="M2 18c2-3 4-3 6 0s4 3 6 0 4-3 6 0"/>',
    "person": '<circle cx="12" cy="8" r="3.2"/><path d="M5 20c0-4 3-6.5 7-6.5s7 2.5 7 6.5"/>',
    "house": '<path d="M4 11 L12 4 L20 11"/><path d="M6 10 V20 H18 V10"/>',
    "leaf": '<path d="M6 20c-2-8 2-15 13-16 1 11-6 15-13 16z"/><path d="M7 19c3-4 6-7 11-13"/>',
    "pin": '<path d="M12 21s7-7.5 7-12a7 7 0 1 0-14 0c0 4.5 7 12 7 12z"/><circle cx="12" cy="9" r="2.4"/>',
    "sun": '<circle cx="12" cy="12" r="4.5"/><line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/><line x1="4.5" y1="4.5" x2="6.5" y2="6.5"/><line x1="17.5" y1="17.5" x2="19.5" y2="19.5"/><line x1="4.5" y1="19.5" x2="6.5" y2="17.5"/><line x1="17.5" y1="6.5" x2="19.5" y2="4.5"/>',
}


def icon_svg(name: str, size: int = 18) -> str:
    body = _ICON_PATHS[name]
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


# The header's own dark green, reused as the KPI cards' data-ink colour --
# a micro-visual should read as part of this page's design, not a grey
# decorative afterthought in a different palette.
KPI_INK = "#2E5943"


def kpi_sparkline_svg(values: list, width: int = 128, height: int = 34) -> str:
    """A real micro-chart: an actual line through the real yearly values,
    first and last value labelled directly on the line -- at this size
    that pair of numbers *is* the axis, not a decoration with nothing
    readable on it. Only called for series with >=2 real points; a
    single-point or empty series renders nothing (see kpi_card below)."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    pad_l, pad_r, pad_t, pad_b = 2, 2, 8, 10
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(vals)
    pts = [(pad_l + i * w / (n - 1), pad_t + h - (v - lo) / rng * h) for i, v in enumerate(vals)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    fx, fy = pts[0]
    lx, ly = pts[-1]
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="display:block;">'
        f'<polyline points="{path}" fill="none" stroke="{KPI_INK}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="2.2" fill="#B9C2AE"/>'
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.6" fill="{KPI_INK}"/>'
        f'<text x="{fx:.1f}" y="{height - 1}" font-size="9" fill="#9aa295" text-anchor="start">{vals[0]:,.0f}</text>'
        f'<text x="{lx:.1f}" y="{height - 1}" font-size="9" fill="{KPI_INK}" text-anchor="end" '
        f'font-weight="600">{vals[-1]:,.0f}</text>'
        f'</svg>'
    )


def kpi_proportion_bar_svg(pct: float, width: int = 128, height: int = 10) -> str:
    """A percentage as a filled proportion bar -- the honest micro-visual
    for a single value with no time dimension to plot as a trend (a fake
    sparkline on one data point would be decoration, not information)."""
    pct = max(0.0, min(100.0, pct))
    filled = pct / 100 * width
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="display:block;">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="{height / 2:.1f}" fill="#E2ECE4"/>'
        f'<rect x="0" y="0" width="{filled:.1f}" height="{height}" rx="{height / 2:.1f}" fill="{KPI_INK}"/>'
        f'</svg>'
    )


def kpi_stacked_bar_svg(segments: list[tuple[float, str]], width: int = 128, height: int = 14) -> str:
    """A composition, not a single proportion -- e.g. farmland vs the rest
    of the municipality, or land vs water. Each segment gets its own real
    share of the total width, with a thin gap between segments (never one
    flat fill implying a single uniform quantity)."""
    total = sum(max(0.0, v) for v, _ in segments) or 1.0
    gap = 2
    parts = []
    x = 0.0
    n = len(segments)
    for i, (v, color) in enumerate(segments):
        raw_w = max(0.0, v) / total * width
        seg_w = max(0.0, raw_w - (gap if i < n - 1 else 0))
        if seg_w > 0:
            parts.append(f'<rect x="{x:.1f}" y="0" width="{seg_w:.1f}" height="{height}" rx="3" fill="{color}"/>')
        x += raw_w
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="display:block;">'
        + "".join(parts) + "</svg>"
    )


def index_range_svg(value: float, vmin: float, vmax: float, width: int = 128, height: int = 10,
                     low_color: str = "#A2712F", high_color: str = "#2E5943", key: str = "idx") -> str:
    """A bounded scientific index (NDVI's -1..+1, a percentage's 0..100,
    etc.) placed on its own real, fixed scale -- a marker on a diverging
    track from low_color (vmin) through a neutral grey midpoint to
    high_color (vmax), both endpoints labelled with the scale itself so
    the number's *position* in its real range is visible, not just its
    digits. `key` only needs to be unique enough to avoid two gradient
    ids colliding on the same page."""
    v = max(vmin, min(vmax, value))
    frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    x = frac * width
    grad_id = f"bd-idx-{key}"
    return (
        f'<svg width="{width}" height="{height + 12}" viewBox="0 0 {width} {height + 12}" style="display:block;">'
        f'<defs><linearGradient id="{grad_id}" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="{low_color}"/><stop offset="50%" stop-color="#CBD3C1"/>'
        f'<stop offset="100%" stop-color="{high_color}"/></linearGradient></defs>'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="{height / 2:.1f}" fill="url(#{grad_id})" opacity="0.5"/>'
        f'<line x1="{x:.1f}" y1="-2" x2="{x:.1f}" y2="{height + 2}" stroke="#16221C" stroke-width="2.2" '
        f'stroke-linecap="round"/>'
        f'<text x="0" y="{height + 11}" font-size="9" fill="#9aa295" text-anchor="start">{vmin:g}</text>'
        f'<text x="{width}" y="{height + 11}" font-size="9" fill="#9aa295" text-anchor="end">{vmax:g}</text>'
        f'</svg>'
    )


def metric_with_bar(col, label: str, value_str: str, visual_svg: str) -> None:
    """A st.metric-style number, immediately followed by one small inline
    range/proportion visual underneath it -- used for the many single
    scientific-index or percentage metrics scattered outside the Overview
    KPI cards (NDVI, NDWI, cloud-free %, soil texture %, IoU %, ...) so
    each one shows *where on its real scale* the number sits, not just
    the digit. Reuses st.metric for the label/value (native styling,
    keyboard/accessibility for free) and adds only the visual under it."""
    col.metric(label, value_str)
    col.markdown(f'<div style="margin-top:-8px;">{visual_svg}</div>', unsafe_allow_html=True)


def kpi_card(label: str, period: str, value: str, unit: str, context: str, visual_svg: str) -> None:
    """Every KPI card, one consistent structure: label, period, big number
    + unit, one-line context, then a fixed-height visual slot (a real
    sparkline/proportion bar, or empty -- never a fake line) -- so all
    five cards render the same height regardless of which have a visual,
    and the More/info buttons below them land at the same row."""
    st.markdown(f"""
    <div class="bd-kpi-card">
      <div class="bd-kpi-label">{label}</div>
      <div class="bd-kpi-period">{period}</div>
      <div class="bd-kpi-value">{value}<span class="bd-kpi-unit">{unit}</span></div>
      <div class="bd-kpi-context">{context}</div>
      <div class="bd-kpi-visual">{visual_svg}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_brp_gdf():
    """The precise (unsimplified) parcel geometry + exact zonal-stat NDVI --
    used for the click-to-inspect lookup, kept separate from the
    display-simplified copy field_explorer_map() draws.

    Returns None, not a raised exception, when BRP_PATH isn't present --
    a deploy that only has the small committed stats.json (see
    .gitignore) doesn't have this file. Every caller below checks for
    None and degrades that one section instead of crashing the whole
    page; centralizing the try/except here, once, replaced a real bug
    where three separate call sites each needed their own guard and one
    (the Land & Crops crop-family chart) didn't have one yet, confirmed
    live on Streamlit Community Cloud."""
    try:
        return gpd.read_file(BRP_PATH)
    except Exception:
        return None


def _needs_brp_geometry(section: str) -> None:
    """The one message every load_brp_gdf()-is-None call site shows --
    centralized so the wording (and the fix instructions) can't drift
    between the map tabs, the crop-family chart, and the Forecast tab's
    rotation summary, all of which read the same missing file."""
    st.info(t(
        f"{section} needs the full BRP parcel archive, which this deployment doesn't have (only the "
        f"small `stats.json` summary is included here) — run `python run_pipeline.py` locally for this.",
        f"{section} heeft het volledige BRP-perceelarchief nodig, dat deze deployment niet heeft (alleen "
        f"de kleine `stats.json`-samenvatting is hier meegenomen) — draai `python run_pipeline.py` lokaal "
        f"hiervoor.",
    ))


def field_at(lat: float, lon: float):
    gdf = load_brp_gdf()
    if gdf is None:
        return None
    point = Point(lon, lat)
    hit = gdf[gdf.contains(point)]
    return hit.iloc[0] if len(hit) else None


RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent / "data" / "processed"


def _sample(path: Path, lat: float, lon: float, band: int = 1):
    """One pixel value from a raster at a WGS84 point, reprojected to
    whatever CRS that raster is actually stored in -- AHN and RIVM stay in
    RD New, the Sentinel rasters in UTM, so this can't assume one CRS."""
    with rasterio.open(path) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        val = next(src.sample([(xs[0], ys[0])]))[band - 1]
        if src.nodata is not None and np.isclose(val, src.nodata):
            return None
        if np.isnan(val):
            return None
        return float(val)


def point_samples(lat: float, lon: float) -> dict:
    """Elevation, canopy/building height, and SAR backscatter/water at one
    clicked point -- the "what else do we know about this exact spot"
    extension to the field-level BRP/NDVI detail panel."""
    elevation = _sample(RAW_DIR / "dtm_05m.tif", lat, lon)
    canopy = _sample(RAW_DIR / "ndsm_05m.tif", lat, lon)
    valid_flag = _sample(PROC_DIR / "sar_2025_indices.tif", lat, lon, band=4)
    vv_db = _sample(PROC_DIR / "sar_2025_indices.tif", lat, lon, band=1) if valid_flag else None
    water = _sample(PROC_DIR / "sar_2025_indices.tif", lat, lon, band=3) if valid_flag else None
    return {
        "elevation_m": elevation,
        "canopy_height_m": canopy,
        "sar_vv_db": vv_db,
        "sar_water": bool(water) if water is not None else None,
    }


st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap">
<style>
:root { --forest: #2E5943; --forest-soft: #E2ECE4; --river: #3B6E8A; --loess: #A2712F; --ink: #16221C; --ink-2: #51604F; --line: #CBD3C1; }
html, body, [class*="css"] { font-family: "Source Sans 3", system-ui, sans-serif; }
h1, h2, h3 { font-family: "Fraunces", Georgia, serif !important; color: var(--ink); }
/* A colored accent bar to the left of every section heading -- one small,
   consistent "designed" touch repeated everywhere, rather than decoration
   piled onto any one spot. h1 is only ever the hero header (styled
   separately below) so this only ever touches real section headings. */
[data-testid="stAppViewContainer"] h2, [data-testid="stAppViewContainer"] h3 {
  border-left: 4px solid var(--forest); padding-left: 14px; margin: 30px 0 14px !important;
}
[data-testid="stAppViewContainer"] h4 { color: var(--ink); margin-top: 22px !important; }
hr { margin: 30px 0 !important; border-color: var(--line) !important; }
.bd-header {
  background:
    repeating-linear-gradient(135deg, rgba(255,255,255,.045) 0px, rgba(255,255,255,.045) 2px, transparent 2px, transparent 16px),
    radial-gradient(circle at 85% 15%, rgba(255,255,255,.07) 0%, transparent 45%),
    linear-gradient(135deg, #2E5943 0%, #1B3E2E 100%);
  color: #F3F6F1; padding: 30px 34px; border-radius: 14px; margin-bottom: 26px;
}
.bd-header h1 { color: #FFFFFF !important; margin: 0 0 8px 0; font-size: 2.1rem; border-left: none !important; padding-left: 0 !important; }
.bd-header p { margin: 0; color: #DCE7DD; font-size: 0.96rem; line-height: 1.5; }
.bd-header .bd-meta { display:block; margin-top:8px; opacity:.72; font-size:.82rem; }
.bd-tag { display:inline-block; font-family:"JetBrains Mono", monospace; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; background: rgba(255,255,255,.15); color:#EAF1EA; padding:4px 11px; border-radius:20px; margin-bottom:12px; }
div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 12px; box-shadow: 0 1px 3px rgba(22,34,28,.07); transition: box-shadow .15s ease, transform .15s ease; }
div[data-testid="stMetric"]:hover { box-shadow: 0 4px 12px rgba(22,34,28,.1); transform: translateY(-1px); }
div[data-testid="stMetricLabel"] { color: var(--ink-2); }
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--forest) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--forest) !important; }
[data-testid="stExpander"] { border-radius: 12px !important; border: 1px solid var(--line) !important; box-shadow: 0 1px 3px rgba(22,34,28,.06); }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }
[data-testid="stPopoverBody"] { border-radius: 12px !important; }
.bd-stat-strip { display: flex; flex-wrap: wrap; gap: 10px; margin: -8px 0 22px; }
.bd-stat-pill {
  flex: 1 1 160px; background: #FFFFFF; border: 1px solid var(--line); border-radius: 12px;
  padding: 12px 16px; box-shadow: 0 1px 3px rgba(22,34,28,.07); display: flex; gap: 12px; align-items: flex-start;
  transition: box-shadow .15s ease, transform .15s ease;
}
.bd-stat-pill:hover { box-shadow: 0 4px 12px rgba(22,34,28,.1); transform: translateY(-1px); }
.bd-stat-icon { flex: 0 0 auto; width: 34px; height: 34px; border-radius: 9px; background: var(--forest-soft); color: var(--forest); display: flex; align-items: center; justify-content: center; }
.bd-stat-num { font-family: "Fraunces", Georgia, serif; font-weight: 600; font-size: 1.5rem; color: var(--forest); line-height: 1.1; }
.bd-stat-lbl { color: var(--ink-2); font-size: .8rem; margin-top: 2px; }
.bd-kpi-card {
  background: #FFFFFF; border: 1px solid var(--line); border-radius: 12px;
  padding: 14px 16px 12px; box-shadow: 0 1px 3px rgba(22,34,28,.07);
  height: 172px; display: flex; flex-direction: column;
  transition: box-shadow .15s ease, transform .15s ease; margin-bottom: 6px;
}
.bd-kpi-card:hover { box-shadow: 0 4px 12px rgba(22,34,28,.1); transform: translateY(-1px); }
.bd-kpi-label { font-size: .82rem; color: var(--ink-2); font-weight: 600; }
.bd-kpi-period { font-size: .7rem; color: #9aa295; margin-top: 1px; }
.bd-kpi-value { font-family: "Fraunces", Georgia, serif; font-weight: 600; font-size: 1.65rem;
  color: var(--ink); margin-top: 6px; line-height: 1.15; }
.bd-kpi-unit { font-family: "Source Sans 3", sans-serif; font-weight: 500; font-size: .8rem;
  color: var(--ink-2); margin-left: 4px; }
.bd-kpi-context { font-size: .78rem; color: var(--ink-2); margin-top: 4px; line-height: 1.25;
  min-height: 2.1em; }
.bd-kpi-visual { margin-top: auto; padding-top: 6px; min-height: 34px; display: flex; align-items: flex-end; }
@media print {
  [data-testid="stSidebar"], header[data-testid="stHeader"],
  [data-testid="stToolbar"], button[data-baseweb="tab"] { display: none !important; }
  [data-testid="stAppViewContainer"] { margin-left: 0 !important; }
  .bd-header { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .bd-print-stamp { display: block !important; }
  /* Streamlit tabs already hide every panel but the active one via its
     own display:none -- print naturally captures just that one tab,
     the same thing the button below is for. */
}
.bd-print-stamp { display: none; color: var(--ink-2); font-size: .8rem; margin: 4px 0 16px; }
</style>
""", unsafe_allow_html=True)

STATS_PATH = Path(__file__).resolve().parent / "data" / "processed" / "stats.json"

LANG = st.session_state.get("lang", "en")


@st.cache_data
def get_stats() -> dict:
    return load_stats()


@st.cache_data
def get_map_html(lang: str, village: str | None) -> str:
    """A folium Map's own .render() is NOT idempotent -- calling it twice on
    the same object produces two different (and, via st_folium, two BROKEN)
    HTML documents; a plain JS ReferenceError inside the component's iframe
    was the symptom, confirmed by rendering the same object twice here and
    diffing the output. Streamlit reruns the script more than once on first
    load, and st_folium's own render call was hitting that same object
    every time under @st.cache_resource. Rendering once here, to a plain
    string cached with @st.cache_data (keyed on `lang` and `village` too,
    so switching either gets its own cached render), and embedding that
    string (below) is immune to it -- there's nothing left to re-render."""
    return make_map(lang=lang, village=village).get_root().render()


_last_updated = (
    datetime.datetime.fromtimestamp(STATS_PATH.stat().st_mtime).strftime("%d %B %Y")
    if STATS_PATH.exists() else "n/a"
)
st.markdown(f"""
<div class="bd-header">
  <span class="bd-tag">{t("Pilot Project · Gemeente Berg en Dal", "Pilotproject · Gemeente Berg en Dal")}</span>
  <h1>🛰️ {t("Berg en Dal — remote sensing pilot", "Berg en Dal — aardobservatie-pilot")}</h1>
  <p>
    {t(
      "Real satellite and LiDAR data for the Berg en Dal municipality — everything on this page was "
      "fetched live, nothing is simulated.",
      "Echte satelliet- en LiDAR-data voor de gemeente Berg en Dal — alles op deze pagina is live "
      "opgehaald, niets is gesimuleerd."
    )}
    <span class="bd-meta">{t("Data last refreshed", "Data laatst ververst")}: {_last_updated}</span>
  </p>
</div>
""", unsafe_allow_html=True)

if not STATS_PATH.exists():
    st.error(t(
        "No pipeline output found yet. Run `python run_pipeline.py`, then `src/fetch_cbs.py`, "
        "`src/fetch_brp.py` and `src/fetch_air_quality.py` from the project root first.",
        "Nog geen pipeline-output gevonden. Draai eerst `python run_pipeline.py`, daarna "
        "`src/fetch_cbs.py`, `src/fetch_brp.py` en `src/fetch_air_quality.py` vanuit de projectmap.",
    ))
    st.stop()

stats = get_stats()
opt = stats.get("optical_summer_2025", {})
change = stats.get("ndvi_change", {})
trend = stats.get("ndvi_trend", {})
cross = stats.get("sar_optical_cross_check", {})
flood = stats.get("flood_extent", {})
flood_event = stats.get("flood_event", {})
weather = stats.get("weather", {})
climate_corr = stats.get("climate_correlation", {})
crop_rotation = stats.get("crop_rotation", {})
ahn = stats.get("ahn", {})
landcover = stats.get("landcover_summer_2025", {}).get("class_pct", {})
cbs = stats.get("cbs", {})
brp = stats.get("brp", {})
air = stats.get("air_quality", {})
landcover_trend = stats.get("landcover_trend", {})
air_quality_trend = stats.get("air_quality_trend", {})
villages = stats.get("villages", {}).get("list", [])
soil = stats.get("soil", {})
cbs_trend = stats.get("cbs_trend", {})
forecast = stats.get("forecast", {})
landcover_change_map = stats.get("landcover_change_map", {})

with st.sidebar:
    st.markdown(f"### 🛰️ {t('Berg en Dal pilot', 'Berg en Dal-pilot')}")
    st.caption(t("Real satellite, LiDAR & registry data — nothing simulated.",
                 "Echte satelliet-, LiDAR- en registratiedata — niets gesimuleerd."))
    st.markdown(f"**{t('Language', 'Taal')}**")
    # No default= here: key="lang" already binds this widget to
    # st.session_state["lang"], which the top of this file sets before
    # the sidebar ever renders -- passing both is exactly the conflict
    # Streamlit's own widget-policy check warns about (confirmed live on
    # Streamlit Community Cloud: "created with a default value but also
    # had its value set via the Session State API").
    st.segmented_control(
        "Language", options=["en", "nl"], format_func=lambda k: {"en": "English", "nl": "Nederlands"}[k],
        label_visibility="collapsed", key="lang",
    )
    st.divider()
    st.markdown(f"**{t('Village / place', 'Plaats / kern')}**")
    _village_options = [t("All of Berg en Dal", "Heel Berg en Dal")] + [v["name"] for v in villages]
    selected_village = st.selectbox(
        t("Filter Field Explorer to:", "Perceelverkenner filteren op:"),
        options=_village_options, label_visibility="collapsed", key="village",
    )
    if selected_village != _village_options[0]:
        _v = next((v for v in villages if v["name"] == selected_village), None)
        if _v:
            st.caption(t(
                f"{_v['population']:,} residents · {_v['land_area_ha']:,.0f} ha" if _v["population"] else "",
                f"{_v['population']:,} inwoners · {_v['land_area_ha']:,.0f} ha" if _v["population"] else "",
            ))
        st.caption(t("Applies to the Overview and Field Explorer maps (other tabs stay municipality-wide).",
                     "Geldt voor de kaarten in Overzicht en Perceelverkenner (andere tabbladen blijven gemeentebreed)."))
    else:
        selected_village = None
    st.divider()
    st.markdown(f"**{t('Print / export', 'Afdrukken / exporteren')}**")
    if st.button("🖨️ " + t("Print this tab", "Print dit tabblad"), use_container_width=True):
        st.components.v1.html("<script>window.parent.print();</script>", height=0)
    st.caption(t(
        "Opens your browser's print dialog for whichever tab is open — a quick snapshot, but it can't "
        "pick layers or bake in a real legend/scale bar. For that, build a PDF report below.",
        "Opent het afdrukdialoogvenster van je browser voor het geopende tabblad — een snelle "
        "momentopname, maar zonder gekozen lagen of een echte legenda/schaalbalk. Bouw daarvoor "
        "hieronder een PDF-rapport.",
    ))
    with st.expander("📄 " + t("Build a PDF report", "Bouw een PDF-rapport")):
        st.caption(t(
            "Pick which real layers to include — each becomes its own page with a title, legend, north "
            "arrow, and a scale bar sized to that page's own real extent (the same idea as a QGIS/ArcGIS "
            "print composer), not a screenshot of the interactive map.",
            "Kies welke echte lagen erin komen — elke laag wordt een eigen pagina met titel, legenda, "
            "noordpijl en een schaalbalk op maat van die pagina's eigen echte oppervlak (hetzelfde idee "
            "als een QGIS/ArcGIS-afdrukcompositie), geen schermafbeelding van de interactieve kaart.",
        ))
        _report_selected = [key for key, label in REPORT_LAYER_CHOICES
                             if st.checkbox(label[0] if LANG == "en" else label[1], value=(key in ("true_color", "ndvi", "landcover")),
                                             key=f"report_layer_{key}")]
        if st.button(t("Generate PDF", "Genereer PDF"), use_container_width=True, disabled=not _report_selected):
            with st.spinner(t("Building report…", "Rapport wordt gebouwd…")):
                try:
                    _pdf_bytes = build_report_pdf(_report_selected, lang=LANG)
                    st.session_state["_report_pdf"] = _pdf_bytes
                except Exception as exc:
                    st.error(t(f"Report generation failed: {exc}", f"Rapport genereren mislukt: {exc}"))
        if st.session_state.get("_report_pdf"):
            st.download_button(
                "⬇️ " + t("Download report PDF", "Download rapport-PDF"),
                data=st.session_state["_report_pdf"],
                file_name=f"bergendal_rapport_{datetime.date.today().isoformat()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    st.divider()
    st.caption(t(
        "Data: Sentinel-1/2, Landsat, AHN LiDAR, KNMI, RIVM, CBS, BRP, ISRIC SoilGrids. See README.md.",
        "Data: Sentinel-1/2, Landsat, AHN LiDAR, KNMI, RIVM, CBS, BRP, ISRIC SoilGrids. Zie README.md.",
    ))

_print_scope = selected_village or t("Berg en Dal (whole municipality)", "Berg en Dal (hele gemeente)")
_print_date = datetime.date.today().strftime("%d %B %Y")
st.markdown(
    f'<div class="bd-print-stamp">{t("Printed", "Afgedrukt")} {_print_date} · '
    f'{t("Berg en Dal remote sensing pilot", "Berg en Dal aardobservatie-pilot")} · {_print_scope}</div>',
    unsafe_allow_html=True,
)

_span = (f"{trend['years'][0]}–{trend['years'][-1]}" if trend.get("years") else "n/a")
_n_years = (trend["years"][-1] - trend["years"][0] + 1) if trend.get("years") else 0
_pills = [
    ("calendar", f"{_n_years} {t('years', 'jaar')}", f"{t('of satellite data', 'aan satellietdata')} · {_span}"),
    ("grid", f"{brp.get('n_parcels', 0):,.0f}", t("farm parcels, individually clickable", "landbouwpercelen, elk afzonderlijk klikbaar")),
    ("satellite", t("5 sensors", "5 sensoren"), t("Sentinel-1/2, Landsat, AHN LiDAR, RIVM, KNMI", "Sentinel-1/2, Landsat, AHN LiDAR, RIVM, KNMI")),
    ("wave", f"{len(flood_event.get('timeline', []))} {t('dates', 'data')}", t("through the Jan 2024 flood event", "door de hoogwatergebeurtenis van jan. 2024")),
]
st.markdown(f"""
<div class="bd-stat-strip">
  {''.join(f'<div class="bd-stat-pill"><div class="bd-stat-icon">{icon_svg(ic)}</div>'
           f'<div><div class="bd-stat-num">{n}</div><div class="bd-stat-lbl">{l}</div></div></div>' for ic, n, l in _pills)}
</div>
""", unsafe_allow_html=True)


def m(section: dict, key: str, fmt: str = "{:,.0f}") -> str:
    v = section.get(key, {}).get("value")
    return fmt.format(v) if v is not None else "n/a"


# Split X/Y so the y-axis (a real quantity) can carry a light, recessive
# horizontal grid -- genuinely easier to read a value off -- without the
# x-axis (almost always an ordinal year: 20+ categories) turning into a
# busy vertical comb. Font matches the app's own body type (Source Sans
# 3) rather than Vega-Lite's default, so a chart never reads as a
# different, less-designed surface bolted onto the page around it.
_CHART_AXIS_BASE = dict(domainColor="#CBD3C1", labelColor="#51604F", titleColor="#51604F",
                        labelFont="Source Sans 3", titleFont="Source Sans 3", titleFontWeight=600)
CHART_AXIS_X_KW = dict(_CHART_AXIS_BASE, grid=False)
CHART_AXIS_Y_KW = dict(_CHART_AXIS_BASE, grid=True, gridColor="#E9ECE1", gridDash=[1, 0])


def _year_highlight_layers(df: pd.DataFrame, value_col: str, highlight_year: int | None) -> list:
    """A ring + light vertical guide at the Year Explorer's selected year --
    reused on both the NDVI and NDWI charts so picking a year in the
    widget below visibly ties back to both trends at once."""
    if highlight_year is None or highlight_year not in df["year"].values:
        return []
    sel = df[df["year"] == highlight_year]
    rule = alt.Chart(sel).mark_rule(color=COLOR_SELECTED, strokeWidth=1, strokeDash=[3, 3], opacity=0.6).encode(x="year:O")
    ring = alt.Chart(sel).mark_point(size=260, filled=False, strokeWidth=2.5, color=COLOR_SELECTED).encode(
        x="year:O", y=f"{value_col}:Q",
    )
    return [rule, ring]


def ndvi_trend_chart(trend: dict, highlight_year: int | None = None) -> alt.LayerChart:
    """22-year NDVI trend, coloured by sensor provenance (Landsat vs
    Sentinel-2 — a real methodological seam documented in the README, not
    noise to hide) with a direct callout on 2018: the documented European
    drought year, and this series' single lowest point — corroborating
    evidence the data is tracking something real, not an artifact.
    `highlight_year` rings whichever year the Year Explorer widget has
    selected, tying that widget back to the chart."""
    source_label = {"landsat": "Landsat (30m)", "sentinel2": "Sentinel-2 (10m)"}
    df = pd.DataFrame({
        "year": trend["years"],
        "NDVI": trend["mean_ndvi"],
        "source": [source_label.get(s, s) for s in trend["source"]],
        "platform": trend["platform"],
        "clear_pct": trend["clear_pct"],
    })

    base = alt.Chart(df).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("NDVI:Q", title=t("Mean NDVI (clear ground)", "Gemiddelde NDVI (onbewolkt)"), scale=alt.Scale(zero=False)),
    )
    line = base.mark_line(color="#B9C2AE", strokeWidth=1.5, interpolate="monotone")
    points = base.mark_point(size=75, filled=True).encode(
        color=alt.Color(
            "source:N", title=t("Source", "Bron"),
            scale=alt.Scale(domain=["Landsat (30m)", "Sentinel-2 (10m)"], range=[COLOR_LANDSAT, COLOR_SENTINEL2]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[
            alt.Tooltip("year:O", title=t("Year", "Jaar")),
            alt.Tooltip("NDVI:Q", format=".3f"),
            alt.Tooltip("source:N", title=t("Source", "Bron")),
            alt.Tooltip("platform:N", title=t("Platform", "Platform")),
            alt.Tooltip("clear_pct:Q", title=t("Clear ground %", "% onbewolkt"), format=".0f"),
        ],
    )
    layers = [line, points]
    if 2018 in trend["years"]:
        drought_df = df[df["year"] == 2018]
        marker = alt.Chart(drought_df).mark_point(
            size=180, filled=False, strokeWidth=2, color=COLOR_DROUGHT,
        ).encode(x="year:O", y="NDVI:Q")
        callout = alt.Chart(drought_df).mark_text(
            text=t("↓ 2018 drought", "↓ droogte 2018"), dy=18, fontSize=11, color=COLOR_DROUGHT, fontWeight="bold",
        ).encode(x="year:O", y="NDVI:Q")
        layers += [marker, callout]
    layers += _year_highlight_layers(df, "NDVI", highlight_year)

    return (
        alt.layer(*layers)
        .properties(height=280)
        .configure_view(strokeWidth=0)
        .configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)
    )


def ndwi_trend_chart(trend: dict, highlight_year: int | None = None) -> alt.LayerChart:
    """Same treatment as the NDVI trend -- source-coloured, same two
    validated hues -- for NDWI (green-NIR): consistently negative here
    since the AOI is mostly land, not water; a *less* negative value means
    relatively wetter/less-vegetated ground that year, not "more water.\""""
    source_label = {"landsat": "Landsat (30m)", "sentinel2": "Sentinel-2 (10m)"}
    df = pd.DataFrame({
        "year": trend["years"],
        "NDWI": trend["mean_ndwi"],
        "source": [source_label.get(s, s) for s in trend["source"]],
        "platform": trend["platform"],
    })
    base = alt.Chart(df).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("NDWI:Q", title=t("Mean NDWI (clear ground)", "Gemiddelde NDWI (onbewolkt)"), scale=alt.Scale(zero=False)),
    )
    line = base.mark_line(color="#B9C2AE", strokeWidth=1.5, interpolate="monotone")
    points = base.mark_point(size=75, filled=True).encode(
        color=alt.Color(
            "source:N", title=t("Source", "Bron"),
            scale=alt.Scale(domain=["Landsat (30m)", "Sentinel-2 (10m)"], range=[COLOR_LANDSAT, COLOR_SENTINEL2]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[alt.Tooltip("year:O", title=t("Year", "Jaar")), alt.Tooltip("NDWI:Q", format=".3f"),
                 alt.Tooltip("source:N", title=t("Source", "Bron")), alt.Tooltip("platform:N", title=t("Platform", "Platform"))],
    )
    layers = [line, points] + _year_highlight_layers(df, "NDWI", highlight_year)
    return (
        alt.layer(*layers)
        .properties(height=240)
        .configure_view(strokeWidth=0)
        .configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)
    )


def weather_year_chart(years: list, values: list, title: str, color: str, fmt: str = ".0f") -> alt.Chart:
    """One weather variable, one colour, bars -- small-multiple companion
    to the NDVI/NDWI trend above rather than a second y-axis on the same
    panel (a dual-axis chart invites reading a spurious correlation into
    two arbitrarily-scaled lines, the method's #1 flagged anti-pattern)."""
    df = pd.DataFrame({"year": years, "value": values})
    chart = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("value:Q", title=title),
        tooltip=[alt.Tooltip("year:O", title=t("Year", "Jaar")), alt.Tooltip("value:Q", title=title, format=fmt)],
    ).properties(height=180)
    return chart.configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)


def correlation_chart(correlations: dict) -> alt.Chart:
    """Pearson r per NDVI/NDWI-vs-weather pair -- a diverging value (-1..+1
    around a meaningful zero), so diverging colour (two hues + neutral
    midpoint), not a categorical or sequential scale."""
    nice_names = {
        "ndvi_vs_august_precip": t("NDVI vs August rainfall", "NDVI vs augustusneerslag"),
        "ndvi_vs_august_temp": t("NDVI vs August temperature", "NDVI vs augustustemperatuur"),
        "ndvi_vs_august_sunshine": t("NDVI vs August sunshine", "NDVI vs augustuszon"),
        "ndvi_vs_august_evapotranspiration": t("NDVI vs August evapotranspiration", "NDVI vs augustusverdamping"),
        "ndwi_vs_august_precip": t("NDWI vs August rainfall", "NDWI vs augustusneerslag"),
        "ndwi_vs_august_temp": t("NDWI vs August temperature", "NDWI vs augustustemperatuur"),
    }
    rows = [{"pair": nice_names.get(k, k), "r": v} for k, v in correlations.items() if v is not None]
    df = pd.DataFrame(rows).sort_values("r")
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=18).encode(
        x=alt.X("r:Q", title="Pearson r (-1 … +1)", scale=alt.Scale(domain=[-1, 1])),
        y=alt.Y("pair:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#2a78d6", "#c9c9c2", "#e34948"]), legend=None),
        tooltip=[alt.Tooltip("pair:N", title=t("Pair", "Paar")), alt.Tooltip("r:Q", title="Pearson r", format="+.2f")],
    ).properties(height=26 * len(df) + 20)
    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#51604F", strokeWidth=1).encode(x="x:Q")
    return alt.layer(chart, zero_line).configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)


# Toned down from a first pass (#008300/#eda100/#e34948/#2a78d6 -- bright,
# "default chart" hues that clashed with the rest of this app's muted
# earth-tone palette) to the same colour families as LANDCOVER_COLORS in
# visualize.py (KMeans's own 6-class breakdown): forest green anchored on
# this app's own --forest, farmland gold matching the KMeans farmland
# colour, built-up a muted terracotta (still clearly its own hue, not
# reused from anything else on this chart), water this app's own --river.
# Kept as its own dict (not literally LANDCOVER_COLORS) because this is a
# different, coarser 4-category breakdown -- the year-over-year trend --
# not the raw 6-cluster KMeans output the other chart shows.
LANDCOVER_TREND_COLORS = {
    "Forest / dense vegetation": "#2E5943",
    "Agriculture": "#C9A227",
    "Built-up": "#B5524B",
    "Water": "#3B6E8A",
}


def landcover_trend_chart(years: list, values: list, category: str) -> alt.Chart:
    """One land-cover category's area over time -- small multiple (own
    y-axis, own colour from the family palette already used elsewhere),
    not stacked/normalized: stacking four series with wildly different
    absolute scale (forest ~5800ha vs built-up ~350ha) would visually
    flatten the smaller ones to a sliver."""
    df = pd.DataFrame({"year": years, "ha": values})
    color = LANDCOVER_TREND_COLORS.get(category, "#51604F")
    chart = alt.Chart(df).mark_area(
        line={"color": color, "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color=color, offset=0), alt.GradientStop(color="#ffffff", offset=1)],
            x1=1, x2=1, y1=1, y2=0,
        ),
        opacity=0.55, interpolate="monotone",
    ).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("ha:Q", title="ha", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("year:O", title=t("Year", "Jaar")), alt.Tooltip("ha:Q", title="ha", format=",.0f")],
    ).properties(height=150)
    return chart.configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)


def forecast_chart(history_years: list, history_values: list, future_years: list, future_values: list,
                    lower80: list, upper80: list, y_title: str, color: str, value_fmt: str = ",.0f",
                    height: int = 220) -> alt.LayerChart:
    """Historical points/line in solid colour, then a dashed line +
    shaded 80% prediction band for the forecast years -- dashed always
    reads as "projected, not observed" without needing a second legend,
    and the band (not the line) is deliberately the visually dominant
    element, since the band is the honest answer here, not the point
    estimate. One shared helper for every forecast.py series (NDVI/NDWI,
    each land-cover category, population, housing stock) rather than
    bespoke chart code per metric."""
    hist_df = pd.DataFrame({"year": history_years, "value": history_values})
    fut_df = pd.DataFrame({"year": future_years, "value": future_values})
    # Bridges the last real point to the first forecast point so the
    # dashed segment starts exactly where the solid one ends, instead of
    # a visual gap at the history/forecast seam.
    bridge_df = pd.concat([hist_df.tail(1), fut_df], ignore_index=True)
    band_df = pd.DataFrame({"year": future_years, "lower": lower80, "upper": upper80}).dropna()

    band = alt.Chart(band_df).mark_area(color=color, opacity=0.16).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("lower:Q", title=y_title, scale=alt.Scale(zero=False)), y2="upper:Q",
    )
    hist_line = alt.Chart(hist_df).mark_line(color=color, strokeWidth=2, interpolate="monotone").encode(x="year:O", y="value:Q")
    hist_points = alt.Chart(hist_df).mark_point(size=55, filled=True, color=color).encode(
        x="year:O", y="value:Q",
        tooltip=[alt.Tooltip("year:O", title=t("Year", "Jaar")), alt.Tooltip("value:Q", title=t("Observed", "Waargenomen"), format=value_fmt)],
    )
    fut_line = alt.Chart(bridge_df).mark_line(color=color, strokeWidth=2, strokeDash=[5, 3], interpolate="monotone").encode(x="year:O", y="value:Q")
    fut_points = alt.Chart(fut_df).mark_point(size=55, filled=False, strokeWidth=2, color=color).encode(
        x="year:O", y="value:Q",
        tooltip=[alt.Tooltip("year:O", title=t("Year", "Jaar")), alt.Tooltip("value:Q", title=t("Forecast", "Voorspelling"), format=value_fmt)],
    )
    return (
        alt.layer(band, hist_line, fut_line, hist_points, fut_points)
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)
    )


def crop_family_forecast_chart(current_ha: dict, projected_ha: dict) -> alt.Chart:
    """Current (2025) vs. projected-next-year hectares per crop family,
    grouped bars -- two series (a real legend, not colour-by-rank), the
    categorical identity (family) on the axis rather than encoded in
    colour twice. `projected_ha` comes straight from forecast.py's Markov
    transition matrix applied to every field's own current family and
    area; see the Field Explorer tab's own colour mode for the
    per-field version of the same prediction."""
    i = 0 if LANG == "en" else 1
    families = sorted(
        set(current_ha) | set(projected_ha),
        key=lambda f: -(current_ha.get(f, 0) + projected_ha.get(f, 0)),
    )
    label_of = {f: f"{CROP_FAMILIES[f][1]} {CROP_FAMILIES[f][2 + i]}" for f in families if f in CROP_FAMILIES}
    series_now, series_next = t("This year", "Dit jaar"), t("Projected next year", "Voorspeld volgend jaar")
    rows = []
    for fam in families:
        label = label_of.get(fam, fam)
        rows.append({"family": label, "series": series_now, "ha": current_ha.get(fam, 0.0)})
        rows.append({"family": label, "series": series_next, "ha": projected_ha.get(fam, 0.0)})
    df = pd.DataFrame(rows)
    order = [label_of.get(f, f) for f in families]
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("ha:Q", title=t("Hectares", "Hectare")),
        y=alt.Y("family:N", title=None, sort=order, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("series:N", sort=[series_now, series_next]),
        color=alt.Color(
            "series:N", title=None,
            scale=alt.Scale(domain=[series_now, series_next], range=[COLOR_SENTINEL2, COLOR_SELECTED]),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[alt.Tooltip("family:N", title=t("Family", "Familie")), alt.Tooltip("series:N", title=None),
                 alt.Tooltip("ha:Q", title=t("Hectares", "Hectare"), format=",.0f")],
    ).properties(height=30 * len(families) + 40)
    return chart.configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)


def air_quality_trend_chart(years: list, values: list, pollutant: str, color: str) -> alt.Chart:
    """One pollutant's annual mean over time -- same small-multiple
    treatment as the weather charts, plus a highlight ring on 2020 (the
    documented COVID-19 lockdown traffic dip) when that year is present."""
    df = pd.DataFrame({"year": years, "value": values})
    base = alt.Chart(df).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("value:Q", title="µg/m³", scale=alt.Scale(zero=False)),
    )
    line = base.mark_line(color=color, strokeWidth=2, interpolate="monotone")
    points = base.mark_point(size=55, filled=True, color=color).encode(
        tooltip=[alt.Tooltip("year:O", title=t("Year", "Jaar")), alt.Tooltip("value:Q", title=pollutant, format=".1f")],
    )
    layers = [line, points]
    if 2020 in years:
        covid_df = df[df["year"] == 2020]
        ring = alt.Chart(covid_df).mark_point(size=160, filled=False, strokeWidth=2, color=COLOR_DROUGHT).encode(
            x="year:O", y="value:Q",
        )
        layers.append(ring)
    return (
        alt.layer(*layers).properties(height=170)
        .configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)
    )


def village_bar_chart(villages: list, value_key: str, title: str, color: str, fmt: str = ",.0f") -> alt.Chart:
    """One metric (population or area) per village, ranked -- a plain
    identity axis (the village name) with one accent colour, since there's
    no shared legend across this and a second village chart to keep
    consistent."""
    df = pd.DataFrame({
        "village": [v["name"] for v in villages],
        "value": [v.get(value_key) or 0 for v in villages],
    }).sort_values("value", ascending=True)
    chart = alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=3, height=16).encode(
        x=alt.X("value:Q", title=title),
        y=alt.Y("village:N", title=None, sort=None),
        tooltip=[alt.Tooltip("village:N", title=t("Village", "Kern")), alt.Tooltip("value:Q", title=title, format=fmt)],
    ).properties(height=22 * len(df) + 20)
    return chart.configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)


def landcover_class_chart(landcover: dict) -> alt.Chart:
    """Each KMeans land-cover class in its own real colour (LANDCOVER_COLORS,
    shared with the interactive map's own legend so a bar here and that
    layer's fill on the map always agree) and its translated label --
    replacing a plain default-blue st.bar_chart that carried none of that.
    Colour is doubly-redundant with the y-axis identity here (each class
    already has its own row/label) -- a real design choice, not an
    oversight: no separate Altair legend is drawn, since the icon+colour+
    label row rendered under this chart (see call site) already serves
    that purpose without a second, cluttered legend saying the same thing."""
    names = list(landcover.keys())
    df = pd.DataFrame({
        "class_en": names,
        "class_label": [landcover_label(n, LANG) for n in names],
        "pct": list(landcover.values()),
    }).sort_values("pct", ascending=True)
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=18).encode(
        x=alt.X("pct:Q", title=t("% of clear ground", "% onbewolkte grond")),
        y=alt.Y("class_label:N", title=None, sort=None),
        color=alt.Color("class_en:N",
                         scale=alt.Scale(domain=list(LANDCOVER_COLORS.keys()), range=list(LANDCOVER_COLORS.values())),
                         legend=None),
        tooltip=[alt.Tooltip("class_label:N", title=t("Class", "Klasse")),
                 alt.Tooltip("pct:Q", title=t("% of clear ground", "% onbewolkte grond"), format=".1f")],
    ).properties(height=26 * len(df) + 20)
    return chart.configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)


def rotation_transitions_chart(top_transitions: dict) -> alt.Chart:
    """Top crop-to-crop transitions across the matched 2020-2025 window --
    one categorical axis (each transition is its own identity, no shared
    hue needed since there's no cross-chart legend to keep consistent) so
    a single accent colour is enough; count carries the magnitude."""
    df = pd.DataFrame({"pair": list(top_transitions.keys()), "count": list(top_transitions.values())})
    df = df.sort_values("count", ascending=True)
    chart = alt.Chart(df).mark_bar(color=COLOR_SENTINEL2, cornerRadiusEnd=3, height=18).encode(
        x=alt.X("count:Q", title=t("Fields", "Percelen")),
        y=alt.Y("pair:N", title=None, sort=None, axis=alt.Axis(labelLimit=340)),
        tooltip=[alt.Tooltip("pair:N", title=t("Transition", "Overgang")), alt.Tooltip("count:Q", title=t("Fields", "Percelen"))],
    ).properties(height=26 * len(df) + 20)
    return chart.configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)


def crop_family_chart(gdf: gpd.GeoDataFrame) -> alt.Chart:
    """Every BRP parcel sorted into one of 8 crop families (+ other),
    summed to hectares — colour-by-family (identity, per CROP_FAMILIES'
    validated palette) with the icon folded directly into the axis label,
    so the mark's colour and its name are both visible without a separate
    legend box (a categorical axis already names each row)."""
    i = 0 if LANG == "en" else 1
    fam = gdf.apply(lambda r: classify_crop(r["gewas"], r["category"]), axis=1)
    area_by_fam = gdf.groupby(fam)["area_ha"].sum()
    df = pd.DataFrame({"family": area_by_fam.index, "ha": area_by_fam.values})
    df["label"] = df["family"].map(lambda f: f"{CROP_FAMILIES[f][1]} {CROP_FAMILIES[f][2 + i]}")
    df["color"] = df["family"].map(lambda f: CROP_FAMILIES[f][0])
    df = df.sort_values("ha", ascending=True)

    chart = alt.Chart(df).mark_bar(height=18, cornerRadiusEnd=3).encode(
        x=alt.X("ha:Q", title=t("Hectares", "Hectare")),
        y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color("label:N", scale=alt.Scale(domain=df["label"].tolist(), range=df["color"].tolist()), legend=None),
        tooltip=[alt.Tooltip("label:N", title=t("Family", "Familie")), alt.Tooltip("ha:Q", title=t("Hectares", "Hectare"), format=",.0f")],
    ).properties(height=26 * len(df) + 20)
    return chart.configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)


def brp_category_chart(cat_ha: dict) -> alt.Chart:
    """BRP's own official land-use category per hectare, coloured by
    BRP_COLORS -- the same colours the map's own "Category (BRP)" layer
    and legend use, so this chart and that map layer never disagree.
    Category names are real BRP registry vocabulary (Grasland, Bouwland,
    ...), left untranslated like every other registry name in this app
    (crop names, family labels' underlying key) -- only the axis title
    and tooltip label route through t()."""
    df = pd.DataFrame({"category": list(cat_ha.keys()), "ha": list(cat_ha.values())})
    df = df.sort_values("ha", ascending=True)
    chart = alt.Chart(df).mark_bar(height=18, cornerRadiusEnd=3).encode(
        x=alt.X("ha:Q", title="ha"),
        y=alt.Y("category:N", title=None, sort=None),
        color=alt.Color("category:N",
                         scale=alt.Scale(domain=list(BRP_COLORS.keys()), range=list(BRP_COLORS.values())),
                         legend=None),
        tooltip=[alt.Tooltip("category:N", title=t("Category", "Categorie")), alt.Tooltip("ha:Q", title="ha", format=",.0f")],
    ).properties(height=26 * len(df) + 20)
    return chart.configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)


def top_crops_chart(crops: dict) -> alt.Chart:
    """The top individual crop names by area, coloured by which
    CROP_FAMILIES family each one belongs to (via the same classify_crop()
    the map's own crop-family layer uses) -- so a reader can see at a
    glance which family dominates the top-crops list without the two
    charts (this one and crop_family_chart above) using unrelated
    colours for the same underlying grouping. Crop names are real BRP
    registry vocabulary, left untranslated like elsewhere in this app."""
    df = pd.DataFrame({"crop": list(crops.keys()), "ha": list(crops.values())})
    df["family"] = df["crop"].map(lambda c: classify_crop(c, ""))
    df["color"] = df["family"].map(lambda f: CROP_FAMILIES[f][0])
    df = df.sort_values("ha", ascending=True)
    chart = alt.Chart(df).mark_bar(height=18, cornerRadiusEnd=3).encode(
        x=alt.X("ha:Q", title="ha"),
        y=alt.Y("crop:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("crop:N", scale=alt.Scale(domain=df["crop"].tolist(), range=df["color"].tolist()), legend=None),
        tooltip=[alt.Tooltip("crop:N", title=t("Crop", "Gewas")), alt.Tooltip("ha:Q", title="ha", format=",.0f")],
    ).properties(height=26 * len(df) + 20)
    return chart.configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)


def flood_timeline_chart(flood_event: dict) -> alt.LayerChart:
    """Flooded-ground share through the event -- one series (a magnitude
    over time), so phases show up via tooltip rather than a second legend
    (a lone series needs none, per the dataviz method)."""
    df = pd.DataFrame(flood_event["timeline"])
    df["date"] = pd.to_datetime(df["date"])

    base = alt.Chart(df).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", labelAngle=0)),
        y=alt.Y("flooded_pct:Q", title=t("Flagged as flooded (% of AOI)", "Gemarkeerd als overstroomd (% van gebied)"), scale=alt.Scale(zero=True)),
    )
    area = base.mark_area(
        line={"color": COLOR_WATER, "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color=COLOR_WATER, offset=0), alt.GradientStop(color="#ffffff", offset=1)],
            x1=1, x2=1, y1=1, y2=0,
        ),
        opacity=0.5, interpolate="monotone",
    )
    points = base.mark_point(size=90, filled=True, color=COLOR_WATER).encode(
        tooltip=[
            alt.Tooltip("date:T", title=t("Date", "Datum"), format="%Y-%m-%d"),
            alt.Tooltip("phase:N", title=t("Phase", "Fase")),
            alt.Tooltip("flooded_pct:Q", title=t("Flooded %", "% overstroomd"), format=".1f"),
            alt.Tooltip("mean_db_vs_reference:Q", title=t("dB vs. reference", "dB t.o.v. referentie"), format="+.2f"),
        ],
    )
    return (
        alt.layer(area, points)
        .properties(height=260)
        .configure_view(strokeWidth=0)
        .configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW)
    )


tab_overview, tab_explorer, tab_villages, tab_land, tab_climate, tab_forecast, tab_env, tab_water, \
    tab_methodology, tab_business = st.tabs([
    t("Overview", "Overzicht"),
    t("Field Explorer", "Perceelverkenner"),
    t("Villages", "Kernen"),
    t("Land & Crops", "Land & Gewassen"),
    t("Trends & Climate", "Trends & Klimaat"),
    t("Forecast", "Voorspelling"),
    t("Environment & Energy", "Milieu & Energie"),
    t("Water", "Water"),
    t("Methodology", "Methodologie"),
    t("Business case", "Businesscase"),
])
METHOD_BY_ID = {e.id: e for e in METHOD_ENTRIES}


def method_popover(entry_id: str) -> None:
    """The 'info icon that links to its entry' from every KPI card and map
    legend -- as a real, always-working st.popover rather than a link to
    the Methodology tab: Streamlit's st.tabs() can't be switched to from a
    link/click (there's no supported API for it, and an anchor pointing
    into a different tab's panel lands on an element hidden by Streamlit's
    own display:none, which a browser can't usefully scroll to) -- so
    "clicking it takes you straight to the real content" is delivered
    inline, on the spot, rather than promising a cross-tab jump this
    Streamlit version doesn't support."""
    e = METHOD_BY_ID.get(entry_id)
    if e is None:
        return
    i = 0 if LANG == "en" else 1
    with st.popover("ℹ️", help=t("How this is calculated", "Hoe dit wordt berekend")):
        st.markdown(f"**{e.title[i]}**")
        st.caption(e.measures[i])
        st.markdown(f"**{t('Source', 'Bron')}:** {e.source}")
        st.markdown(f"**{t('Date/period', 'Datum/periode')}:** {e.date_range[i]}")
        st.markdown(f"**{t('Resolution', 'Resolutie')}:** {e.resolution[i]}")
        st.markdown(f"**{t('Processing', 'Verwerking')}:**")
        for step in e.processing[i]:
            st.caption(f"• {step}")
        st.markdown(f"**{t('Limitations', 'Beperkingen')}:**")
        for lim in e.limitations[i]:
            st.caption(f"• {lim}")
        if e.open_questions[i]:
            st.markdown(f"**{t('Open questions', 'Open vragen')}:**")
            for q in e.open_questions[i]:
                st.caption(f"• {q}")
        st.caption(f"{t('Code', 'Code')}: `{e.code_ref}` — {t('full write-up on the Methodology tab', 'volledige uitleg op het tabblad Methodologie')}.")

# ======================================================================
with tab_overview:
    st.caption(t("Click any card's ⓘ for its method, or More for extra detail.",
                  "Klik op ⓘ voor de methode, of Meer voor extra detail."))
    _cbs_year = cbs.get("year") or "?"
    _brp_year = brp.get("year") or "?"
    _pop_series = [v for v in (cbs_trend.get("population") or []) if v is not None]
    _hh_series = [v for v in (cbs_trend.get("households") or []) if v is not None]
    _pop_delta = (_pop_series[-1] - _pop_series[0]) if len(_pop_series) >= 2 else None
    _hh_delta = (_hh_series[-1] - _hh_series[0]) if len(_hh_series) >= 2 else None
    _cy0 = (cbs_trend.get("years") or [None])[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card(
            t("Population", "Inwoners"), str(_cbs_year), m(cbs, "population"), t("residents", "inwoners"),
            (f"+{_pop_delta:,.0f} {t('since', 'sinds')} {_cy0}" if _pop_delta else ""),
            kpi_sparkline_svg(cbs_trend.get("population") or []),
        )
        bc1, bc2 = st.columns([3, 1])
        with bc1, st.popover(t("More", "Meer"), use_container_width=True):
            st.markdown(t(
                f"**{m(cbs, 'population_density')} residents/km²** — "
                f"{m(cbs, 'population_male')} male, {m(cbs, 'population_female')} female.\n\n"
                f"Age: {m(cbs,'population_under15_pct')}% under 15, {m(cbs,'population_25to45_pct')}% "
                f"25–45, {m(cbs,'population_65plus_pct')}% 65+.\n\n"
                f"Last year: {m(cbs,'births_last_year')} births, {m(cbs,'deaths_last_year')} deaths — "
                "natural decline, offset by migration (the population is still real, positive net "
                "growth — see the growth figure on Villages/Forecast).",
                f"**{m(cbs, 'population_density')} inwoners/km²** — "
                f"{m(cbs, 'population_male')} man, {m(cbs, 'population_female')} vrouw.\n\n"
                f"Leeftijd: {m(cbs,'population_under15_pct')}% onder 15, {m(cbs,'population_25to45_pct')}% "
                f"25–45, {m(cbs,'population_65plus_pct')}% 65+.\n\n"
                f"Afgelopen jaar: {m(cbs,'births_last_year')} geboorten, {m(cbs,'deaths_last_year')} "
                "sterfgevallen — natuurlijke krimp, gecompenseerd door migratie (de bevolking groeit nog "
                "steeds netto — zie het groeicijfer bij Kernen/Voorspelling).",
            ))
        with bc2:
            method_popover("population")
    with c2:
        kpi_card(
            t("Households", "Huishoudens"), str(_cbs_year), m(cbs, "households"), t("households", "huishoudens"),
            (f"+{_hh_delta:,.0f} {t('since', 'sinds')} {_cy0}" if _hh_delta else ""),
            kpi_sparkline_svg(cbs_trend.get("households") or []),
        )
        bc1, bc2 = st.columns([3, 1])
        with bc1, st.popover(t("More", "Meer"), use_container_width=True):
            st.markdown(t(
                f"**{m(cbs, 'avg_household_size', '{:.1f}')} people/household** on average.\n\n"
                f"{m(cbs,'single_person_households_pct')}% single-person, "
                f"{m(cbs,'households_with_children_pct')}% with children.\n\n"
                f"{m(cbs, 'new_homes_last_year', '{:.0f}')} new homes last year — "
                f"{m(cbs, 'housing_stock')} total housing stock.",
                f"**{m(cbs, 'avg_household_size', '{:.1f}')} personen/huishouden** gemiddeld.\n\n"
                f"{m(cbs,'single_person_households_pct')}% eenpersoons, "
                f"{m(cbs,'households_with_children_pct')}% met kinderen.\n\n"
                f"{m(cbs, 'new_homes_last_year', '{:.0f}')} nieuwe woningen afgelopen jaar — "
                f"{m(cbs, 'housing_stock')} woningvoorraad totaal.",
            ))
        with bc2:
            method_popover("households")
    with c3:
        _land_ha = (cbs.get("land_area_ha") or {}).get("value") or 0
        _farm_ha = brp.get("total_area_ha", 0)
        _farm_rest_ha = max(0.0, _land_ha - _farm_ha)
        _farm_pct = (_farm_ha / _land_ha * 100) if _land_ha else 0
        kpi_card(
            t("Registered farmland", "Landbouwgrond"), str(_brp_year), f"{_farm_ha:,.0f}", "ha",
            f"{brp.get('n_parcels', 0):,} " + t("parcels", "percelen") +
            f" · {_farm_pct:.0f}% " + t("of the municipality's land", "van de landoppervlakte"),
            # composition, not a fake trend -- real farmland ha vs the
            # rest of the municipality's land, both shares of one total
            kpi_stacked_bar_svg([(_farm_ha, KPI_INK), (_farm_rest_ha, "#CBD3C1")]),
        )
        bc1, bc2 = st.columns([3, 1])
        with bc1, st.popover(t("More", "Meer"), use_container_width=True):
            st.markdown(t("**Top crops by area:**", "**Grootste gewassen naar oppervlakte:**"))
            for crop, ha in list(brp.get("top_crops_ha", {}).items())[:5]:
                st.caption(f"{crop} — {ha:,.0f} ha")
        with bc2:
            method_popover("registered_farmland")
    with c4:
        _water_ha = (cbs.get("water_area_ha") or {}).get("value") or 0
        _terr_ha = _land_ha + _water_ha
        _water_pct = (_water_ha / _terr_ha * 100) if _terr_ha else 0
        kpi_card(
            t("Land area", "Landoppervlakte"), str(_cbs_year), m(cbs, "land_area_ha"), "ha",
            f"{m(cbs, 'water_area_ha')} ha " + t("water", "water") + f" · {_water_pct:.0f}%",
            # composition -- land vs water share of the municipality's
            # full territory, not the land figure alone
            kpi_stacked_bar_svg([(_land_ha, KPI_INK), (_water_ha, "#3B6E8A")]),
        )
        bc1, bc2 = st.columns([3, 1])
        with bc1, st.popover(t("More", "Meer"), use_container_width=True):
            st.markdown(t(
                f"**{m(cbs, 'water_area_ha')} ha** of that is water (Rhine/Waal floodplain) — not "
                "counted in the land-area figure shown.\n\n"
                f"{m(cbs, 'population_density')} residents/km² overall — see Villages for the "
                "per-place breakdown.",
                f"**{m(cbs, 'water_area_ha')} ha** daarvan is water (uiterwaarden van Rijn/Waal) — niet "
                "meegeteld in het getoonde landoppervlaktecijfer.\n\n"
                f"{m(cbs, 'population_density')} inwoners/km² gemeentebreed — zie Kernen voor de "
                "uitsplitsing per plaats.",
            ))
        with bc2:
            method_popover("land_area")
    with c5:
        _solar_pct = (cbs.get("homes_with_solar_pct") or {}).get("value") or 0
        kpi_card(
            t("Homes with solar", "Zonnepanelen"), str(_cbs_year), f"{_solar_pct:.0f}", "%",
            f"{m(cbs, 'gas_free_homes_pct', '{:.0f}')}% " + t("fully gas-free", "volledig aardgasvrij"),
            kpi_proportion_bar_svg(_solar_pct),
        )
        bc1, bc2 = st.columns([3, 1])
        with bc1, st.popover(t("More", "Meer"), use_container_width=True):
            st.markdown(t(
                f"Only **{m(cbs, 'gas_free_homes_pct', '{:.0f}')}% gas-free** — the remaining energy "
                "transition is heating, not solar adoption.\n\n"
                f"Avg. electricity use: {m(cbs, 'avg_electricity_use_kwh', '{:.0f}')} kWh/yr · avg. solar "
                f"feed-in: {m(cbs, 'avg_solar_feedback_kwh', '{:.0f}')} kWh/yr.",
                f"Slechts **{m(cbs, 'gas_free_homes_pct', '{:.0f}')}% aardgasvrij** — de resterende "
                "energietransitie gaat over verwarming, niet over zonneadoptie.\n\n"
                f"Gem. elektriciteitsverbruik: {m(cbs, 'avg_electricity_use_kwh', '{:.0f}')} kWh/jr · gem. "
                f"teruglevering zon: {m(cbs, 'avg_solar_feedback_kwh', '{:.0f}')} kWh/jr.",
            ))
        with bc2:
            method_popover("homes_with_solar")

    st.divider()
    st.subheader(t("Map", "Kaart"))
    # Static embed, not st_folium -- this map has no click callback to wire
    # up (returned_objects was always []), and a plain iframe embed of one
    # fixed HTML string sidesteps the render-not-idempotent bug documented
    # on get_map_html() above. Full-width, tall: this is the centrepiece,
    # not a sidebar thumbnail -- every layer (true colour, NDVI, land cover,
    # both flood-extent methods, SAR, elevation, BRP fields...) toggles from
    # its own control at top-right, translated with the rest of the page.
    if selected_village:
        st.caption("📍 " + t(f"Zoomed to **{selected_village}** (outlined) — change in the sidebar.",
                              f"Ingezoomd op **{selected_village}** (omlijnd) — wijzig in de zijbalk."))
    try:
        st.components.v1.html(get_map_html(LANG, selected_village), height=760)
    except Exception as exc:
        # A deploy that only has data/processed/stats.json (the small,
        # committed subset) and not the full satellite/LiDAR raster
        # archive (data/raw, data/processed/*.tif -- deliberately kept out
        # of git, see .gitignore) can't render the map layers, which read
        # those rasters/vector files directly. Everything else on this
        # page (KPI cards, every chart, Methodology, Business case) only
        # needs stats.json and still works -- so this degrades one
        # section instead of crashing the whole app, and says exactly why.
        # Deliberately broad, not the narrower (FileNotFoundError,
        # RasterioIOError) tried first: a real Streamlit Cloud deploy hit
        # this exact code path missing brp_parcels.geojson and geopandas'
        # pyogrio backend raised pyogrio.errors.DataSourceError instead --
        # a different exception per missing-file *type* (raster vs.
        # vector), and there's no guarantee it's the last one. This
        # try block only ever runs code that reads local data files with
        # no other side effects, so catching broadly here means "any
        # reason this specific read failed," not "swallow unrelated bugs."
        st.info(t(
            f"The interactive map needs the full satellite/LiDAR raster archive, which this "
            f"deployment doesn't have (only the small `stats.json` summary is included here) — "
            f"run `python run_pipeline.py` locally for the full map. ({exc.__class__.__name__})",
            f"De interactieve kaart heeft het volledige satelliet-/LiDAR-archief nodig, dat deze "
            f"deployment niet heeft (alleen de kleine `stats.json`-samenvatting is hier "
            f"meegenomen) — draai `python run_pipeline.py` lokaal voor de volledige kaart. "
            f"({exc.__class__.__name__})",
        ))

    st.divider()
    st.subheader(t("Land cover (KMeans, satellite-derived)", "Landgebruik (KMeans, satellietafgeleid)"))
    if landcover:
        st.altair_chart(landcover_class_chart(landcover), use_container_width=True)
        _lc_legend_rows = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;margin:2px 12px 2px 0;">'
            f'<span style="width:16px;height:16px;border-radius:50%;background:{LANDCOVER_COLORS.get(n,"#999")};'
            f'display:inline-flex;align-items:center;justify-content:center;flex:none;">'
            f'{landcover_icon_svg(n, size=10, stroke="#fff")}</span>'
            f'<span style="font-size:12px;color:var(--ink-2);">{landcover_label(n, LANG)}</span></span>'
            for n in landcover.keys()
        )
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;margin-top:2px;">{_lc_legend_rows}</div>',
                    unsafe_allow_html=True)
    st.caption(t(
        f"Mean NDVI Aug 2025: {opt.get('mean_ndvi', 0):.3f} ({change.get('mean_delta', 0):+.3f} vs Aug 2024). "
        f"Elevation (AHN): {ahn.get('elevation_min_m', 0):.0f}–{ahn.get('elevation_max_m', 0):.0f} m NAP.",
        f"Gemiddelde NDVI aug. 2025: {opt.get('mean_ndvi', 0):.3f} ({change.get('mean_delta', 0):+.3f} t.o.v. "
        f"aug. 2024). Hoogte (AHN): {ahn.get('elevation_min_m', 0):.0f}–{ahn.get('elevation_max_m', 0):.0f} m NAP.",
    ))
    st.markdown(t(
        "**What's not validated yet:**\n"
        "- Land cover cluster names are a spectral best guess (no ground-truth join) — except where the "
        "Land & Crops tab now backs them with real BRP field registrations.\n"
        f"- SAR/optical water agreement: {cross.get('iou_pct', 0):.0f}% IoU — see the Water tab.",
        "**Nog niet gevalideerd:**\n"
        "- Namen van landgebruiksclusters zijn een spectrale inschatting (geen koppeling aan grondwaarheid) "
        "— behalve waar het tabblad Land & Gewassen ze onderbouwt met echte BRP-registraties.\n"
        f"- SAR/optische overeenstemming water: {cross.get('iou_pct', 0):.0f}% IoU — zie het tabblad Water.",
    ))

    if trend.get("years"):
        st.divider()
        st.subheader(f"{_n_years} " + t(f"years of data ({_span})", f"jaar data ({_span})"))
        r_temp_headline = climate_corr.get("correlations", {}).get("ndvi_vs_august_temp")
        st.info(t(
            f"Full NDVI/NDWI trend, real August weather back to {trend['years'][0]}, and how they line up "
            "— see the **Trends & Climate** tab. Headline: net "
            f"{trend.get('net_change', 0):+.3f} NDVI over {len(trend['years']) - 1} years, and August "
            "temperature is the single strongest weather correlate of that year's NDVI"
            + (f" (r ≈ {r_temp_headline:+.2f})" if r_temp_headline is not None else "") +
            " — hotter Augusts, lower vigour, physically the direction you'd expect.",
            f"Volledige NDVI/NDWI-trend, echte augustusweer-data terug tot {trend['years'][0]}, en hoe ze "
            "samenhangen — zie het tabblad **Trends & Klimaat**. Kernpunt: netto "
            f"{trend.get('net_change', 0):+.3f} NDVI over {len(trend['years']) - 1} jaar, en de "
            "augustustemperatuur is de sterkste weerscorrelatie met de NDVI van dat jaar"
            + (f" (r ≈ {r_temp_headline:+.2f})" if r_temp_headline is not None else "") +
            " — hetere augustusmaanden, minder vitaliteit, precies de fysiek verwachte richting.",
        ))

# ======================================================================
with tab_explorer:
    st.subheader(t("Every field, selectable", "Elk perceel selecteerbaar"))
    st.caption(t(
        "Pick what colours the fields, then click one for its details and how it's changed year over year.",
        "Kies waarop de percelen gekleurd worden, klik dan op een perceel voor details en de verandering "
        "sinds vorig jaar.",
    ))

    color_key = st.selectbox(
        t("Colour fields by:", "Percelen kleuren op:"),
        options=list(FIELD_COLOR_MODES.keys()),
        format_func=lambda k: FIELD_COLOR_MODES[k][0 if LANG == "en" else 1],
    )

    if selected_village:
        st.caption("📍 " + t(f"Showing **{selected_village}** only — change in the sidebar.",
                              f"Alleen **{selected_village}** getoond — wijzig in de zijbalk."))

    map_col, detail_col = st.columns([5, 2])
    map_state = None
    with map_col:
        try:
            field_map = field_explorer_map(color_by=color_key, lang=LANG, village=selected_village)
            map_state = st_folium(
                field_map, width=None, height=720,
                returned_objects=["last_object_clicked"], key=f"field_map_{color_key}_{LANG}_{selected_village}",
            )
        except Exception as exc:
            # Same real limitation as the Overview map above, and the same
            # deliberately broad except -- confirmed live on Streamlit
            # Community Cloud: this exact call site, missing
            # brp_parcels.geojson, raised pyogrio.errors.DataSourceError
            # (geopandas' vector-file backend), not FileNotFoundError or
            # RasterioIOError (rasterio's own exception, for the raster
            # side only) -- so the narrower catch tried first let this
            # crash straight through instead of degrading gracefully.
            st.info(t(
                f"Field Explorer's map needs the full satellite/LiDAR raster archive, which this "
                f"deployment doesn't have — run `python run_pipeline.py` locally for the full map. "
                f"({exc.__class__.__name__})",
                f"De kaart van Perceelverkenner heeft het volledige satelliet-/LiDAR-archief nodig, "
                f"dat deze deployment niet heeft — draai `python run_pipeline.py` lokaal voor de "
                f"volledige kaart. ({exc.__class__.__name__})",
            ))

    with detail_col:
        st.markdown(f"**{t('Selected field', 'Geselecteerd perceel')}**")
        click = (map_state or {}).get("last_object_clicked")
        row = field_at(click["lat"], click["lng"]) if click else None
        if row is not None:
            family = classify_crop(row["gewas"], row["category"])
            fam_color, fam_icon, fam_en, fam_nl = CROP_FAMILIES[family]
            st.markdown(f"#### {fam_icon} {row['gewas']}")
            st.markdown(
                f'<span style="display:inline-block;width:10px;height:10px;background:{fam_color};'
                f'border-radius:2px;margin-right:5px;"></span>{fam_en if LANG == "en" else fam_nl}',
                unsafe_allow_html=True,
            )
            st.write(f"**{t('Category', 'Categorie')}:** {row['category']}")
            st.write(f"**{t('Area', 'Oppervlakte')}:** {row['area_ha']:.2f} ha")
            ndvi25, ndvi24 = row.get("ndvi_2025"), row.get("ndvi_2024")
            if pd.notna(ndvi25) and pd.notna(ndvi24):
                _ndvi_delta = ndvi25 - ndvi24
                st.write(f"**NDVI 2025:** {ndvi25:.2f}")
                st.markdown(index_range_svg(ndvi25, -1, 1, low_color="#A2712F", high_color=KPI_INK, key="field-ndvi"),
                            unsafe_allow_html=True)
                st.write(f"**{t('NDVI change (2024→2025)', 'NDVI-verandering (2024→2025)')}:** {_ndvi_delta:+.2f}")
                # same brown-to-green convention as the map's own NDVI-change
                # legend (browning -> greening), so a field's number and its
                # colour on the map always agree.
                st.markdown(index_range_svg(_ndvi_delta, -1, 1, low_color="#A2712F", high_color=KPI_INK, key="field-ndvichg"),
                            unsafe_allow_html=True)
                _how_going = t("How it's going", "Hoe het ervoor staat")
                st.markdown(f"**{_how_going}**")
                # KPI_INK, not a default Streamlit blue -- the same forest
                # green every other NDVI visual on this page uses (the
                # range bars just above, the map's own NDVI layer), so this
                # small chart doesn't read as an unrelated colour system.
                _ndvi_hist_df = pd.DataFrame({"year": ["2024", "2025"], "NDVI": [ndvi24, ndvi25]})
                _ndvi_hist_chart = alt.Chart(_ndvi_hist_df).mark_bar(color=KPI_INK, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("year:N", title=None),
                    y=alt.Y("NDVI:Q", scale=alt.Scale(domain=[0, 1])),
                    tooltip=[alt.Tooltip("year:N", title=t("Year", "Jaar")), alt.Tooltip("NDVI:Q", format=".3f")],
                ).properties(height=140)
                st.altair_chart(
                    _ndvi_hist_chart.configure_view(strokeWidth=0).configure_axisX(**CHART_AXIS_X_KW).configure_axisY(**CHART_AXIS_Y_KW),
                    use_container_width=True,
                )
            else:
                st.caption(t(
                    "No NDVI trend for this field (cloud-masked in one of the two years).",
                    "Geen NDVI-trend voor dit perceel (bewolkt gemaskeerd in een van beide jaren).",
                ))

            rot_raw = row.get("rotation_json")
            if rot_raw is not None and (isinstance(rot_raw, dict) or (isinstance(rot_raw, str) and rot_raw)):
                rot = rot_raw if isinstance(rot_raw, dict) else json.loads(rot_raw)
                years_present = sorted(y for y, c in rot.items() if c)
                st.markdown(f"**{t('Crop rotation', 'Gewasrotatie')} ({years_present[0]}–{years_present[-1]})**" if years_present else "")
                prev_norm = None
                for y in sorted(rot.keys()):
                    crop = rot[y]
                    if crop is None:
                        st.caption(f"{y}: {t('not registered / no match this year', 'niet geregistreerd / geen match dit jaar')}")
                        prev_norm = None
                        continue
                    norm = normalize_crop(crop)
                    changed = prev_norm is not None and norm != prev_norm
                    marker = "🔄 " if changed else ""
                    st.write(f"{marker}**{y}:** {crop}")
                    prev_norm = norm
                if not row.get("rotation_changed", True) and row.get("rotation_n_years", 0) >= 2:
                    st.caption(t(
                        "No crop change across the years matched above.",
                        "Geen gewasverandering over de hierboven gematchte jaren.",
                    ))
                st.caption(t(
                    "Matched year-to-year by which historical parcel overlaps this one most (≥30% of its "
                    "area) — boundary redraws mean not every field matches every year. See Land & Crops for "
                    "the municipality-wide rotation picture and its caveats.",
                    "Jaar-op-jaar gematcht op het perceel dat het meest overlapt (≥30% van de oppervlakte) "
                    "— door herziene perceelgrenzen matcht niet elk veld elk jaar. Zie Land & Gewassen voor "
                    "het gemeentebrede rotatiebeeld en de kanttekeningen daarbij.",
                ))

            st.markdown(f"**{t('At this exact point', 'Op dit exacte punt')}**")
            samples = point_samples(click["lat"], click["lng"])
            s1, s2 = st.columns(2)
            elev = samples["elevation_m"]
            s1.metric(t("Elevation", "Hoogte"), f"{elev:.1f} m NAP" if elev is not None else "n/a")
            canopy = samples["canopy_height_m"]
            s2.metric(t("Canopy/roof height", "Bladerdak-/dakhoogte"), f"{canopy:.1f} m" if canopy is not None else "n/a")
            if samples["sar_vv_db"] is not None:
                water_txt = (" · " + t("flagged as open water", "gemarkeerd als open water")) if samples["sar_water"] else ""
                st.caption(t(
                    f"SAR backscatter (VV, Aug 2025): {samples['sar_vv_db']:.1f} dB{water_txt}",
                    f"SAR-terugkaatsing (VV, aug. 2025): {samples['sar_vv_db']:.1f} dB{water_txt}",
                ))
            else:
                st.caption(t(
                    "No SAR reading here (cloud/shadow-free radar has no gaps, but this AOI clip might not "
                    "extend to this point).",
                    "Geen SAR-waarde op dit punt (wolk-/schaduwvrije radar heeft geen gaten, maar deze "
                    "uitsnede reikt hier mogelijk niet tot).",
                ))
        elif click:
            st.warning(t(
                "That point isn't inside a registered BRP field.",
                "Dat punt ligt niet binnen een geregistreerd BRP-perceel.",
            ))
        else:
            st.info(t(
                "Click a field on the map to see its details here.",
                "Klik op een perceel op de kaart voor de details hier.",
            ))

    st.caption(t(
        "Legend by mode — **Category**: green tones for grassland/nature, brown for arable. "
        "**Crop family**: every one of the 102 real BRP crop names sorted into 8 families "
        "(🌱 grassland, 🌽 maize, 🌾 cereals, 🥔 root/tuber, 🥬 vegetables, 🍎 fruit, 🍀 cover/oilseed, "
        "🌳 nature/water) plus ❔ other — hover or click any field for its exact crop name and icon, or "
        "see the map's own legend (bottom-right) when this mode is selected. **NDVI 2025**: red→green, "
        "low→high vigour. **NDVI change**: brown→green, browning→greening since 2024. Fields with no valid "
        "pixel in one of the two dates (cloud-masked) show grey.",
        "Legenda per modus — **Categorie**: groentinten voor grasland/natuur, bruin voor bouwland. "
        "**Gewasfamilie**: alle 102 echte BRP-gewasnamen ingedeeld in 8 families (🌱 grasland, 🌽 mais, "
        "🌾 granen, 🥔 wortel-/knolgewassen, 🥬 groenten, 🍎 fruit, 🍀 groenbemesters/oliehoudend, "
        "🌳 natuur/water) plus ❔ overig — beweeg over of klik op een perceel voor de exacte gewasnaam en "
        "het icoon, of zie de eigen legenda van de kaart (rechtsonder) bij deze modus. **NDVI 2025**: "
        "rood→groen, laag→hoge vitaliteit. **NDVI-verandering**: bruin→groen, verbruining→vergroening "
        "sinds 2024. Percelen zonder geldige pixel in een van beide data (bewolkt) tonen grijs.",
    ))

# ======================================================================
with tab_villages:
    st.subheader(t("13 places, one municipality", "13 plaatsen, één gemeente"))
    st.caption(t(
        "Berg en Dal is a 2015 merger of the former municipalities Groesbeek, Millingen aan de Rijn and "
        "Ubbergen — it reads as 13 real villages day to day, not one blob. Real CBS district (\"wijk\") "
        "boundaries and population/area, not estimated. Pick one in the sidebar to filter the Field "
        "Explorer map to just that place.",
        "Berg en Dal is een fusie uit 2015 van de voormalige gemeenten Groesbeek, Millingen aan de Rijn "
        "en Ubbergen — het voelt dagelijks als 13 echte kernen, niet één geheel. Echte CBS-wijkgrenzen en "
        "bevolking/oppervlakte, niet geschat. Kies er één in de zijbalk om de kaart in Perceelverkenner "
        "tot die plaats te beperken.",
    ))

    if villages:
        vc1, vc2, vc3 = st.columns(3)
        vc1.metric(t("Villages", "Kernen"), f"{len(villages)}")
        vc2.metric(t("Largest", "Grootste"), villages[0]["name"],
                   delta=f"{villages[0]['population']:,} " + t("residents", "inwoners"), delta_color="off")
        _smallest = min(villages, key=lambda v: v["population"] or 0)
        vc3.metric(t("Smallest", "Kleinste"), _smallest["name"],
                   delta=f"{_smallest['population']:,} " + t("residents", "inwoners"), delta_color="off")

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown(f"**{t('Population', 'Inwoners')}**")
            st.altair_chart(village_bar_chart(villages, "population", t("Residents", "Inwoners"), COLOR_SENTINEL2), use_container_width=True)
        with vcol2:
            st.markdown(f"**{t('Land area', 'Landoppervlakte')}**")
            st.altair_chart(village_bar_chart(villages, "land_area_ha", "ha", COLOR_TEMP), use_container_width=True)

        st.markdown(f"**{t('All 13, in detail', 'Alle 13, in detail')}**")
        _v_df = pd.DataFrame(villages).rename(columns={
            "name": t("Village", "Kern"), "population": t("Residents", "Inwoners"),
            "households": t("Households", "Huishoudens"), "land_area_ha": t("Land area (ha)", "Landoppervlakte (ha)"),
            "density_per_km2": t("Density (res./km²)", "Dichtheid (inw./km²)"),
        })
        st.dataframe(_v_df, use_container_width=True, hide_index=True)

    if cbs_trend.get("years"):
        st.divider()
        st.markdown(f"**{t('Municipality-wide growth, 2018–2024', 'Groei gemeentebreed, 2018–2024')}**")
        _cy0, _cy1 = cbs_trend["years"][0], cbs_trend["years"][-1]
        _pop0, _pop1 = cbs_trend["population"][0], cbs_trend["population"][-1]
        _stock0, _stock1 = cbs_trend["housing_stock"][0], cbs_trend["housing_stock"][-1]
        gc1, gc2 = st.columns(2)
        if _pop0 and _pop1:
            gc1.metric(t("Population", "Inwoners"), f"{_pop1:,}", delta=f"{_pop1 - _pop0:+,} ({_cy0}→{_cy1})")
        if _stock0 and _stock1:
            gc2.metric(t("Housing stock", "Woningvoorraad"), f"{_stock1:,}", delta=f"{_stock1 - _stock0:+,} homes ({_cy0}→{_cy1})" if LANG == "en" else f"{_stock1 - _stock0:+,} woningen ({_cy0}→{_cy1})")
        st.caption(t(
            "Real CBS registry counts, not modelled — this is the actual 'how much did housing grow' "
            "answer for the whole municipality; the per-village population/area above is this same "
            "source at finer geography, current-year only (CBS doesn't publish the district-level "
            "breakdown for every past year the way it does the municipality total).",
            "Echte CBS-registratieaantallen, niet gemodelleerd — dit is het daadwerkelijke antwoord op "
            "'hoeveel is de woningvoorraad gegroeid' voor de hele gemeente; de bevolking/oppervlakte per "
            "kern hierboven komt uit dezelfde bron op fijnere geografie, alleen voor het huidige jaar "
            "(CBS publiceert de wijkuitsplitsing niet voor elk voorbij jaar zoals wel bij het "
            "gemeentetotaal).",
        ))

# ======================================================================
with tab_land:
    st.subheader(t("Real field boundaries, real crops", "Echte perceelgrenzen, echte gewassen"))
    st.caption(t(
        f"Source: BRP (Basisregistratie Gewaspercelen) — every parcel a farmer registered for subsidy, "
        f"{brp.get('year', '')}. {brp.get('n_parcels', 0):,} parcels, {brp.get('total_area_ha', 0):,.0f} ha "
        "total inside the municipal boundary.",
        f"Bron: BRP (Basisregistratie Gewaspercelen) — elk perceel dat een boer heeft opgegeven voor "
        f"subsidie, {brp.get('year', '')}. {brp.get('n_parcels', 0):,} percelen, "
        f"{brp.get('total_area_ha', 0):,.0f} ha totaal binnen de gemeentegrens.",
    ))

    c1, c2, c3 = st.columns(3)
    cat_ha = brp.get("by_category_ha", {})
    c1.metric(t("Grassland / pasture", "Grasland / weiland"), f"{cat_ha.get('Grasland', 0):,.0f} ha")
    c2.metric(t("Arable / cropland", "Bouwland / akkerland"), f"{cat_ha.get('Bouwland', 0):,.0f} ha")
    c3.metric(t("Nature terrain", "Natuurterrein"), f"{cat_ha.get('Natuurterrein', 0):,.0f} ha")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**{t('Land use category', 'Landgebruikcategorie')}**")
        if cat_ha:
            st.altair_chart(brp_category_chart(cat_ha), use_container_width=True)
    with right:
        st.markdown(f"**{t('Top crops by area', 'Grootste gewassen naar oppervlakte')}**")
        crops = brp.get("top_crops_ha", {})
        if crops:
            st.altair_chart(top_crops_chart(crops), use_container_width=True)

    st.markdown(f"**{t('All 102 crops, by family', 'Alle 102 gewassen, per familie')}**")
    st.caption(t(
        "Every registered crop name sorted into one family (see Field Explorer for the full per-field "
        "icon legend), so the arable/grassland mix above resolves further than 'top 9 crops, everything "
        "else grouped' without needing 102 individual colours.",
        "Elke geregistreerde gewasnaam ingedeeld in één familie (zie Perceelverkenner voor de volledige "
        "icoonlegenda per perceel), zodat de bouwland/grasland-mix hierboven verder wordt uitgesplitst dan "
        "'top 9 gewassen, de rest samen' zonder dat er 102 losse kleuren nodig zijn.",
    ))
    _brp_gdf = load_brp_gdf()
    if _brp_gdf is not None:
        st.altair_chart(crop_family_chart(_brp_gdf), use_container_width=True)
    else:
        _needs_brp_geometry(t("This chart", "Deze grafiek"))

    st.info(t(
        "**A dairy-and-arable mix, not a monoculture:** permanent grassland (1,417 ha) and silage maize "
        "(469 ha) together point to livestock farming as the anchor land use, alongside winter wheat, "
        "sugar beet and potatoes as the arable rotation. That mix is exactly what determines nitrogen "
        "exposure — livestock farms are the ones a nitrogen-permit product would actually serve.",
        "**Een mix van veeteelt en akkerbouw, geen monocultuur:** blijvend grasland (1.417 ha) en "
        "snijmaïs (469 ha) wijzen samen op veeteelt als de dominante landgebruiksvorm, naast wintertarwe, "
        "suikerbieten en aardappelen als bouwlandrotatie. Precies die mix bepaalt de stikstofbelasting — "
        "veehouderijen zijn de bedrijven die een stikstofvergunning-product daadwerkelijk zou bedienen.",
    ))

    if crop_rotation.get("top_transitions"):
        st.divider()
        _ry0, _ry1 = crop_rotation["years"][0], crop_rotation["years"][-1]
        st.subheader(t(f"Crop rotation, {_ry0}–{_ry1}", f"Gewasrotatie, {_ry0}–{_ry1}"))
        st.caption(t(
            "Each 2025 field matched back to its best-overlapping parcel in every earlier year "
            f"(≥{crop_rotation.get('min_overlap_frac', 0.3):.0%} shared area) — real per-field history "
            "from BRP's own archive, not simulated. Click any field in Field Explorer for its individual "
            "timeline.",
            "Elk perceel van 2025 teruggekoppeld aan het meest overlappende perceel in elk eerder jaar "
            f"(≥{crop_rotation.get('min_overlap_frac', 0.3):.0%} gedeelde oppervlakte) — echte "
            "geschiedenis per perceel uit het eigen BRP-archief, niet gesimuleerd. Klik op een perceel in "
            "Perceelverkenner voor de individuele tijdlijn.",
        ))
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric(t("Fields with full history", "Percelen met volledige historie"),
                   f"{crop_rotation.get('n_parcels_full_history', 0):,}")
        rc2.metric(t("Never changed crop", "Nooit van gewas gewisseld"),
                   f"{crop_rotation.get('pct_stable_full_history', 0):.0f}%")
        rc3.metric(t("Actively rotated", "Actief geroteerd"),
                   f"{100 - crop_rotation.get('pct_stable_full_history', 0):.0f}%")

        st.markdown(f"**{t('Most common crop-to-crop transitions', 'Meest voorkomende gewasovergangen')}**")
        st.altair_chart(rotation_transitions_chart(crop_rotation["top_transitions"]), use_container_width=True)
        st.caption(t(
            "The top transitions are textbook Dutch arable rotation — winter wheat ↔ sugar beet, potatoes "
            "↔ wheat — real crop-management signal, not noise. 'Grasland, tijdelijk → Grasland, blijvend' "
            "is a registry reclassification (temporary → permanent pasture status) rather than a physical "
            "land-use change, and is counted as a transition here since the *label* genuinely changed; "
            "worth knowing before reading it as farmers converting cropland to pasture at that scale.",
            "De belangrijkste overgangen zijn schoolvoorbeelden van Nederlandse bouwlandrotatie — "
            "wintertarwe ↔ suikerbieten, aardappelen ↔ tarwe — een echt teeltsignaal, geen ruis. "
            "'Grasland, tijdelijk → Grasland, blijvend' is een registratieherclassificatie (status "
            "tijdelijk → blijvend grasland) in plaats van een fysieke landgebruiksverandering, en telt "
            "hier wel mee als overgang omdat het *label* daadwerkelijk verandert — goed om te weten "
            "voordat je dit leest als boeren die op die schaal bouwland omzetten naar grasland.",
        ))
        st.markdown(t(
            "**Two real data bugs found and fixed building this** (see `src/crop_rotation.py`): BRP's "
            "historical archive has trailing whitespace on some crop names in 2023/2024 that the current "
            "registry doesn't, and spells maize with a diaeresis (\"Maïs\") where the live registry drops "
            "it (\"Mais\") — both would have registered as false crop changes on every matching field "
            "before being normalized away. **A genuine data-coverage limit, not a bug:** parcel counts "
            "jump from ~1,750 (2020-2022) to ~4,100+ (2023-2025) at roughly constant area — BRP started "
            "registering landscape elements (hedgerows, ditches) as their own small parcels around 2023, "
            "so a small 2025 hedge parcel matching a much larger 2020 field is that schema change showing "
            "up, not a real event.",
            "**Twee echte databugs gevonden en opgelost tijdens de bouw** (zie `src/crop_rotation.py`): "
            "het historische BRP-archief heeft in 2023/2024 een spatie achter sommige gewasnamen die het "
            "huidige register niet heeft, en spelt mais met een trema (\"Maïs\") waar het live register "
            "dat weglaat (\"Mais\") — beide zouden bij elk overeenkomend perceel als valse "
            "gewasverandering zijn geregistreerd zonder normalisatie. **Een echte data-dekkingsgrens, "
            "geen bug:** het aantal percelen springt van ~1.750 (2020-2022) naar ~4.100+ (2023-2025) bij "
            "een vrijwel gelijk blijvend oppervlak — BRP is rond 2023 landschapselementen (heggen, sloten) "
            "als eigen kleine percelen gaan registreren, dus een klein heg-perceel uit 2025 dat matcht met "
            "een veel groter perceel uit 2020 is die schemawijziging die zichtbaar wordt, geen echte "
            "gebeurtenis.",
        ))

    if soil.get("properties"):
        st.divider()
        st.subheader(t("Soil: what's actually under the farmland", "Bodem: wat er daadwerkelijk onder het landbouwland zit"))
        _sp = soil["properties"]
        st.caption(t(
            f"Sampled at {soil.get('n_points_sampled', 0)} real BRP-parcel centroids ({soil.get('n_points_sampled', 0) - soil.get('n_points_null', 0)} landed on mapped ground; the rest fell on open water/no-coverage and were dropped), "
            "topsoil (0-5cm), from ISRIC SoilGrids — a global 250m model built from real profile "
            "observations plus environmental covariates, not a Dutch source: PDOK/BRO's own soil map "
            "(Bodemkaart) has no discoverable public WFS/WMS under any endpoint pattern this pipeline's "
            "other PDOK services use (checked directly), and its province-level alternatives (Utrecht, "
            "Zeeland, Zuid-Holland) don't cover Gelderland. A real methodology substitution, not a silent one.",
            f"Bemonsterd op {soil.get('n_points_sampled', 0)} echte BRP-perceelcentroïden ({soil.get('n_points_sampled', 0) - soil.get('n_points_null', 0)} kwamen op gekarteerde grond terecht; de rest viel op open water/geen dekking en is weggelaten), "
            "bovengrond (0-5cm), van ISRIC SoilGrids — een wereldwijd 250m-model op basis van echte "
            "profielwaarnemingen plus omgevingsvariabelen, geen Nederlandse bron: de eigen bodemkaart van "
            "PDOK/BRO heeft geen vindbare publieke WFS/WMS onder enig endpoint-patroon dat de andere "
            "PDOK-diensten van deze pipeline gebruiken (rechtstreeks gecontroleerd), en de "
            "provincie-alternatieven (Utrecht, Zeeland, Zuid-Holland) dekken Gelderland niet. Een echte "
            "methodologische vervanging, geen stille.",
        ))

        sc1, sc2, sc3, sc4 = st.columns(4)
        soc_val = _sp.get("soc", {}).get("mean")
        sc1.metric(t("Organic carbon", "Organische koolstof"), f"{soc_val:.0f} g/kg" if soc_val else "n/a",
                   delta=(f"≈{soc_val * 1.724 / 10:.1f}% " + t("organic matter", "organische stof")) if soc_val else None,
                   delta_color="off")
        ph_val = _sp.get("phh2o", {}).get("mean")
        sc2.metric(t("pH (water)", "pH (water)"), f"{ph_val:.1f}" if ph_val else "n/a")
        n_val = _sp.get("nitrogen", {}).get("mean")
        sc3.metric(t("Total nitrogen", "Totaal stikstof"), f"{n_val:.1f} g/kg" if n_val else "n/a")
        cec_val = _sp.get("cec", {}).get("mean")
        sc4.metric(t("Cation exchange (CEC)", "Kationuitwisseling (CEC)"), f"{cec_val:.0f} cmol(c)/kg" if cec_val else "n/a")

        tc1, tc2, tc3, tc4 = st.columns(4)
        clay_val, sand_val, silt_val = _sp.get("clay", {}).get("mean"), _sp.get("sand", {}).get("mean"), _sp.get("silt", {}).get("mean")
        if clay_val:
            metric_with_bar(tc1, t("Clay", "Klei"), f"{clay_val:.0f}%", kpi_proportion_bar_svg(clay_val))
        else:
            tc1.metric(t("Clay", "Klei"), "n/a")
        if sand_val:
            metric_with_bar(tc2, t("Sand", "Zand"), f"{sand_val:.0f}%", kpi_proportion_bar_svg(sand_val))
        else:
            tc2.metric(t("Sand", "Zand"), "n/a")
        if silt_val:
            metric_with_bar(tc3, t("Silt", "Silt"), f"{silt_val:.0f}%", kpi_proportion_bar_svg(silt_val))
        else:
            tc3.metric(t("Silt", "Silt"), "n/a")
        bd_val = _sp.get("bdod", {}).get("mean")
        tc4.metric(t("Bulk density", "Bulkdichtheid"), f"{bd_val:.2f} kg/dm³" if bd_val else "n/a")

        if all(v is not None for v in (soc_val, ph_val, cec_val, bd_val, clay_val, sand_val, silt_val)):
            om_pct = soc_val * 1.724 / 10
            st.info(t(
                "**A generally favourable picture, read as a first-pass indicator, not a certified soil "
                f"test:** organic matter (≈{om_pct:.0f}%) is high for Dutch mineral farmland, consistent "
                "with the grassland-heavy land use this dashboard's own BRP data shows (pasture builds "
                "organic matter faster than continuous arable) — high organic matter and the low bulk "
                f"density ({bd_val:.2f} kg/dm³) reading agree with each other, since dense soil and high "
                "organic matter rarely coincide. CEC in the moderate-high range means decent "
                f"nutrient-holding capacity. pH ({ph_val:.1f}) sits slightly acidic — fine for grass, on "
                "the low side for some arable crops, a real reason a farmer might lime specific fields. "
                f"Texture (clay {clay_val:.0f}% / sand {sand_val:.0f}% / silt {silt_val:.0f}%) reads as "
                "loam/silt loam — workable, holds both water and nutrients reasonably, neither the "
                "heavy-clay nor droughty-sand extreme. **What this can't tell you:** SoilGrids is a 250m "
                "model, not a field sample — real soil varies within that footprint, and an actual soil "
                "test is what any real agronomic decision should be based on, not this.",
                "**Een over het algemeen gunstig beeld, te lezen als eerste indicatie, geen gecertificeerde "
                f"bodemtest:** het organische-stofgehalte (≈{om_pct:.0f}%) is hoog voor Nederlandse "
                "minerale landbouwgrond, consistent met het grasland-zware landgebruik dat de eigen "
                "BRP-data van dit dashboard laat zien (weiland bouwt sneller organische stof op dan "
                f"continu bouwland) — hoog organische stof en de lage bulkdichtheid ({bd_val:.2f} kg/dm³) "
                "passen bij elkaar, want dichte grond en hoog organische stof gaan zelden samen. CEC in "
                f"het matig-hoge bereik betekent een redelijk vermogen om voedingsstoffen vast te houden. "
                f"De pH ({ph_val:.1f}) is licht zuur — prima voor gras, aan de lage kant voor sommige "
                "bouwlandgewassen, een reële reden om bepaalde percelen te bekalken. De textuur "
                f"(klei {clay_val:.0f}% / zand {sand_val:.0f}% / silt {silt_val:.0f}%) leest als leem/"
                "siltige leem — bewerkbaar, houdt zowel water als voedingsstoffen redelijk vast, geen van "
                "beide uitersten van zware klei of droogtegevoelig zand. **Wat dit niet kan vertellen:** "
                "SoilGrids is een 250m-model, geen veldmonster — echte bodem varieert binnen die "
                "voetafdruk, en een echte bodemtest is waar elke echte agronomische beslissing op hoort "
                "te steunen, niet dit.",
            ))

    st.subheader(t("Forest & nature", "Bos & natuur"))
    nature_brp = cat_ha.get("Natuurterrein", 0)
    forest_satellite_pct = sum(v for k, v in landcover.items() if "vegetation" in k.lower())
    st.markdown(t(
        f"- **{nature_brp:,.0f} ha** registered as nature terrain in BRP (mostly agri-environment-scheme "
        "grazing — e.g. the Millingerwaard rewilding area).\n"
        f"- **{forest_satellite_pct:.0f}%** of clear satellite ground reads as dense vegetation (the "
        "KMeans land cover) — this is the honest, broader estimate, since unmanaged forest on the ridge "
        "isn't a BRP-registered parcel at all. The two numbers measure different things and shouldn't be "
        "added together.",
        f"- **{nature_brp:,.0f} ha** geregistreerd als natuurterrein in de BRP (grotendeels "
        "agrarisch-natuurbeheer-beweiding — bijv. het rewildingsgebied de Millingerwaard).\n"
        f"- **{forest_satellite_pct:.0f}%** van de onbewolkte satellietgrond leest als dichte vegetatie "
        "(de KMeans-landgebruikskaart) — dit is de eerlijke, bredere schatting, omdat onbeheerd bos op de "
        "stuwwal helemaal geen BRP-geregistreerd perceel is. De twee cijfers meten iets anders en horen "
        "niet bij elkaar opgeteld te worden.",
    ))

    if landcover_change_map.get("ha"):
        st.divider()
        _lccy0, _lccy1 = landcover_change_map["first_year"], landcover_change_map["last_year"]
        st.subheader(t(f"Where the forest actually went, {_lccy0}→{_lccy1}", f"Waar het bos daadwerkelijk gebleven is, {_lccy0}→{_lccy1}"))
        _ha = landcover_change_map["ha"]
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric(t("Forest lost", "Bos verloren"), f"{_ha.get('Forest loss', 0):,.0f} ha")
        fc2.metric(t("Forest gained", "Bos gewonnen"), f"{_ha.get('Forest gain', 0):,.0f} ha")
        _net_forest = _ha.get("Forest gain", 0) - _ha.get("Forest loss", 0)
        fc3.metric(t("Net forest change", "Netto bosverandering"), f"{_net_forest:+,.0f} ha")
        fc4.metric(t("Built-up growth", "Bebouwingsgroei"), f"{_ha.get('Built-up growth', 0):,.0f} ha")
        st.caption(t(
            f"A real spatial map, not just this hectare summary — every pixel that changed class between "
            f"{_lccy0} and {_lccy1} is drawn on the **Overview** tab's map as its own toggleable layer "
            "(\"Forest & land cover change\"), so you can see *where* forest was lost and gained, not just "
            "the net number. Net change here should closely match the year-over-year trend above — a real "
            "cross-check between two independently computed views of the same underlying classification.",
            f"Een echte ruimtelijke kaart, niet alleen deze hectare-samenvatting — elke pixel die van "
            f"klasse veranderde tussen {_lccy0} en {_lccy1} staat getekend op de kaart van het tabblad "
            "**Overzicht** als eigen aan-/uitzetbare laag (\"Bos & landgebruikverandering\"), zodat je ziet "
            "*waar* bos verloren en gewonnen is, niet alleen het nettocijfer. De nettoverandering hier komt "
            "in de buurt van de jaar-op-jaar-trend hierboven — een echte kruiscontrole tussen twee "
            "onafhankelijk berekende weergaven van dezelfde onderliggende classificatie.",
        ))
        with st.expander(t("How this map is calculated", "Hoe deze kaart wordt berekend")):
            st.markdown(t(
                f"1. Every Sentinel-2 August scene from {_lccy0} to {_lccy1} is independently run through "
                "the same unsupervised KMeans classifier (Land & Crops / Trends & Climate tabs), each "
                "pixel landing in one of six spectral clusters, then rolled up into four broad categories "
                "(Water, Built-up, Agriculture, Forest/dense vegetation).\n"
                f"2. Because every one of those years shares the exact same 10m pixel grid (checked "
                f"directly — identical transform, shape and CRS, not assumed), the {_lccy0} and {_lccy1} "
                f"rasters can be compared pixel-for-pixel: a pixel that was 'Forest' in {_lccy0} and "
                f"something else in {_lccy1} is real forest loss at that exact location, not a resampling "
                "artefact.\n"
                "3. Where a pixel is both 'left forest' and 'became built-up' in the same window, it's "
                "labelled forest loss — the category this map exists to answer — rather than split across "
                "two labels.\n"
                f"4. Restricted to {_lccy0}–{_lccy1} on purpose: the fuller 2005–present trend elsewhere "
                "in this dashboard uses 30m Landsat imagery for the earlier years, a *different* pixel "
                "grid entirely. Comparing that against this 10m grid pixel-for-pixel without reprojecting "
                "one onto the other first would silently misalign ground, not just lose precision — so "
                "this map's honest spatial window is the Sentinel-2 era, even though the trend *line* "
                "goes back further.",
                f"1. Elke Sentinel-2-augustusopname van {_lccy0} tot {_lccy1} wordt onafhankelijk door "
                "dezelfde ongestuurde KMeans-classifier gehaald (tabbladen Land & Gewassen / Trends & "
                "Klimaat), waarbij elke pixel in een van zes spectrale clusters valt, daarna opgeschaald "
                "naar vier brede categorieën (Water, Bebouwd, Landbouw, Bos/dichte vegetatie).\n"
                f"2. Omdat elk van die jaren precies hetzelfde 10m-pixelraster deelt (rechtstreeks "
                f"gecontroleerd — identieke transform, vorm en CRS, niet aangenomen), kunnen de rasters "
                f"van {_lccy0} en {_lccy1} pixel-voor-pixel vergeleken worden: een pixel die in {_lccy0} "
                f"'Bos' was en in {_lccy1} iets anders, is echt bosverlies op die exacte locatie, geen "
                "resampling-artefact.\n"
                "3. Waar een pixel in hetzelfde venster zowel 'bos verlaten' als 'bebouwd geworden' is, "
                "wordt hij gelabeld als bosverlies — de categorie waar deze kaart voor bedoeld is — in "
                "plaats van over twee labels verdeeld.\n"
                f"4. Bewust beperkt tot {_lccy0}–{_lccy1}: de volledigere trend sinds 2005 elders in dit "
                "dashboard gebruikt voor de eerdere jaren 30m Landsat-beelden, een *ander* pixelraster. "
                "Dat pixel-voor-pixel vergelijken met dit 10m-raster zonder eerst het een op het ander te "
                "reprojecteren zou de grond stilletjes verkeerd uitlijnen, niet alleen precisie verliezen "
                "— dus het eerlijke ruimtelijke venster van deze kaart is het Sentinel-2-tijdperk, ook al "
                "gaat de trendlijn verder terug.",
            ))

# ======================================================================
with tab_climate:
    st.subheader(t("22 years of vegetation, weather, and how they connect",
                    "22 jaar vegetatie, weer, en hun onderlinge samenhang"))
    st.caption(t(
        "Two satellite indices (NDVI, NDWI) since 2005, real daily weather since 2005 from KNMI's nearest "
        "station, and a direct statistical comparison between them — not just shown side by side, but "
        "actually correlated.",
        "Twee satellietindices (NDVI, NDWI) sinds 2005, echte dagelijkse weersdata sinds 2005 van het "
        "dichtstbijzijnde KNMI-station, en een directe statistische vergelijking ertussen — niet alleen "
        "naast elkaar getoond, maar daadwerkelijk gecorreleerd.",
    ))

    sel_year = None
    if trend.get("years"):
        st.markdown("#### " + t("Year Explorer — pick a year", "Jaarverkenner — kies een jaar"))
        st.caption(t(
            "Every number this pipeline has for one specific year, gathered in one place — the charts "
            "below ring whichever year is picked here, so it's visible in context, not just as a number.",
            "Alle cijfers die deze pipeline voor één specifiek jaar heeft, op één plek — de grafieken "
            "hieronder markeren het gekozen jaar, zodat het in context zichtbaar is, niet alleen als getal.",
        ))
        _years = trend["years"]
        sel_year = st.select_slider(t("Year", "Jaar"), options=_years, value=_years[-1], key="year_explorer")
        _idx = _years.index(sel_year)
        _source_label = {"landsat": "Landsat (30m)", "sentinel2": "Sentinel-2 (10m)"}

        yc1, yc2, yc3, yc4 = st.columns(4)
        # Each index placed on its own real, fixed scale (-1..+1 for
        # NDVI/NDWI, 0..100% for cloud-free coverage) via a small coloured
        # range bar under the number -- not just the digit on its own.
        metric_with_bar(yc1, "NDVI", f"{trend['mean_ndvi'][_idx]:.3f}",
                         index_range_svg(trend["mean_ndvi"][_idx], -1, 1, low_color="#A2712F", high_color=KPI_INK, key="ndvi"))
        _ndwi_series = trend.get("mean_ndwi") or [None] * len(_years)
        _ndwi_val = _ndwi_series[_idx] if _idx < len(_ndwi_series) else None
        if _ndwi_val is not None:
            metric_with_bar(yc2, "NDWI", f"{_ndwi_val:.3f}",
                             index_range_svg(_ndwi_val, -1, 1, low_color="#A2712F", high_color="#3B6E8A", key="ndwi"))
        else:
            yc2.metric("NDWI", "n/a")
        yc3.metric(t("Sensor", "Sensor"), _source_label.get(trend["source"][_idx], trend["source"][_idx]))
        metric_with_bar(yc4, t("Cloud-free coverage", "Wolkenvrije dekking"), f"{trend['clear_pct'][_idx]:.0f}%",
                         kpi_proportion_bar_svg(trend["clear_pct"][_idx]))
        _yic1, _yic2 = st.columns([10, 1])
        with _yic2:
            method_popover("ndvi")

        _aug = weather.get("august", {})
        if sel_year in _aug.get("years", []):
            _widx = _aug["years"].index(sel_year)
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric(t("August rainfall", "Augustusneerslag"), f"{_aug['total_precip_mm'][_widx]:.0f} mm")
            wc2.metric(t("August mean temp", "Gem. augustustemperatuur"), f"{_aug['mean_temp_c'][_widx]:.1f}°C")
            wc3.metric(t("August sunshine", "Augustus zonuren"), f"{_aug['total_sunshine_h'][_widx]:.0f} h")
        else:
            st.caption(t(f"No weather record loaded for {sel_year} yet — run `src/fetch_weather.py`.",
                          f"Nog geen weersdata geladen voor {sel_year} — draai `src/fetch_weather.py`."))

        _matched_by_year = crop_rotation.get("parcels_matched_by_year", {})
        if str(sel_year) in _matched_by_year:
            _ref_year = crop_rotation.get("years", [None])[-1]
            _n_total = brp.get("n_parcels", 0)
            st.caption(t(
                f"Crop rotation: {_matched_by_year[str(sel_year)]:,} of {_ref_year}'s {_n_total:,} fields "
                f"have a matched registered crop for {sel_year} — see Land & Crops for the full rotation picture.",
                f"Gewasrotatie: {_matched_by_year[str(sel_year)]:,} van de {_n_total:,} percelen van "
                f"{_ref_year} hebben een gematcht geregistreerd gewas voor {sel_year} — zie Land & Gewassen "
                "voor het volledige rotatiebeeld.",
            ))

        _notes = []
        if sel_year == 2018:
            _notes.append(t("Documented 2018 Northwestern-Europe drought — this series' single lowest NDVI year.",
                             "Gedocumenteerd droogtejaar 2018 in Noordwest-Europa — laagste NDVI van deze hele reeks."))
        if sel_year in _aug.get("years", []):
            _widx = _aug["years"].index(sel_year)
            if _aug["mean_temp_c"][_widx] == max(_aug["mean_temp_c"]):
                _notes.append(t("Hottest August in this whole 22-year weather record.", "Heetste augustus uit deze hele 22-jarige weersreeks."))
            if _aug["total_precip_mm"][_widx] == min(_aug["total_precip_mm"]):
                _notes.append(t("Driest August in this whole 22-year weather record.", "Droogste augustus uit deze hele 22-jarige weersreeks."))
            if _aug["total_precip_mm"][_widx] == max(_aug["total_precip_mm"]):
                _notes.append(t("Wettest August in this whole 22-year weather record.", "Natste augustus uit deze hele 22-jarige weersreeks."))
        if sel_year == 2024:
            _notes.append(t("January this year: the documented Rhine/Waal high water this pipeline's flood analysis covers — see the Water tab.",
                             "Januari dit jaar: het gedocumenteerde hoogwater van Rijn/Waal dat de overstromingsanalyse van deze pipeline behandelt — zie het tabblad Water."))
        for _n in _notes:
            st.info(_n)

        st.markdown("#### " + t("Vegetation indices · NDVI & NDWI", "Vegetatie-indices · NDVI & NDWI"))
        st.caption(t(
            "Two sensors, one series: Landsat (30m) before Sentinel-2 L2A coverage gets reliable here, "
            "Sentinel-2 (10m) from 2018 on — colour marks which. Each point is that August's single "
            "least-cloudy date, so real phenological/weather noise sits inside the line alongside any "
            "trend; hover a point for its exact date's platform and cloud-free coverage.",
            "Twee sensoren, één reeks: Landsat (30m) voordat de dekking van Sentinel-2 L2A hier "
            "betrouwbaar wordt, Sentinel-2 (10m) vanaf 2018 — kleur geeft aan welke. Elk punt is de "
            "minst bewolkte datum van die augustus, dus echte fenologische/weersruis zit in de lijn naast "
            "elke trend; beweeg over een punt voor het platform en de onbewolkte dekking van die datum.",
        ))
        st.altair_chart(ndvi_trend_chart(trend, highlight_year=sel_year), use_container_width=True)
        st.caption(t(
            f"Net {trend.get('net_change', 0):+.3f} NDVI over {len(trend['years']) - 1} years "
            f"({trend.get('slope_per_year', 0):+.4f}/yr) — noisy year to year; read the shape, not the "
            "slope, as the finding. 2018's dip lines up with the documented 2018 Northwestern-Europe "
            "drought, corroborating evidence for the pipeline rather than something to explain away.",
            f"Netto {trend.get('net_change', 0):+.3f} NDVI over {len(trend['years']) - 1} jaar "
            f"({trend.get('slope_per_year', 0):+.4f}/jaar) — grillig van jaar tot jaar; lees de vorm, "
            "niet de helling, als de bevinding. De dip van 2018 komt overeen met de gedocumenteerde "
            "droogte van 2018 in Noordwest-Europa — bevestigend bewijs voor de pipeline, geen anomalie "
            "om weg te verklaren.",
        ))

        if trend.get("mean_ndwi") and any(v is not None for v in trend["mean_ndwi"]):
            st.altair_chart(ndwi_trend_chart(trend, highlight_year=sel_year), use_container_width=True)
            st.caption(t(
                "NDWI (green minus near-infrared): consistently negative here since the AOI is mostly "
                "land, not water — a less negative value means relatively wetter or less-vegetated "
                "ground that year, not literally \"more water.\"",
                "NDWI (groen min nabij-infrarood): hier steeds negatief omdat het gebied grotendeels "
                "land is, geen water — minder negatief betekent dat jaar relatief nattere of minder "
                "begroeide grond, niet letterlijk \"meer water.\"",
            ))

    if landcover_trend.get("years"):
        st.divider()
        _lc_y0, _lc_y1 = landcover_trend["years"][0], landcover_trend["years"][-1]
        st.markdown("#### " + t(f"Land cover area, {_lc_y0}–{_lc_y1}", f"Landgebruikoppervlak, {_lc_y0}–{_lc_y1}"))
        st.caption(t(
            "Built-up, forest, farmland and water, as real hectares — not this year's snapshot. Compared "
            "only within the footprint valid in *every* year (cloud coverage differs by year, so a raw "
            "per-year hectare count would confuse 'more cloud-free ground' with 'more of this category'); "
            "still, read the *shape* across years, not one year-to-year jump, as the finding — an "
            "unsupervised classifier re-run independently each year has real year-to-year label wobble, "
            "especially between 'Built-up/bare' and 'Agriculture' (bare/just-harvested cropland can look "
            "spectrally similar to hard surfaces in a single snapshot).",
            "Bebouwd, bos, landbouw en water, als echte hectares — niet de momentopname van dit jaar. "
            "Alleen vergeleken binnen het gebied dat in *elk* jaar geldig is (bewolking verschilt per "
            "jaar, dus een ruwe hectaretelling per jaar zou 'meer onbewolkte grond' verwarren met 'meer "
            "van deze categorie'); lees toch de *vorm* over de jaren, niet één sprong van jaar op jaar, "
            "als de bevinding — een ongestuurde classifier die elk jaar apart draait heeft echte "
            "jaar-op-jaar labelruis, vooral tussen 'Bebouwd/kaal' en 'Landbouw' (kale of net geoogste "
            "akkers kunnen spectraal op verharding lijken in één opname).",
        ))
        _lc_cols = st.columns(4)
        for _lc_col, (_cat, _vals) in zip(_lc_cols, landcover_trend["series_ha"].items()):
            with _lc_col:
                _chg = landcover_trend.get("change", {}).get(_cat, {})
                st.markdown(f"**{_cat}**")
                st.altair_chart(landcover_trend_chart(landcover_trend["years"], _vals, _cat), use_container_width=True)
                st.caption(f"{_chg.get('net_ha', 0):+,.0f} ha ({_chg.get('net_pct', 0):+.1f}%)")
        st.caption(t(
            "Built-up reads as a net decrease here despite the municipality's own new-homes figures "
            "elsewhere in this dashboard (CBS: real, positive housing growth) — 2018, this series' first "
            "year, was also the documented 2018 drought (see above): exceptionally bare, stressed ground "
            "that August plausibly over-counted into 'Built-up/bare' that specific year, inflating the "
            "starting point rather than construction ever having reversed. Forest and water are the more "
            "spectrally stable categories and worth more confidence; built-up specifically should be read "
            "as directional, not a precise hectare count, until cross-checked against BAG (the building "
            "registry) — a real next step, not done here.",
            "Bebouwd oogt hier als een netto afname ondanks de eigen nieuwbouwcijfers van de gemeente "
            "elders op dit dashboard (CBS: echte, positieve woningbouwgroei) — 2018, het eerste jaar van "
            "deze reeks, was ook de gedocumenteerde droogte van 2018 (zie hierboven): die augustus "
            "uitzonderlijk kale, gestreste grond telde plausibel mee als 'Bebouwd/kaal' dat specifieke "
            "jaar, wat het startpunt opblaast in plaats van dat bouwactiviteit ooit is teruggedraaid. Bos "
            "en water zijn de spectraal stabielere categorieën en verdienen meer vertrouwen; bebouwd "
            "moet specifiek gelezen worden als richting, geen precieze hectaretelling, tot het getoetst "
            "is aan de BAG (het gebouwenregister) — een echte vervolgstap, hier nog niet gedaan.",
        ))

    if weather.get("august", {}).get("years"):
        st.divider()
        st.markdown("#### " + t(
            f"Real weather · {weather.get('station_name', '?')} station, {weather.get('first_year', '?')}–{weather.get('last_year', '?')}",
            f"Echte weersdata · station {weather.get('station_name', '?')}, {weather.get('first_year', '?')}–{weather.get('last_year', '?')}",
        ))
        st.caption(t(
            "Nearest official KNMI station to the AOI (~25km north — no station sits inside the "
            "municipality itself) — the Dutch national weather service's own daily record, free and "
            "public, no key required. KNMI's own disclaimer: raw daily series like this aren't "
            "homogenized for trend analysis (station moves, instrument changes over 20 years) — a real "
            "caveat, not a formality.",
            "Dichtstbijzijnde officiële KNMI-station bij het gebied (~25km noordelijker — er staat geen "
            "station binnen de gemeente zelf) — het eigen dagelijkse archief van het KNMI, gratis en "
            "openbaar, geen sleutel nodig. KNMI's eigen kanttekening: ruwe dagreeksen zoals deze zijn "
            "niet gehomogeniseerd voor trendanalyse (stationsverplaatsingen, instrumentwijzigingen over "
            "20 jaar) — een echte beperking, geen formaliteit.",
        ))
        aug = weather["august"]
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.altair_chart(
                weather_year_chart(aug["years"], aug["total_precip_mm"], t("August rainfall (mm)", "Augustusneerslag (mm)"), COLOR_SENTINEL2),
                use_container_width=True,
            )
        with wc2:
            st.altair_chart(
                weather_year_chart(aug["years"], aug["mean_temp_c"], t("August mean temp (°C)", "Gem. augustustemperatuur (°C)"), COLOR_TEMP, fmt=".1f"),
                use_container_width=True,
            )
        with wc3:
            st.altair_chart(
                weather_year_chart(aug["years"], aug["total_sunshine_h"], t("August sunshine (hours)", "Augustus zonuren"), COLOR_SUN),
                use_container_width=True,
            )
        st.caption(t(
            f"Hottest August: {weather['august']['years'][int(np.argmax(aug['mean_temp_c']))]}. "
            f"Driest: {weather['august']['years'][int(np.argmin(aug['total_precip_mm']))]}. "
            f"Wettest: {weather['august']['years'][int(np.argmax(aug['total_precip_mm']))]}.",
            f"Heetste augustus: {weather['august']['years'][int(np.argmax(aug['mean_temp_c']))]}. "
            f"Droogste: {weather['august']['years'][int(np.argmin(aug['total_precip_mm']))]}. "
            f"Natste: {weather['august']['years'][int(np.argmax(aug['total_precip_mm']))]}.",
        ))

    if climate_corr.get("correlations"):
        st.divider()
        st.markdown("#### " + t("Does the weather actually explain the vegetation trend?",
                                 "Verklaart het weer de vegetatietrend daadwerkelijk?"))
        st.altair_chart(correlation_chart(climate_corr["correlations"]), use_container_width=True)
        r_temp = climate_corr["correlations"].get("ndvi_vs_august_temp")
        r_precip = climate_corr["correlations"].get("ndvi_vs_august_precip")
        r_temp_txt = f"{r_temp:+.2f}" if r_temp is not None else "n/a"
        r_precip_txt = f"{r_precip:+.2f}" if r_precip is not None else "n/a"
        st.info(t(
            f"Pearson correlation across {climate_corr.get('n_years', 0)} years. **August temperature is "
            f"the strongest single correlate of that year's NDVI (r = {r_temp_txt})** — hotter Augusts, "
            "lower vigour, the physically expected direction (heat stress). Rainfall alone correlates "
            f"weakly (r = {r_precip_txt}) — in a temperate, non-irrigated landscape like this, a hot, "
            "sunny month can stress vegetation even with adequate total rainfall, so temperature (and the "
            "evapotranspiration it drives) tracking NDVI better than rainfall alone is a real, physically "
            "sensible finding, not a fluke of these particular points.\n\n"
            "**Read this carefully, though:** this sample of years is small for a correlation coefficient "
            "— treat anything under about |r|=0.4 as noise, not \"no relationship.\" Each year's NDVI is "
            "one snapshot day; the weather side is a whole month's total/mean — a snapshot compared "
            "against an integral, not two directly comparable measurements. And temperature, rainfall and "
            "sunshine all move together across a real summer, so correlation here can't cleanly isolate "
            "which one variable is \"the\" driver.",
            f"Pearson-correlatie over {climate_corr.get('n_years', 0)} jaar. **De augustustemperatuur is "
            f"de sterkste afzonderlijke correlatie met de NDVI van dat jaar (r = {r_temp_txt})** — hetere "
            "augustusmaanden, minder vitaliteit, precies de fysiek verwachte richting (hittestress). "
            f"Neerslag alleen correleert zwak (r = {r_precip_txt}) — in een gematigd, niet-beregend "
            "landschap als dit kan een hete, zonnige maand vegetatie stress geven zelfs bij voldoende "
            "totale neerslag, dus dat temperatuur (en de verdamping die het aandrijft) de NDVI beter "
            "volgt dan neerslag alleen is een echte, fysiek zinnige bevinding, geen toevalstreffer van "
            "deze specifieke punten.\n\n"
            "**Lees dit wel met de nodige voorzichtigheid:** dit aantal jaren is klein voor een "
            "correlatiecoëfficiënt — behandel alles onder ongeveer |r|=0,4 als ruis, niet als \"geen "
            "verband.\" De NDVI van elk jaar is één momentopname; de weerskant is het totaal/gemiddelde "
            "van een hele maand — een momentopname vergeleken met een integraal, geen twee direct "
            "vergelijkbare metingen. En temperatuur, neerslag en zonneschijn bewegen in een echte zomer "
            "samen op, dus correlatie kan hier niet precies aanwijzen welke ene variabele \"de\" aanjager is.",
        ))
    elif trend.get("years"):
        st.caption(t(
            "Run `python src/fetch_weather.py` and `python src/climate_correlation.py` to add the weather "
            "comparison and correlation analysis here.",
            "Draai `python src/fetch_weather.py` en `python src/climate_correlation.py` om de "
            "weersvergelijking en correlatie-analyse hier toe te voegen.",
        ))

# ======================================================================
with tab_forecast:
    st.subheader(t("What the recent trend suggests, if it continues",
                    "Wat de recente trend suggereert, als die doorzet"))
    st.caption(t(
        "Simple, transparent models on data this pipeline already has — no black box, and every number "
        "ships with its own uncertainty rather than a bare point guess. Method and full limitations in "
        "`src/forecast.py`.",
        "Eenvoudige, transparante modellen op data die deze pipeline al heeft — geen black box, en elk "
        "cijfer heeft zijn eigen onzekerheid in plaats van een kale puntschatting. Methode en volledige "
        "beperkingen in `src/forecast.py`.",
    ))
    st.warning(t(
        "**Read every chart below as \"if the recent trend continues,\" never a guarantee.** Each series "
        "is 5–9 real annual points, several with real year-to-year noise (weather, cloud cover, classifier "
        "wobble) — a straight-line extrapolation on that few, that noisy a sample has a wide honest "
        "interval, shown as the shaded band. **Read the band, not the dashed line, as the actual answer.**",
        "**Lees elke grafiek hieronder als \"als de recente trend doorzet,\" nooit als garantie.** Elke "
        "reeks is 5–9 echte jaarpunten, meerdere met echte jaar-op-jaar ruis (weer, bewolking, "
        "classifier-wiebel) — een rechte-lijn-extrapolatie op zo weinig, zo'n ruizige steekproef heeft een "
        "breed eerlijk interval, getoond als de gearceerde band. **Lees de band, niet de gestippelde "
        "lijn, als het daadwerkelijke antwoord.**",
    ))

    with st.expander(t("How every forecast on this tab is actually calculated", "Hoe elke voorspelling op dit tabblad daadwerkelijk wordt berekend")):
        st.markdown(t(
            "**Vegetation, land cover, population, housing stock — one method, four series:** ordinary "
            "least squares (`sklearn.linear_model.LinearRegression`) fit on that series' own real annual "
            "values, then projected 5 years past the last observed one. The shaded band is a real 80% "
            "prediction interval (`scipy.stats.t`, the textbook formula) — it widens the further a "
            "projected year sits from the data's own mean year, because a straight line is always more "
            "uncertain further from what it was actually fit on. The R² shown next to each chart is how "
            "much of that series' year-to-year variation the straight line explains at all — low R² (like "
            "NDVI's 0.31) means treat the band, not the point estimate, as the real answer; high R² (like "
            "housing stock's 0.98) means the straight line is doing real work, not just decoration.\n\n"
            "**Crop rotation — a different method entirely:** not a trend line. Every field's real "
            "multi-year BRP history (crop_rotation.py's matched 2020–2025 records) is walked year-to-year, "
            "and every observed crop-family-to-crop-family transition is counted. Dividing each family's "
            "outgoing counts by their total gives `transition_matrix[A][B]` — the empirical probability "
            "that a field grown as family A one year is family B the next, among the transitions this "
            "pipeline's own matched history actually contains. Field Explorer's colour mode looks up each "
            "field's current family in that matrix and colours it by the most likely next family; the "
            "municipality-wide bar chart below spreads every field's own area across next year's families "
            "by those same probabilities (an expected value, not 3,787 individual hard predictions).\n\n"
            "**None of this is a trained, tuned, cross-validated model** — it's the simplest method that's "
            "honest about a 5–9-point, genuinely noisy sample, on real data this pipeline already had. "
            "Full code and reasoning: `src/forecast.py`.",
            "**Vegetatie, landgebruik, bevolking, woningvoorraad — één methode, vier reeksen:** gewone "
            "kleinste-kwadraten-regressie (`sklearn.linear_model.LinearRegression`) op de eigen echte "
            "jaarwaarden van die reeks, daarna 5 jaar voorbij het laatst waargenomen jaar geprojecteerd. "
            "De gearceerde band is een echt 80%-voorspelinterval (`scipy.stats.t`, de standaardformule) — "
            "hij wordt breder naarmate een voorspeld jaar verder van het eigen gemiddelde jaar van de data "
            "afligt, omdat een rechte lijn altijd onzekerder is verder van waar hij daadwerkelijk op is "
            "gefit. De R² naast elke grafiek is hoeveel van de jaar-op-jaar-variatie van die reeks de "
            "rechte lijn überhaupt verklaart — lage R² (zoals NDVI's 0,31) betekent: lees de band, niet de "
            "puntschatting, als het echte antwoord; hoge R² (zoals de woningvoorraad's 0,98) betekent dat "
            "de rechte lijn echt werk verzet, geen versiering.\n\n"
            "**Gewasrotatie — een compleet andere methode:** geen trendlijn. De echte meerjarige "
            "BRP-geschiedenis van elk perceel (crop_rotation.py's gematchte 2020-2025-records) wordt "
            "jaar-op-jaar doorlopen, en elke waargenomen overgang van de ene gewasfamilie naar de andere "
            "wordt geteld. Het delen van elke familie's uitgaande tellingen door hun totaal geeft "
            "`transition_matrix[A][B]` — de empirische kans dat een perceel met familie A het jaar erop "
            "familie B heeft, onder de overgangen die de eigen gematchte geschiedenis van deze pipeline "
            "daadwerkelijk bevat. De kleurmodus van Perceelverkenner zoekt de huidige familie van elk "
            "perceel op in die matrix en kleurt het naar de meest waarschijnlijke volgende familie; de "
            "gemeentebrede staafgrafiek hieronder verdeelt de oppervlakte van elk perceel over de families "
            "van volgend jaar volgens diezelfde kansen (een verwachtingswaarde, geen 3.787 individuele "
            "harde voorspellingen).\n\n"
            "**Niets hiervan is een getraind, afgesteld, kruisgevalideerd model** — het is de eenvoudigste "
            "methode die eerlijk is over een steekproef van 5-9 punten die echt ruizig is, op echte data "
            "die deze pipeline al had. Volledige code en redenering: `src/forecast.py`.",
        ))

    if not forecast:
        st.info(t("Run `python src/forecast.py` (or the full pipeline) to add forecasts here.",
                   "Draai `python src/forecast.py` (of de hele pipeline) om hier voorspellingen toe te voegen."))
    else:
        veg = forecast.get("vegetation") or {}
        if veg.get("ndvi"):
            st.divider()
            st.markdown("#### " + t("Vegetation health (NDVI)", "Vegetatievitaliteit (NDVI)"))
            f = veg["ndvi"]
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric(f"NDVI {f['future_years'][0]}", f"{f['future_values'][0]:.3f}",
                       delta=f"[{f['future_lower80'][0]:.3f}, {f['future_upper80'][0]:.3f}] 80% CI",
                       delta_color="off")
            fc2.metric(t("Trend", "Trend"), f"{f['slope_per_year']:+.4f}/yr")
            fc3.metric("R²", f"{f['r_squared']:.2f}")
            st.altair_chart(
                forecast_chart(f["history_years"], f["history_values"], f["future_years"], f["future_values"],
                                f["future_lower80"], f["future_upper80"], "NDVI", COLOR_SENTINEL2, value_fmt=".3f"),
                use_container_width=True,
            )
            st.caption(t(
                f"R²={f['r_squared']:.2f} — the straight line explains only {f['r_squared'] * 100:.0f}% of "
                "this series' year-to-year variation (real weather noise, see Trends & Climate), so read "
                "the shaded interval, not the dashed line itself, as the honest forecast.",
                f"R²={f['r_squared']:.2f} — de rechte lijn verklaart maar {f['r_squared'] * 100:.0f}% van "
                "de jaar-op-jaar variatie in deze reeks (echte weersruis, zie Trends & Klimaat), dus lees "
                "de gearceerde band, niet de gestippelde lijn zelf, als de eerlijke voorspelling.",
            ))
            if veg.get("ndwi"):
                with st.expander(t("NDWI forecast too", "Ook NDWI-voorspelling")):
                    fw = veg["ndwi"]
                    st.altair_chart(
                        forecast_chart(fw["history_years"], fw["history_values"], fw["future_years"], fw["future_values"],
                                        fw["future_lower80"], fw["future_upper80"], "NDWI", COLOR_LANDSAT, value_fmt=".3f", height=180),
                        use_container_width=True,
                    )

        lc = forecast.get("landcover") or {}
        if lc:
            st.divider()
            _lc_last_year = next(iter(lc.values()))["future_years"][-1]
            st.markdown("#### " + t(f"Land cover, projected to {_lc_last_year}", f"Landgebruik, geprojecteerd tot {_lc_last_year}"))
            st.caption(t(
                "Same four categories and common-footprint methodology as the Trends & Climate tab's "
                "historical series — Forest and Water are the more spectrally stable, more trustworthy "
                "projections; Built-up specifically inherits that series' own real classifier-wobble "
                "caveat, now compounded by a wide extrapolation interval on top of it.",
                "Dezelfde vier categorieën en common-footprint-methodologie als de historische reeks in "
                "Trends & Klimaat — Bos en Water zijn de spectraal stabielere, betrouwbaardere projecties; "
                "Bebouwd erft specifiek de eigen classifier-wiebel-kanttekening van die reeks, nu nog eens "
                "vergroot door een breed extrapolatie-interval.",
            ))
            _lc_cols = st.columns(len(lc))
            for _lc_col, (cat, f) in zip(_lc_cols, lc.items()):
                with _lc_col:
                    st.markdown(f"**{cat}**")
                    st.altair_chart(
                        forecast_chart(f["history_years"], f["history_values"], f["future_years"], f["future_values"],
                                        f["future_lower80"], f["future_upper80"], "ha", LANDCOVER_TREND_COLORS.get(cat, "#51604F"), height=170),
                        use_container_width=True,
                    )
                    st.caption(f"{f['future_years'][-1]}: {f['future_values'][-1]:,.0f} ha")

        pop = forecast.get("population") or {}
        if pop:
            st.divider()
            st.markdown("#### " + t("Population & housing stock", "Bevolking & woningvoorraad"))
            st.caption(t(
                "Real CBS registry counts (same source as the Villages tab), extrapolated — housing stock "
                "in particular is a near-straight administrative series (R² shown below), so this is the "
                "single most defensible forecast on this whole tab.",
                "Echte CBS-registratieaantallen (zelfde bron als het tabblad Kernen), geëxtrapoleerd — "
                "vooral de woningvoorraad is een bijna rechte administratieve reeks (R² hieronder), dus dit "
                "is de meest verdedigbare voorspelling op dit hele tabblad.",
            ))
            pc1, pc2 = st.columns(2)
            if pop.get("population"):
                f = pop["population"]
                with pc1:
                    st.altair_chart(
                        forecast_chart(f["history_years"], f["history_values"], f["future_years"], f["future_values"],
                                        f["future_lower80"], f["future_upper80"], t("Residents", "Inwoners"), COLOR_SENTINEL2),
                        use_container_width=True,
                    )
                    st.caption(t(
                        f"R²={f['r_squared']:.2f} · {f['future_years'][-1]}: {f['future_values'][-1]:,.0f} "
                        f"[{f['future_lower80'][-1]:,.0f}, {f['future_upper80'][-1]:,.0f}]",
                        f"R²={f['r_squared']:.2f} · {f['future_years'][-1]}: {f['future_values'][-1]:,.0f} "
                        f"[{f['future_lower80'][-1]:,.0f}, {f['future_upper80'][-1]:,.0f}]",
                    ))
            if pop.get("housing_stock"):
                f = pop["housing_stock"]
                with pc2:
                    st.altair_chart(
                        forecast_chart(f["history_years"], f["history_values"], f["future_years"], f["future_values"],
                                        f["future_lower80"], f["future_upper80"], t("Homes", "Woningen"), COLOR_TEMP),
                        use_container_width=True,
                    )
                    st.caption(t(
                        f"R²={f['r_squared']:.2f} · {f['future_years'][-1]}: {f['future_values'][-1]:,.0f} "
                        f"[{f['future_lower80'][-1]:,.0f}, {f['future_upper80'][-1]:,.0f}]",
                        f"R²={f['r_squared']:.2f} · {f['future_years'][-1]}: {f['future_values'][-1]:,.0f} "
                        f"[{f['future_lower80'][-1]:,.0f}, {f['future_upper80'][-1]:,.0f}]",
                    ))

        cr = forecast.get("crop_rotation") or {}
        if cr.get("transition_matrix"):
            st.divider()
            st.markdown("#### " + t("Crop rotation: what's likely next, per field", "Gewasrotatie: wat waarschijnlijk volgt, per perceel"))
            st.caption(t(
                f"Not a trend line — a first-order Markov chain: the empirical probability that a field "
                f"grown as one crop family is each other family the following year, from "
                f"{cr['n_transitions_observed']:,} real transitions in this pipeline's own matched "
                "BRP history (crop_rotation.py). Projected area below is an expected value — each field's "
                "own area spread across next year's families by its own family's observed probabilities, "
                "not a hard per-field call.",
                f"Geen trendlijn — een Markov-keten van de eerste orde: de empirische kans dat een perceel "
                f"met de ene gewasfamilie het jaar erop een andere familie heeft, op basis van "
                f"{cr['n_transitions_observed']:,} echte overgangen in de eigen gematchte BRP-geschiedenis "
                "van deze pipeline (crop_rotation.py). De voorspelde oppervlakte hieronder is een "
                "verwachtingswaarde — de oppervlakte per perceel verdeeld over de families van volgend "
                "jaar volgens de eigen waargenomen kansen van die familie, geen harde keuze per perceel.",
            ))
            _rot_brp_gdf = load_brp_gdf()
            if _rot_brp_gdf is not None:
                _fam_series = _rot_brp_gdf.apply(lambda r: classify_crop(r["gewas"], r["category"]), axis=1)
                _current_family_ha = _rot_brp_gdf.groupby(_fam_series)["area_ha"].sum().round(1).to_dict()
                st.altair_chart(
                    crop_family_forecast_chart(_current_family_ha, cr["projected_area_ha_next_year"]),
                    use_container_width=True,
                )
                st.info(t(
                    "**See it on the map, per field:** switch Field Explorer's \"Colour fields by\" to "
                    "**Predicted next crop (ML)** — each field is coloured by its own most likely next-year "
                    "family, with the predicted probability in its tooltip.",
                    "**Bekijk het op de kaart, per perceel:** zet \"Percelen kleuren op\" in Perceelverkenner op "
                    "**Voorspeld volgend gewas (ML)** — elk perceel is gekleurd naar de eigen meest "
                    "waarschijnlijke gewasfamilie van volgend jaar, met de voorspelde kans in de tooltip.",
                ))
            else:
                _needs_brp_geometry(t("This chart", "Deze grafiek"))

# ======================================================================
with tab_env:
    st.subheader(t("Air quality — RIVM, 2024", "Luchtkwaliteit — RIVM, 2024"))
    no2, pm10, pm25, nh3 = air.get("NO2", {}), air.get("PM10", {}), air.get("PM25", {}), air.get("NH3", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NO₂", f"{no2.get('mean_ug_m3', 0)} µg/m³",
              delta=f"WHO: {no2.get('who_guideline_ug_m3', 0)} · {no2.get('pct_area_over_who_guideline', 0)}% {t('over', 'boven')}",
              delta_color="off")
    c2.metric("PM₁₀", f"{pm10.get('mean_ug_m3', 0)} µg/m³",
              delta=f"WHO: {pm10.get('who_guideline_ug_m3', 0)} · {pm10.get('pct_area_over_who_guideline', 0)}% {t('over', 'boven')}",
              delta_color="off")
    c3.metric("PM₂.₅", f"{pm25.get('mean_ug_m3', 0)} µg/m³",
              delta=f"WHO: {pm25.get('who_guideline_ug_m3', 0)} · {pm25.get('pct_area_over_who_guideline', 0)}% {t('over', 'boven')}",
              delta_color="off")
    if nh3:
        _nh3_prior = (nh3.get("prior_years") or {})
        _nh3_prior_year, _nh3_prior_entry = next(iter(_nh3_prior.items()), (None, None))
        _nh3_prior_val = _nh3_prior_entry.get("mean_ug_m3") if _nh3_prior_entry else None
        _nh3_delta = (f"{_nh3_prior_year}→{nh3.get('year')}: {nh3['mean_ug_m3'] - _nh3_prior_val:+.2f} µg/m³"
                      if _nh3_prior_val is not None else t("no WHO guideline for NH₃", "geen WHO-richtlijn voor NH₃"))
        c4.metric("NH₃", f"{nh3.get('mean_ug_m3', 0)} µg/m³", delta=_nh3_delta, delta_color="off")
        _bc1, _bc2 = st.columns([10, 1])
        with _bc2:
            method_popover("nh3")
    else:
        c4.metric("NH₃", "n/a")
    st.caption(t(
        "Modelled 1×1 km RIVM national air-quality grids (the same ones used in official NSL reporting), "
        "clipped to the municipal boundary — not raw satellite pixels, but the authoritative reference "
        "any satellite-based air product would be validated against. NH₃ is a separate RIVM product (the "
        "GCN concentration download, not the Atlas Leefomgeving WCS the other three come from) — see the "
        "ⓘ next to it for what that means for how much to trust it.",
        "Gemodelleerde 1×1 km RIVM-luchtkwaliteitsrasters (dezelfde die in de officiële NSL-rapportage "
        "worden gebruikt), uitgesneden op de gemeentegrens — geen ruwe satellietpixels, maar de "
        "gezaghebbende referentie waaraan elk satellietgebaseerd luchtproduct gevalideerd zou worden. NH₃ "
        "komt uit een apart RIVM-product (de GCN-concentratiedownload, niet de Atlas Leefomgeving-WCS "
        "waar de andere drie vandaan komen) — zie de ⓘ ernaast voor wat dat betekent voor de "
        "betrouwbaarheid.",
    ))

    if air_quality_trend.get("series", {}).get("NO2", {}).get("years"):
        st.markdown("#### " + t("12-year trend, 2013–2024", "12-jarige trend, 2013–2024"))
        st.caption(t(
            "Same RIVM service, every year it actually publishes (checked directly against its own "
            "capabilities list). EC (elemental carbon / soot) is included as the closest real combustion "
            "proxy this free service has — not a CO₂ or GHG number itself, see the note below.",
            "Dezelfde RIVM-dienst, elk jaar dat hij daadwerkelijk publiceert (rechtstreeks gecontroleerd "
            "tegen de eigen capabilities-lijst). EC (elementair koolstof / roet) is opgenomen als de "
            "dichtstbijzijnde echte verbrandingsproxy die deze gratis dienst heeft — geen CO₂- of "
            "broeikasgascijfer zelf, zie de kanttekening hieronder.",
        ))
        _aq_colors = {"NO2": "#eb6834", "PM10": "#eda100", "PM25": "#2a78d6", "EC": "#51604F"}
        _aq_cols = st.columns(4)
        for _aq_col, (_pol, _s) in zip(_aq_cols, air_quality_trend["series"].items()):
            if not _s.get("years"):
                continue
            with _aq_col:
                st.markdown(f"**{_pol.replace('PM25','PM₂.₅').replace('PM10','PM₁₀').replace('NO2','NO₂')}**")
                st.altair_chart(
                    air_quality_trend_chart(_s["years"], _s["mean_ug_m3"], _pol, _aq_colors.get(_pol, "#51604F")),
                    use_container_width=True,
                )
                net = _s["mean_ug_m3"][-1] - _s["mean_ug_m3"][0]
                st.caption(f"{net:+.1f} µg/m³ ({_s['years'][0]}→{_s['years'][-1]})")
        st.caption(t(
            "All four fell over this period — real, documented Dutch/EU air-quality improvement (cleaner "
            "vehicles, stricter emission standards), not a pipeline artifact. The 2020 dip (ringed) is the "
            "documented COVID-19 lockdown traffic drop, not noise — it recovers most of the way in 2021, "
            "consistent with traffic (not industry) driving it. **What this still can't show:** CO₂/GHG, "
            "and a multi-year NH₃ trend the way there is for these four — RIVM's GCN NH₃ concentration "
            "product (added below) only publishes two real years, 2024 and 2025, not a historical back-"
            "series. Checked directly, not assumed absent: municipal CO₂ (Klimaatmonitor) sits behind an "
            "authenticated OData API (confirmed: HTTP 401 \"Guest user group not found\").",
            "Alle vier daalden in deze periode — een echte, gedocumenteerde Nederlandse/Europese "
            "luchtkwaliteitsverbetering (schonere voertuigen, strengere emissie-eisen), geen "
            "pipeline-artefact. De dip van 2020 (omcirkeld) is de gedocumenteerde COVID-19-"
            "lockdownverkeersdaling, geen ruis — herstelt grotendeels in 2021, consistent met verkeer "
            "(niet industrie) als aandrijver. **Wat dit nog steeds niet kan laten zien:** CO₂/"
            "broeikasgassen, en een meerjarige NH₃-trend zoals bij deze vier — RIVM's GCN-NH₃-"
            "concentratieproduct (hieronder toegevoegd) publiceert maar twee echte jaren, 2024 en 2025, "
            "geen historische reeks. Rechtstreeks gecontroleerd, niet aangenomen als afwezig: "
            "gemeentelijke CO₂ (Klimaatmonitor) zit achter een geauthenticeerde OData-API (bevestigd: "
            "HTTP 401 \"Guest user group not found\").",
        ))

    st.info(t(
        "**NH₃ concentration is now real, measured data above — but it is not the deposition figure farm "
        "nitrogen permitting actually runs on.** RIVM's Atlas Leefomgeving WCS (used for NO₂/PM₁₀/PM₂.₅ "
        "above) genuinely has no NH₃ layer, confirmed again directly against its live capabilities list — "
        "but RIVM separately publishes NH₃ *concentration* (µg/m³) as an open, no-key-needed grid download "
        "(GCN, data.rivm.nl/data/gcn/), which is what the NH₃ card above uses: real 2024/2025 values for "
        "this exact municipality, not a placeholder. **What's still genuinely missing:** nitrogen "
        "*deposition* (mol N/ha/yr on nature areas) — a different quantity from concentration, published "
        "separately via RIVM's AERIUS/GDN product (aerius.nl) rather than this download, and still not "
        "fetched here. Concentration is a real, useful proxy for where ammonia in the air is elevated, but "
        "cannot substitute for an actual AERIUS calculation in a permitting decision.",
        "**NH₃-concentratie is hierboven nu echte, gemeten data — maar dat is niet het depositiecijfer "
        "waar stikstofvergunningverlening voor de landbouw op draait.** RIVM's Atlas Leefomgeving-WCS "
        "(gebruikt voor NO₂/PM₁₀/PM₂.₅ hierboven) heeft echt geen NH₃-laag, opnieuw rechtstreeks "
        "gecontroleerd tegen de live capabilities-lijst — maar RIVM publiceert NH₃-*concentratie* (µg/m³) "
        "apart als een open rasterdownload zonder sleutel (GCN, data.rivm.nl/data/gcn/), wat de "
        "NH₃-kaart hierboven gebruikt: echte 2024/2025-waarden voor precies deze gemeente, geen "
        "plaatshouder. **Wat nog echt ontbreekt:** stikstof*depositie* (mol N/ha/jr op natuurgebieden) — "
        "een andere grootheid dan concentratie, apart gepubliceerd via RIVM's AERIUS/GDN-product "
        "(aerius.nl) in plaats van deze download, en nog niet opgehaald. Concentratie is een echte, "
        "bruikbare indicatie van waar ammoniak in de lucht verhoogd is, maar kan geen echte "
        "AERIUS-berekening in een vergunningsbeslissing vervangen.",
    ))

    st.divider()
    st.subheader(t("Energy transition — CBS, 2024", "Energietransitie — CBS, 2024"))
    c1, c2, c3, c4 = st.columns(4)
    _solar_pct2 = (cbs.get("homes_with_solar_pct") or {}).get("value") or 0
    _gasfree_pct2 = (cbs.get("gas_free_homes_pct") or {}).get("value") or 0
    metric_with_bar(c1, t("Homes with solar", "Zonnepanelen"), f"{_solar_pct2:.0f}%", kpi_proportion_bar_svg(_solar_pct2))
    metric_with_bar(c2, t("Gas-free homes", "Aardgasvrije woningen"), f"{_gasfree_pct2:.0f}%", kpi_proportion_bar_svg(_gasfree_pct2))
    c3.metric(t("Avg. electricity use", "Gem. elektriciteitsverbruik"), f"{m(cbs,'avg_electricity_use_kwh')} kWh/jr")
    c4.metric(t("Avg. solar feed-in", "Gem. teruglevering zon"), f"{m(cbs,'avg_solar_feedback_kwh')} kWh/jr")
    st.caption(t(
        f"88% of homes are still gas-heated against 6% gas-free — a {m(cbs,'housing_stock')}-home stock "
        "with 46% rooftop solar adoption already, meaning the remaining transition is heating, not "
        "electricity generation.",
        f"88% van de woningen wordt nog met gas verwarmd tegenover 6% aardgasvrij — een woningvoorraad "
        f"van {m(cbs,'housing_stock')} met al 46% zonnepanelen, wat betekent dat de resterende transitie "
        "over verwarming gaat, niet over elektriciteitsopwekking.",
    ))

    st.error(t(
        "**Grid capacity (installed MW, congestion status) — identified, not wired up.** Netbeheer "
        "Nederland's Capaciteitskaart publishes exactly this at postcode-6 level, but as an interactive "
        "map / downloadable file, not a queryable API — confirmed directly rather than guessed. "
        "**Internet/broadband coverage** has no comparable clean open geodata in the Netherlands at all; "
        "that would need a telecom partner's own data. Both are real next steps, not silent omissions.",
        "**Netcapaciteit (geïnstalleerd MW, congestiestatus) — geïdentificeerd, nog niet aangesloten.** "
        "Netbeheer Nederland's Capaciteitskaart publiceert precies dit op postcode-6-niveau, maar als "
        "interactieve kaart/downloadbaar bestand, geen bevraagbare API — direct gecontroleerd, niet "
        "geraden. **Internet-/breedbanddekking** heeft in Nederland helemaal geen vergelijkbare schone "
        "open geodata; dat zou de eigen data van een telecompartner vergen. Beide zijn echte "
        "vervolgstappen, geen stille omissies.",
    ))

# ======================================================================
with tab_water:
    if flood_event.get("timeline"):
        st.success(t(
            "**Flood extent, documented Jan 2024 high water (Lobith peaked at 14.5–14.7 m NAP, a "
            "~1-in-5-year event) — change-detection method:** each date is compared against its own "
            f"pixel's normal backscatter (a per-pixel median built from "
            f"{len(flood_event.get('reference_dates', []))} ordinary-condition autumn 2023 passes), not "
            f"one fixed cutoff. Result: **{flood_event.get('pre_event_flooded_pct', 0):.1f}% → "
            f"{flood_event.get('peak_flooded_pct', 0):.1f}%** flagged as flooded, pre-event to peak "
            f"({flood_event.get('peak_date', '?')}, {flood_event.get('net_change_pct', 0):+.1f} points "
            "net) — a real, clearly-above-noise signal, unlike the old method's ≈0 net change below.",
            "**Overstromingsgebied, gedocumenteerd hoogwater jan. 2024 (Lobith piekte op 14,5–14,7 m NAP, "
            "een gebeurtenis van ~1-op-5-jaar) — verandering-detectiemethode:** elke datum wordt "
            f"vergeleken met de eigen normale terugkaatsing van die pixel (een mediaan per pixel op basis "
            f"van {len(flood_event.get('reference_dates', []))} passages onder normale omstandigheden in "
            f"het najaar van 2023), niet één vaste drempel. Resultaat: "
            f"**{flood_event.get('pre_event_flooded_pct', 0):.1f}% → "
            f"{flood_event.get('peak_flooded_pct', 0):.1f}%** gemarkeerd als overstroomd, van vóór de "
            f"gebeurtenis tot de piek ({flood_event.get('peak_date', '?')}, "
            f"{flood_event.get('net_change_pct', 0):+.1f} punten netto) — een reëel signaal, duidelijk "
            "boven de ruis, anders dan de ≈0 netto verandering van de oude methode hieronder.",
        ))
        st.markdown(f"**{t('Flooded share of the AOI through the event', 'Overstroomd aandeel van het gebied tijdens de gebeurtenis')}**")
        st.altair_chart(flood_timeline_chart(flood_event), use_container_width=True)
        _phases = " → ".join(f"{r['date']} ({r['phase']})" for r in flood_event["timeline"])
        st.caption(t(
            f"Phases: {_phases}. Pre-event ({flood_event.get('pre_event_flooded_pct', 0):.1f}%) isn't 0% "
            "— this method has real background noise (speckle, seasonal moisture drift between the "
            "reference dates and the event window), so read the *change* as the signal, not the absolute "
            "level. Peak lands a few days after Lobith's own gauge peak, consistent with a floodplain "
            "filling and draining on its own delay rather than tracking the river stage instantly — a "
            "real hydrological effect this series can now show, not something the old single-snapshot "
            "read could have caught either way.",
            f"Fasen: {_phases}. Vóór de gebeurtenis ({flood_event.get('pre_event_flooded_pct', 0):.1f}%) "
            "is niet 0% — deze methode heeft reële achtergrondruis (speckle, seizoensgebonden "
            "vochtverschil tussen de referentiedata en het gebeurtenisvenster), dus lees de *verandering* "
            "als het signaal, niet het absolute niveau. De piek valt een paar dagen na de eigen piek van "
            "het meetstation Lobith, wat past bij een uiterwaard die met eigen vertraging vult en "
            "leegloopt in plaats van de rivierstand direct te volgen — een reëel hydrologisch effect dat "
            "deze reeks nu kan laten zien, iets wat de oude momentopname sowieso niet had kunnen vangen.",
        ))
        st.divider()

    if flood:
        net = flood.get("newly_flooded_pct", 0) - flood.get("newly_dry_pct", 0)
        st.info(t(
            "**Old method (kept for comparison) — one fixed -17dB threshold, one post-peak date:** "
            f"{flood.get('newly_flooded_pct', 0):.1f}% of the AOI's clear ground went dry→wet, but "
            f"{flood.get('newly_dry_pct', 0):.1f}% went wet→dry — net change ≈ {net:+.1f}%, essentially "
            "zero. This is the limitation the change-detection result above was built to fix: a fixed "
            "absolute cutoff doesn't transfer cleanly across scenes shot in different conditions/orbits. "
            "The Ooijpolder/Millingerwaard floodplain being an engineered washland (built to absorb "
            "moderate high water within its existing footprint) is a separate, still-open explanation for "
            "why even the improved method's signal is a rise in extent rather than dramatic new inundation.",
            "**Oude methode (bewaard ter vergelijking) — één vaste -17dB-drempel, één datum na de piek:** "
            f"{flood.get('newly_flooded_pct', 0):.1f}% van de onbewolkte grond ging van droog naar nat, "
            f"maar {flood.get('newly_dry_pct', 0):.1f}% ging van nat naar droog — netto verandering ≈ "
            f"{net:+.1f}%, vrijwel nul. Dit is precies de beperking die het resultaat van de "
            "verandering-detectie hierboven moest oplossen: een vaste absolute drempel vertaalt zich niet "
            "goed tussen opnames onder verschillende omstandigheden/banen. Dat de "
            "Ooijpolder/Millingerwaard een aangelegde uiterwaard is (gebouwd om matig hoogwater binnen "
            "het bestaande gebied op te vangen) is een aparte, nog open verklaring voor waarom zelfs het "
            "signaal van de verbeterde methode een toename in oppervlak is en geen dramatische nieuwe "
            "overstroming.",
        ))
    c1, c2 = st.columns(2)
    c1.metric(t("SAR ↔ optical water agreement", "SAR ↔ optische overeenstemming water"), f"{cross.get('iou_pct', 0):.0f}% IoU",
              delta=f"SAR {cross.get('sar_water_pct', 0):.1f}% vs " + t("optical", "optisch") + f" {cross.get('optical_water_pct', 0):.1f}%",
              delta_color="off")
    c1.markdown(kpi_proportion_bar_svg(cross.get("iou_pct", 0)), unsafe_allow_html=True)
    c2.metric(t("Elevation range (AHN)", "Hoogtebereik (AHN)"), f"{ahn.get('elevation_min_m', 0):.0f}–{ahn.get('elevation_max_m', 0):.0f} m NAP",
              delta=t(f"canopy/roofs >15m over {ahn.get('ndsm_over_15m_pct', 0):.1f}% of ground",
                      f"bladerdak/daken >15m over {ahn.get('ndsm_over_15m_pct', 0):.1f}% van de grond"),
              delta_color="off")
    st.markdown(t(
        "- Still not done: validation against Waterschap Rivierenland's own gauge/extent data — the "
        "concept note's recommended starting point for turning this into a trusted product.",
        "- Nog niet gedaan: validatie tegen de eigen peilstok-/oppervlaktedata van Waterschap "
        "Rivierenland — het aanbevolen startpunt uit de concept-notitie om hier een vertrouwd product "
        "van te maken.",
    ))

# ======================================================================
with tab_methodology:
    st.subheader(t("How every number on this site is actually calculated",
                    "Hoe elk cijfer op deze site daadwerkelijk wordt berekend"))
    st.caption(t(
        "One entry per indicator: what it measures, the exact source and processing, and its known "
        "limitations — read directly from the code that computes it "
        "(`src/methodology.py`), not written from memory. Where the code itself doesn't pin something "
        "down, it's listed as an open question below rather than guessed. The same file generates "
        "`METHODOLOGY.md` in the repository root, so the two can't drift apart.",
        "Eén item per indicator: wat het meet, de exacte bron en verwerking, en de bekende beperkingen "
        "— rechtstreeks uitgelezen uit de code die het berekent (`src/methodology.py`), niet uit het "
        "geheugen geschreven. Waar de code zelf iets niet vastlegt, staat dat hieronder als open vraag "
        "in plaats van geraden. Hetzelfde bestand genereert `METHODOLOGY.md` in de hoofdmap van de "
        "repository, zodat de twee niet uit elkaar kunnen lopen.",
    ))
    i = 0 if LANG == "en" else 1
    for _e in METHOD_ENTRIES:
        with st.expander(f"{_e.title[i]}"):
            st.caption(_e.measures[i])
            st.markdown(f"**{t('Source', 'Bron')}:** {_e.source}")
            st.markdown(f"**{t('Date/period', 'Datum/periode')}:** {_e.date_range[i]}")
            st.markdown(f"**{t('Resolution', 'Resolutie')}:** {_e.resolution[i]}")
            st.markdown(f"**{t('Processing steps', 'Verwerkingsstappen')}:**")
            for _step in _e.processing[i]:
                st.markdown(f"- {_step}")
            st.markdown(f"**{t('Known limitations', 'Bekende beperkingen')}:**")
            for _lim in _e.limitations[i]:
                st.markdown(f"- {_lim}")
            if _e.open_questions[i]:
                st.warning(t("**Open question — not guessed:**", "**Open vraag — niet geraden:**"))
                for _q in _e.open_questions[i]:
                    st.markdown(f"- {_q}")
            st.caption(f"{t('Source code', 'Broncode')}: `{_e.code_ref}`")
    st.divider()
    st.caption(t(
        "Not yet covered here: crop rotation (see the Land & Crops tab's own bug-and-method writeup), "
        "the forecast models (see the Forecast tab's own \"how this is calculated\" expander), and the "
        "villages/soil/CBS-trend/land-cover-change modules — real methods, documented inline on their "
        "own tabs already rather than duplicated here twice.",
        "Hier nog niet behandeld: gewasrotatie (zie de eigen bug-en-methode-uitleg op het tabblad Land "
        "& Gewassen), de voorspellingsmodellen (zie de eigen \"hoe dit wordt berekend\"-uitklapper op "
        "het tabblad Voorspelling), en de modules voor kernen/bodem/CBS-trend/landgebruikverandering — "
        "echte methodes, al inline gedocumenteerd op hun eigen tabblad in plaats van hier dubbel.",
    ))

# ======================================================================
with tab_business:
    st.subheader(t("What this could actually be", "Wat dit daadwerkelijk zou kunnen worden"))
    st.markdown(t(
        "Everything in the other tabs is a **working pipeline on free public data** — nothing here "
        "needed a paid subscription to build. That's the pitch: the raw capability already exists, for "
        "free, for any Dutch municipality. The product is turning it into something the gemeente or a "
        "farmer actually operates against, on a schedule, instead of a one-off pull.",
        "Alles op de andere tabbladen is een **werkende pipeline op gratis openbare data** — niets "
        "hiervan vroeg om een betaald abonnement om te bouwen. Dat is de pitch: de ruwe mogelijkheid "
        "bestaat al, gratis, voor elke Nederlandse gemeente. Het product is dit omzetten in iets waar de "
        "gemeente of een boer daadwerkelijk mee werkt, op een vast schema, in plaats van een eenmalige "
        "download.",
    ))

    st.markdown("#### " + t("Who pays for what", "Wie betaalt waarvoor"))
    st.markdown(t(
        "| Buyer | What they'd actually use it for | Offer shape |\n"
        "|---|---|---|\n"
        "| **Gemeente Berg en Dal** | Built-edge/forest-encroachment monitoring, heat mapping, "
        "housing-growth tracking against the 76 new homes/yr baseline | Annual data contract or embedded "
        "dashboard, refreshed on each new cloud-free satellite pass |\n"
        "| **Individual farmers / LTO members** | Field-level NDVI stress alerts on their own registered "
        "BRP parcels; real NH₃ concentration by area (Environment & Energy tab) as a first ammonia "
        "signal, with an AERIUS-grade deposition number as the next step once that data-sharing "
        "conversation with RIVM happens | Per-farm subscription, priced per registered hectare |\n"
        "| **Waterschap Rivierenland** | The change-detection flood-extent series (Water tab) as a "
        "standing product, validated against their own gauge/extent data | Paid pilot to close that "
        "validation gap, then a standing monitoring contract |\n"
        "| **ARK Nature / Staatsbosbeheer** | Millingerwaard succession trend as a standing report "
        "instead of a one-off | Annual ecological monitoring report |\n",
        "| Koper | Waar ze het daadwerkelijk voor zouden gebruiken | Vorm van het aanbod |\n"
        "|---|---|---|\n"
        "| **Gemeente Berg en Dal** | Monitoring van bebouwingsrand/bosoprukking, hittekartering, "
        "volgen van woningbouwgroei t.o.v. de basislijn van 76 nieuwe woningen/jr | Jaarlijks "
        "datacontract of ingebed dashboard, ververst bij elke nieuwe wolkenvrije satellietpassage |\n"
        "| **Individuele boeren / LTO-leden** | NDVI-stressmeldingen op perceelniveau voor hun eigen "
        "geregistreerde BRP-percelen; echte NH₃-concentratie per gebied (tabblad Milieu & Energie) als "
        "eerste ammoniaksignaal, met een AERIUS-waardig depositiecijfer als volgende stap zodra dat "
        "gesprek over datadeling met RIVM plaatsvindt | Abonnement per bedrijf, geprijsd per "
        "geregistreerde hectare |\n"
        "| **Waterschap Rivierenland** | De verandering-detectie-overstromingsreeks (tabblad Water) als "
        "vast product, gevalideerd tegen hun eigen peilstok-/oppervlaktedata | Betaalde pilot om dat "
        "validatiehiaat te dichten, daarna een vast monitoringcontract |\n"
        "| **ARK Natuurontwikkeling / Staatsbosbeheer** | Successietrend Millingerwaard als vast "
        "rapport in plaats van eenmalig | Jaarlijks ecologisch monitoringrapport |\n",
    ))

    st.markdown("#### " + t("The gaps that are also the roadmap", "De hiaten die ook de routekaart zijn"))
    st.markdown(t(
        "1. **Nitrogen deposition (mol N/ha/yr) via RIVM's AERIUS/GDN product.** NH₃ *concentration* "
        "(µg/m³) is real data on the Environment & Energy tab now — RIVM's open GCN download, no "
        "conversation needed. *Deposition on nature areas*, the number farm-nitrogen permitting actually "
        "runs on, is a different RIVM product (aerius.nl) not available as an open download the way "
        "concentration is — closing this gap means a data-sharing conversation with RIVM/AERIUS, and it "
        "is what actually unlocks a farmer-facing nitrogen-permit product, the biggest commercial "
        "opportunity here given the scale of the Dutch nitrogen crisis.\n"
        "2. **Grid capacity via Liander/Netbeheer Nederland** — needed before any pitch involving new "
        "solar or business connections; currently a manual postcode lookup, not an integrated data feed.\n"
        "3. **Ground-truth validation of the flood-extent series against Waterschap Rivierenland's own "
        "gauge/extent data** — the multi-date change-detection method (Water tab) replaced the old "
        "single-snapshot fixed-threshold read; what's left is checking it against someone else's "
        "independent measurement before pitching it as a trusted product.\n"
        "4. **Higher-resolution height/canopy data beyond AHN's ~5-year national LiDAR refresh cycle.** "
        "AHN (used for elevation/canopy height here) is real LiDAR, but nationally flown, so any single "
        "municipality only gets a new pass every few years. A drone (UAV LiDAR/photogrammetry) survey, "
        "commissioned specifically for Berg en Dal, would give centimetre-scale, any-time-of-year detail "
        "AHN can't — e.g. tracking individual-tree canopy loss within a season, not just between national "
        "flights. No such commissioned drone dataset exists for this municipality today; this is a "
        "genuine future option, not something this pipeline currently has access to.\n"
        "5. **ML beyond the current trend/rotation models.** The Forecast tab already runs a real OLS "
        "trend projection and a Markov crop-rotation model on this pipeline's own historical data — both "
        "genuine statistics, not black boxes. Natural next steps, none built yet: a proper time-series "
        "model (e.g. per-field NDVI anomaly detection flagging fields that deviate from their own "
        "multi-year normal, useful as an early-stress alert) and a land-cover classifier trained on "
        "multiple years of this pipeline's own KMeans output plus BRP labels, rather than KMeans alone.",
        "1. **Stikstofdepositie (mol N/ha/jr) via RIVM's AERIUS/GDN-product.** NH₃-*concentratie* "
        "(µg/m³) is nu echte data op het tabblad Milieu & Energie — RIVM's open GCN-download, geen "
        "gesprek nodig. *Depositie op natuurgebieden*, het cijfer waar stikstofvergunningverlening voor "
        "de landbouw daadwerkelijk op draait, is een ander RIVM-product (aerius.nl) dat niet als open "
        "download beschikbaar is zoals concentratie — dit hiaat dichten vraagt om een gesprek over "
        "datadeling met RIVM/AERIUS, en is wat een stikstofvergunning-product voor boeren daadwerkelijk "
        "mogelijk maakt, gezien de omvang van de Nederlandse stikstofcrisis de grootste commerciële kans "
        "hier.\n"
        "2. **Netcapaciteit via Liander/Netbeheer Nederland** — nodig vóór elke pitch met nieuwe "
        "zonne- of bedrijfsaansluitingen; nu een handmatige postcode-opzoeking, geen geïntegreerde "
        "datafeed.\n"
        "3. **Validatie van de overstromingsreeks tegen de eigen peilstok-/oppervlaktedata van "
        "Waterschap Rivierenland** — de verandering-detectiemethode met meerdere data (tabblad Water) "
        "verving de oude momentopname met vaste drempel; wat rest is dit toetsen aan een onafhankelijke "
        "meting van iemand anders voordat het als vertrouwd product wordt gepitcht.\n"
        "4. **Hogere-resolutie hoogte-/bladerdakdata voorbij AHN's ~5-jaarlijkse landelijke LiDAR-cyclus.** "
        "AHN (gebruikt voor hoogte/bladerdak hier) is echte LiDAR, maar landelijk gevlogen, dus elke "
        "gemeente krijgt maar om de paar jaar een nieuwe opname. Een drone-opname (UAV-LiDAR/"
        "fotogrammetrie), speciaal voor Berg en Dal opgezet, zou centimeterschaal-detail geven dat AHN "
        "niet kan — bijv. bladerdakverlies per individuele boom binnen één seizoen volgen, niet alleen "
        "tussen landelijke vluchten. Zo'n opdracht-drone-dataset bestaat vandaag niet voor deze gemeente; "
        "dit is een echte toekomstoptie, geen data die deze pipeline nu al heeft.\n"
        "5. **ML voorbij de huidige trend-/rotatiemodellen.** Het tabblad Voorspelling draait al een "
        "echte OLS-trendprojectie en een Markov-gewasrotatiemodel op de eigen historische data van deze "
        "pipeline — beide echte statistiek, geen black box. Logische volgende stappen, nog niet gebouwd: "
        "een echt tijdreeksmodel (bijv. NDVI-afwijkingsdetectie per perceel dat percelen markeert die "
        "afwijken van hun eigen meerjarige normaal, bruikbaar als vroege-stressmelding) en een "
        "landgebruiksclassifier getraind op meerdere jaren van de eigen KMeans-output van deze pipeline "
        "plus BRP-labels, in plaats van KMeans alleen.",
    ))

    st.caption(t(
        "None of the above numbers are fabricated to make this pitch look better than the data supports "
        "— see the caveats on every other tab. That honesty is itself part of the offer: a client who "
        "checks the methodology finds it holds up.",
        "Geen van bovenstaande cijfers is verzonnen om deze pitch er beter uit te laten zien dan de data "
        "toelaat — zie de kanttekeningen op elk ander tabblad. Die eerlijkheid is zelf onderdeel van het "
        "aanbod: een klant die de methodologie controleert, ziet dat die standhoudt.",
    ))

st.caption(t(
    "Data: Sentinel-2 L2A, Sentinel-1 RTC & Landsat 5/7/8 Collection 2 (Planetary Computer) · AHN4 LiDAR, "
    "municipal boundary & BRP crop parcels (PDOK) · Population/energy statistics (CBS) · Air quality "
    "(RIVM) · Daily weather (KNMI). See README.md.",
    "Data: Sentinel-2 L2A, Sentinel-1 RTC & Landsat 5/7/8 Collection 2 (Planetary Computer) · AHN4 LiDAR, "
    "gemeentegrens & BRP-gewaspercelen (PDOK) · Bevolkings-/energiestatistieken (CBS) · Luchtkwaliteit "
    "(RIVM) · Dagelijkse weersdata (KNMI). Zie README.md.",
))
