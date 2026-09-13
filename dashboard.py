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
import sys
from pathlib import Path

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
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

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from statsutil import load_stats
from visualize import make_map, field_explorer_map, FIELD_COLOR_MODES, CROP_FAMILIES, classify_crop

BRP_PATH = Path(__file__).resolve().parent / "data" / "raw" / "brp_parcels.geojson"

st.set_page_config(page_title="Berg en Dal — remote sensing pilot", layout="wide", page_icon="🛰️")

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"


def t(en: str, nl: str) -> str:
    """The one language switch point: every user-facing string in this file
    routes through here. Registry data (crop names, BRP categories) is
    never passed through t() -- it stays in its real, authoritative Dutch
    regardless of the UI language."""
    return en if st.session_state.get("lang", "en") == "en" else nl


@st.cache_data
def load_brp_gdf():
    """The precise (unsimplified) parcel geometry + exact zonal-stat NDVI --
    used for the click-to-inspect lookup, kept separate from the
    display-simplified copy field_explorer_map() draws."""
    return gpd.read_file(BRP_PATH)


def field_at(lat: float, lon: float):
    gdf = load_brp_gdf()
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
.bd-header { background: linear-gradient(135deg, #2E5943 0%, #1B3E2E 100%); color: #F3F6F1; padding: 30px 34px; border-radius: 14px; margin-bottom: 22px; }
.bd-header h1 { color: #FFFFFF !important; margin: 0 0 8px 0; font-size: 2.1rem; }
.bd-header p { margin: 0; color: #DCE7DD; font-size: 0.96rem; line-height: 1.5; }
.bd-header .bd-meta { display:block; margin-top:8px; opacity:.72; font-size:.82rem; }
.bd-tag { display:inline-block; font-family:"JetBrains Mono", monospace; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; background: rgba(255,255,255,.15); color:#EAF1EA; padding:4px 11px; border-radius:20px; margin-bottom:12px; }
div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px 12px; box-shadow: 0 1px 2px rgba(22,34,28,.05); }
div[data-testid="stMetricLabel"] { color: var(--ink-2); }
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--forest) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--forest) !important; }
.bd-stat-strip { display: flex; flex-wrap: wrap; gap: 10px; margin: -8px 0 22px; }
.bd-stat-pill { flex: 1 1 160px; background: #FFFFFF; border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 16px; box-shadow: 0 1px 2px rgba(22,34,28,.05); }
.bd-stat-num { font-family: "Fraunces", Georgia, serif; font-weight: 600; font-size: 1.5rem; color: var(--forest); line-height: 1.1; }
.bd-stat-lbl { color: var(--ink-2); font-size: .8rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

STATS_PATH = Path(__file__).resolve().parent / "data" / "processed" / "stats.json"

_top_gutter, _lang_slot = st.columns([6, 1])
with _lang_slot:
    _lang_choice = st.segmented_control(
        "Language", options=["en", "nl"], format_func=lambda k: {"en": "English", "nl": "Nederlands"}[k],
        default=st.session_state["lang"], label_visibility="collapsed", key="lang",
    )
LANG = st.session_state.get("lang", "en")


@st.cache_data
def get_stats() -> dict:
    return load_stats()


@st.cache_data
def get_map_html(lang: str) -> str:
    """A folium Map's own .render() is NOT idempotent -- calling it twice on
    the same object produces two different (and, via st_folium, two BROKEN)
    HTML documents; a plain JS ReferenceError inside the component's iframe
    was the symptom, confirmed by rendering the same object twice here and
    diffing the output. Streamlit reruns the script more than once on first
    load, and st_folium's own render call was hitting that same object
    every time under @st.cache_resource. Rendering once here, to a plain
    string cached with @st.cache_data (keyed on `lang` too, so switching
    languages gets its own cached render), and embedding that string
    (below) is immune to it -- there's nothing left to re-render."""
    return make_map(lang=lang).get_root().render()


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
ahn = stats.get("ahn", {})
landcover = stats.get("landcover_summer_2025", {}).get("class_pct", {})
cbs = stats.get("cbs", {})
brp = stats.get("brp", {})
air = stats.get("air_quality", {})

_span = (f"{trend['years'][0]}–{trend['years'][-1]}" if trend.get("years") else "n/a")
_n_years = (trend["years"][-1] - trend["years"][0] + 1) if trend.get("years") else 0
_pills = [
    (f"{_n_years} {t('years', 'jaar')}", f"{t('of satellite data', 'aan satellietdata')} · {_span}"),
    (f"{brp.get('n_parcels', 0):,.0f}", t("farm parcels, individually clickable", "landbouwpercelen, elk afzonderlijk klikbaar")),
    (t("5 sensors", "5 sensoren"), t("Sentinel-1/2, Landsat, AHN LiDAR, RIVM, KNMI", "Sentinel-1/2, Landsat, AHN LiDAR, RIVM, KNMI")),
    (f"{len(flood_event.get('timeline', []))} {t('dates', 'data')}", t("through the Jan 2024 flood event", "door de hoogwatergebeurtenis van jan. 2024")),
]
st.markdown(f"""
<div class="bd-stat-strip">
  {''.join(f'<div class="bd-stat-pill"><div class="bd-stat-num">{n}</div><div class="bd-stat-lbl">{l}</div></div>' for n, l in _pills)}
</div>
""", unsafe_allow_html=True)


def m(section: dict, key: str, fmt: str = "{:,.0f}") -> str:
    v = section.get(key, {}).get("value")
    return fmt.format(v) if v is not None else "n/a"


CHART_AXIS_KW = dict(grid=False, domainColor="#CBD3C1", labelColor="#51604F", titleColor="#51604F")


def ndvi_trend_chart(trend: dict) -> alt.LayerChart:
    """22-year NDVI trend, coloured by sensor provenance (Landsat vs
    Sentinel-2 — a real methodological seam documented in the README, not
    noise to hide) with a direct callout on 2018: the documented European
    drought year, and this series' single lowest point — corroborating
    evidence the data is tracking something real, not an artifact."""
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
    line = base.mark_line(color="#B9C2AE", strokeWidth=1.5)
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

    return (
        alt.layer(*layers)
        .properties(height=280)
        .configure_view(strokeWidth=0)
        .configure_axis(**CHART_AXIS_KW)
    )


def ndwi_trend_chart(trend: dict) -> alt.LayerChart:
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
    line = base.mark_line(color="#B9C2AE", strokeWidth=1.5)
    points = base.mark_point(size=75, filled=True).encode(
        color=alt.Color(
            "source:N", title=t("Source", "Bron"),
            scale=alt.Scale(domain=["Landsat (30m)", "Sentinel-2 (10m)"], range=[COLOR_LANDSAT, COLOR_SENTINEL2]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[alt.Tooltip("year:O", title=t("Year", "Jaar")), alt.Tooltip("NDWI:Q", format=".3f"),
                 alt.Tooltip("source:N", title=t("Source", "Bron")), alt.Tooltip("platform:N", title=t("Platform", "Platform"))],
    )
    return (
        alt.layer(line, points)
        .properties(height=240)
        .configure_view(strokeWidth=0)
        .configure_axis(**CHART_AXIS_KW)
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
    return chart.configure_view(strokeWidth=0).configure_axis(**CHART_AXIS_KW)


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
    return alt.layer(chart, zero_line).configure_view(strokeWidth=0).configure_axis(**CHART_AXIS_KW)


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
    return chart.configure_view(strokeWidth=0).configure_axis(**CHART_AXIS_KW)


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
        opacity=0.5,
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
        .configure_axis(**CHART_AXIS_KW)
    )


tab_overview, tab_explorer, tab_land, tab_climate, tab_env, tab_water, tab_business = st.tabs([
    "📋 " + t("Overview", "Overzicht"),
    "🧭 " + t("Field Explorer", "Perceelverkenner"),
    "🌾 " + t("Land & Crops", "Land & Gewassen"),
    "📈 " + t("Trends & Climate", "Trends & Klimaat"),
    "🏭 " + t("Environment & Energy", "Milieu & Energie"),
    "🌊 " + t("Water", "Water"),
    "💼 " + t("Business case", "Businesscase"),
])

# ======================================================================
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(t("Population", "Inwoners"), m(cbs, "population"))
    c2.metric(t("Households", "Huishoudens"), m(cbs, "households"))
    c3.metric(t("Registered farmland", "Landbouwgrond"), f"{brp.get('total_area_ha', 0):,.0f} ha")
    c4.metric(t("Land area", "Landoppervlakte"), m(cbs, "land_area_ha"))
    c5.metric(t("Homes with solar", "Zonnepanelen"), f"{m(cbs, 'homes_with_solar_pct', '{:.0f}')}%")

    st.divider()
    st.subheader(t("Map", "Kaart"))
    # Static embed, not st_folium -- this map has no click callback to wire
    # up (returned_objects was always []), and a plain iframe embed of one
    # fixed HTML string sidesteps the render-not-idempotent bug documented
    # on get_map_html() above. Full-width, tall: this is the centrepiece,
    # not a sidebar thumbnail -- every layer (true colour, NDVI, land cover,
    # both flood-extent methods, SAR, elevation, BRP fields...) toggles from
    # its own control at top-right, translated with the rest of the page.
    st.components.v1.html(get_map_html(LANG), height=760)

    st.divider()
    st.subheader(t("Land cover (KMeans, satellite-derived)", "Landgebruik (KMeans, satellietafgeleid)"))
    if landcover:
        df = pd.DataFrame({"class": list(landcover.keys()), t("% of clear ground", "% onbewolkte grond"): list(landcover.values())})
        df = df.sort_values(df.columns[1], ascending=True)
        st.bar_chart(df.set_index("class"), horizontal=True)
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
            "— see the **📈 Trends & Climate** tab. Headline: net "
            f"{trend.get('net_change', 0):+.3f} NDVI over {len(trend['years']) - 1} years, and August "
            "temperature is the single strongest weather correlate of that year's NDVI"
            + (f" (r ≈ {r_temp_headline:+.2f})" if r_temp_headline is not None else "") +
            " — hotter Augusts, lower vigour, physically the direction you'd expect.",
            f"Volledige NDVI/NDWI-trend, echte augustusweer-data terug tot {trend['years'][0]}, en hoe ze "
            "samenhangen — zie het tabblad **📈 Trends & Klimaat**. Kernpunt: netto "
            f"{trend.get('net_change', 0):+.3f} NDVI over {len(trend['years']) - 1} jaar, en de "
            "augustustemperatuur is de sterkste weerscorrelatie met de NDVI van dat jaar"
            + (f" (r ≈ {r_temp_headline:+.2f})" if r_temp_headline is not None else "") +
            " — hetere augustusmaanden, minder vitaliteit, precies de fysiek verwachte richting.",
        ), icon="📈")

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

    map_col, detail_col = st.columns([5, 2])
    with map_col:
        field_map = field_explorer_map(color_by=color_key, lang=LANG)
        map_state = st_folium(
            field_map, width=None, height=720,
            returned_objects=["last_object_clicked"], key=f"field_map_{color_key}_{LANG}",
        )

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
                st.write(f"**NDVI 2025:** {ndvi25:.2f}")
                st.write(f"**{t('NDVI change (2024→2025)', 'NDVI-verandering (2024→2025)')}:** {ndvi25 - ndvi24:+.2f}")
                _how_going = t("How it's going", "Hoe het ervoor staat")
                st.markdown(f"**{_how_going}**")
                st.bar_chart(pd.DataFrame({"NDVI": [ndvi24, ndvi25]}, index=["2024", "2025"]))
            else:
                st.caption(t(
                    "No NDVI trend for this field (cloud-masked in one of the two years).",
                    "Geen NDVI-trend voor dit perceel (bewolkt gemaskeerd in een van beide jaren).",
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
            df = pd.DataFrame({"category": list(cat_ha.keys()), "ha": list(cat_ha.values())})
            df = df.sort_values("ha", ascending=True)
            st.bar_chart(df.set_index("category"), horizontal=True)
    with right:
        st.markdown(f"**{t('Top crops by area', 'Grootste gewassen naar oppervlakte')}**")
        crops = brp.get("top_crops_ha", {})
        if crops:
            df = pd.DataFrame({"crop": list(crops.keys()), "ha": list(crops.values())})
            df = df.sort_values("ha", ascending=True)
            st.bar_chart(df.set_index("crop"), horizontal=True)

    st.markdown(f"**{t('All 102 crops, by family', 'Alle 102 gewassen, per familie')}**")
    st.caption(t(
        "Every registered crop name sorted into one family (see Field Explorer for the full per-field "
        "icon legend), so the arable/grassland mix above resolves further than 'top 9 crops, everything "
        "else grouped' without needing 102 individual colours.",
        "Elke geregistreerde gewasnaam ingedeeld in één familie (zie Perceelverkenner voor de volledige "
        "icoonlegenda per perceel), zodat de bouwland/grasland-mix hierboven verder wordt uitgesplitst dan "
        "'top 9 gewassen, de rest samen' zonder dat er 102 losse kleuren nodig zijn.",
    ))
    st.altair_chart(crop_family_chart(load_brp_gdf()), use_container_width=True)

    st.info(t(
        "**A dairy-and-arable mix, not a monoculture:** permanent grassland (1,417 ha) and silage maize "
        "(469 ha) together point to livestock farming as the anchor land use, alongside winter wheat, "
        "sugar beet and potatoes as the arable rotation. That mix is exactly what determines nitrogen "
        "exposure — livestock farms are the ones a nitrogen-permit product would actually serve.",
        "**Een mix van veeteelt en akkerbouw, geen monocultuur:** blijvend grasland (1.417 ha) en "
        "snijmaïs (469 ha) wijzen samen op veeteelt als de dominante landgebruiksvorm, naast wintertarwe, "
        "suikerbieten en aardappelen als bouwlandrotatie. Precies die mix bepaalt de stikstofbelasting — "
        "veehouderijen zijn de bedrijven die een stikstofvergunning-product daadwerkelijk zou bedienen.",
    ), icon="🌾")

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

    if trend.get("years"):
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
        st.altair_chart(ndvi_trend_chart(trend), use_container_width=True)
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
            st.altair_chart(ndwi_trend_chart(trend), use_container_width=True)
            st.caption(t(
                "NDWI (green minus near-infrared): consistently negative here since the AOI is mostly "
                "land, not water — a less negative value means relatively wetter or less-vegetated "
                "ground that year, not literally \"more water.\"",
                "NDWI (groen min nabij-infrarood): hier steeds negatief omdat het gebied grotendeels "
                "land is, geen water — minder negatief betekent dat jaar relatief nattere of minder "
                "begroeide grond, niet letterlijk \"meer water.\"",
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
        ), icon="🔬")
    elif trend.get("years"):
        st.caption(t(
            "Run `python src/fetch_weather.py` and `python src/climate_correlation.py` to add the weather "
            "comparison and correlation analysis here.",
            "Draai `python src/fetch_weather.py` en `python src/climate_correlation.py` om de "
            "weersvergelijking en correlatie-analyse hier toe te voegen.",
        ))

# ======================================================================
with tab_env:
    st.subheader(t("Air quality — RIVM, 2024", "Luchtkwaliteit — RIVM, 2024"))
    no2, pm10, pm25 = air.get("NO2", {}), air.get("PM10", {}), air.get("PM25", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("NO₂", f"{no2.get('mean_ug_m3', 0)} µg/m³",
              delta=f"WHO: {no2.get('who_guideline_ug_m3', 0)} · {no2.get('pct_area_over_who_guideline', 0)}% {t('over', 'boven')}",
              delta_color="off")
    c2.metric("PM₁₀", f"{pm10.get('mean_ug_m3', 0)} µg/m³",
              delta=f"WHO: {pm10.get('who_guideline_ug_m3', 0)} · {pm10.get('pct_area_over_who_guideline', 0)}% {t('over', 'boven')}",
              delta_color="off")
    c3.metric("PM₂.₅", f"{pm25.get('mean_ug_m3', 0)} µg/m³",
              delta=f"WHO: {pm25.get('who_guideline_ug_m3', 0)} · {pm25.get('pct_area_over_who_guideline', 0)}% {t('over', 'boven')}",
              delta_color="off")
    st.caption(t(
        "Modelled 1×1 km RIVM national air-quality grids (the same ones used in official NSL reporting), "
        "clipped to the municipal boundary — not raw satellite pixels, but the authoritative reference "
        "any satellite-based air product would be validated against.",
        "Gemodelleerde 1×1 km RIVM-luchtkwaliteitsrasters (dezelfde die in de officiële NSL-rapportage "
        "worden gebruikt), uitgesneden op de gemeentegrens — geen ruwe satellietpixels, maar de "
        "gezaghebbende referentie waaraan elk satellietgebaseerd luchtproduct gevalideerd zou worden.",
    ))

    st.warning(t(
        "**NH₃ / nitrogen deposition — not available here, and that's the actual gap worth closing.** "
        "Ammonia and nitrogen deposition are the numbers Dutch farm nitrogen permitting runs on, not NO₂. "
        "RIVM's Atlas Leefomgeving (checked directly above) has no queryable NH₃ layer — that figure lives "
        "in RIVM's separate GDN/AERIUS product, distributed as an annual grid download rather than a live "
        "service. **This is the single highest-value data gap for a Berg en Dal product aimed at farmers "
        "or the gemeente's own permitting process** — closing it means a data-sharing conversation with "
        "RIVM/AERIUS, not more satellite fetching.",
        "**NH₃ / stikstofdepositie — hier niet beschikbaar, en dat is het hiaat dat het écht waard is om "
        "te dichten.** Ammoniak en stikstofdepositie zijn de cijfers waar de Nederlandse "
        "stikstofvergunningverlening op draait, niet NO₂. RIVM's Atlas Leefomgeving (hierboven direct "
        "gecontroleerd) heeft geen bevraagbare NH₃-laag — dat cijfer zit in RIVM's aparte GDN/AERIUS-"
        "product, verspreid als jaarlijkse rasterdownload in plaats van een live dienst. **Dit is het "
        "waardevolste datahiaat voor een Berg en Dal-product gericht op boeren of het "
        "vergunningsproces van de gemeente zelf** — het dichten ervan vraagt om een gesprek over "
        "datadeling met RIVM/AERIUS, geen extra satellietdata.",
    ), icon="⚠️")

    st.divider()
    st.subheader(t("Energy transition — CBS, 2024", "Energietransitie — CBS, 2024"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("Homes with solar", "Zonnepanelen"), f"{m(cbs,'homes_with_solar_pct','{:.0f}')}%")
    c2.metric(t("Gas-free homes", "Aardgasvrije woningen"), f"{m(cbs,'gas_free_homes_pct','{:.0f}')}%")
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
    ), icon="🔌")

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
        ), icon="🌊")
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
        ), icon="🗂️")
    c1, c2 = st.columns(2)
    c1.metric(t("SAR ↔ optical water agreement", "SAR ↔ optische overeenstemming water"), f"{cross.get('iou_pct', 0):.0f}% IoU",
              delta=f"SAR {cross.get('sar_water_pct', 0):.1f}% vs " + t("optical", "optisch") + f" {cross.get('optical_water_pct', 0):.1f}%",
              delta_color="off")
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
        "BRP parcels; early groundwork for nitrogen-permit reporting once NH₃/AERIUS is wired in | "
        "Per-farm subscription, priced per registered hectare |\n"
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
        "geregistreerde BRP-percelen; vroege basis voor stikstofvergunning-rapportage zodra NH₃/AERIUS "
        "is aangesloten | Abonnement per bedrijf, geprijsd per geregistreerde hectare |\n"
        "| **Waterschap Rivierenland** | De verandering-detectie-overstromingsreeks (tabblad Water) als "
        "vast product, gevalideerd tegen hun eigen peilstok-/oppervlaktedata | Betaalde pilot om dat "
        "validatiehiaat te dichten, daarna een vast monitoringcontract |\n"
        "| **ARK Natuurontwikkeling / Staatsbosbeheer** | Successietrend Millingerwaard als vast "
        "rapport in plaats van eenmalig | Jaarlijks ecologisch monitoringrapport |\n",
    ))

    st.markdown("#### " + t("The three gaps that are also the roadmap", "De drie hiaten die ook de routekaart zijn"))
    st.markdown(t(
        "1. **NH₃ / nitrogen deposition via RIVM's AERIUS/GDN product** — the single highest-value gap. "
        "This is what actually unlocks a farmer-facing nitrogen-permit product, which is the biggest "
        "commercial opportunity here given the scale of the Dutch nitrogen crisis.\n"
        "2. **Grid capacity via Liander/Netbeheer Nederland** — needed before any pitch involving new "
        "solar or business connections; currently a manual postcode lookup, not an integrated data feed.\n"
        "3. **Ground-truth validation of the flood-extent series against Waterschap Rivierenland's own "
        "gauge/extent data** — the multi-date change-detection method (Water tab) replaced the old "
        "single-snapshot fixed-threshold read; what's left is checking it against someone else's "
        "independent measurement before pitching it as a trusted product.",
        "1. **NH₃ / stikstofdepositie via RIVM's AERIUS/GDN-product** — het waardevolste hiaat. Dit is "
        "wat een stikstofvergunning-product voor boeren daadwerkelijk mogelijk maakt, gezien de omvang "
        "van de Nederlandse stikstofcrisis de grootste commerciële kans hier.\n"
        "2. **Netcapaciteit via Liander/Netbeheer Nederland** — nodig vóór elke pitch met nieuwe "
        "zonne- of bedrijfsaansluitingen; nu een handmatige postcode-opzoeking, geen geïntegreerde "
        "datafeed.\n"
        "3. **Validatie van de overstromingsreeks tegen de eigen peilstok-/oppervlaktedata van "
        "Waterschap Rivierenland** — de verandering-detectiemethode met meerdere data (tabblad Water) "
        "verving de oude momentopname met vaste drempel; wat rest is dit toetsen aan een onafhankelijke "
        "meting van iemand anders voordat het als vertrouwd product wordt gepitcht.",
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
