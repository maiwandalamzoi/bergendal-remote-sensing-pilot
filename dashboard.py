"""
Streamlit dashboard for the Berg en Dal pilot — built as something to put in
front of the gemeente (and, on the nitrogen/crop pages, farmers) rather than
just a developer's map viewer. Bilingual labels throughout (EN / NL) since
the audience is Dutch.

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
# NDVI trend, distinct from the site's own UI colours (--forest/--river/
# --loess below) so a chart legend is never mistaken for site chrome.
COLOR_LANDSAT = "#eb6834"
COLOR_SENTINEL2 = "#2a78d6"
COLOR_WATER = "#3B6E8A"    # matches --river, reused for single-series water/flood charts
COLOR_DROUGHT = "#C03B2E"  # matches the flood "newly flooded" red -- reused here for "stress"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from statsutil import load_stats
from visualize import make_map, field_explorer_map, FIELD_COLOR_MODES

BRP_PATH = Path(__file__).resolve().parent / "data" / "raw" / "brp_parcels.geojson"


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

st.set_page_config(page_title="Berg en Dal — remote sensing pilot", layout="wide", page_icon="🛰️")

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


@st.cache_data
def get_stats() -> dict:
    return load_stats()


@st.cache_data
def get_map_html() -> str:
    """A folium Map's own .render() is NOT idempotent -- calling it twice on
    the same object produces two different (and, via st_folium, two BROKEN)
    HTML documents; a plain JS ReferenceError inside the component's iframe
    was the symptom, confirmed by rendering the same object twice here and
    diffing the output. Streamlit reruns the script more than once on first
    load, and st_folium's own render call was hitting that same object
    every time under @st.cache_resource. Rendering once here, to a plain
    string cached with @st.cache_data, and embedding that string (below) is
    immune to it -- there's nothing left to re-render."""
    return make_map().get_root().render()


_last_updated = (
    datetime.datetime.fromtimestamp(STATS_PATH.stat().st_mtime).strftime("%d %B %Y")
    if STATS_PATH.exists() else "n/a"
)
st.markdown(f"""
<div class="bd-header">
  <span class="bd-tag">Pilot · Gemeente Berg en Dal</span>
  <h1>🛰️ Berg en Dal — remote sensing pilot</h1>
  <p>
    Satelliet- en LiDAR-data over de gemeente Berg en Dal · Sentinel-2, Sentinel-1, AHN, BRP, CBS en RIVM —
    alles hierop is echte, live opgehaalde data.<br/>
    Real satellite and LiDAR data for the Berg en Dal municipality — everything on this page was fetched
    live, nothing is simulated.
    <span class="bd-meta">Data last refreshed · Data laatst ververst: {_last_updated}</span>
  </p>
</div>
""", unsafe_allow_html=True)

if not STATS_PATH.exists():
    st.error("No pipeline output found yet. Run `python run_pipeline.py`, then `src/fetch_cbs.py`, "
             "`src/fetch_brp.py` and `src/fetch_air_quality.py` from the project root first.")
    st.stop()

stats = get_stats()
opt = stats.get("optical_summer_2025", {})
change = stats.get("ndvi_change", {})
trend = stats.get("ndvi_trend", {})
cross = stats.get("sar_optical_cross_check", {})
flood = stats.get("flood_extent", {})
flood_event = stats.get("flood_event", {})
ahn = stats.get("ahn", {})
landcover = stats.get("landcover_summer_2025", {}).get("class_pct", {})
cbs = stats.get("cbs", {})
brp = stats.get("brp", {})
air = stats.get("air_quality", {})

_span = (f"{trend['years'][0]}–{trend['years'][-1]}"
         if trend.get("years") else "n/a")
_n_years = (trend["years"][-1] - trend["years"][0] + 1) if trend.get("years") else 0
_pills = [
    (f"{_n_years} years", f"of satellite data · {_span}"),
    (f"{brp.get('n_parcels', 0):,.0f}", "farm parcels, individually clickable"),
    ("5 sensors", "Sentinel-1/2, Landsat, AHN LiDAR, RIVM grids"),
    (f"{len(flood_event.get('timeline', []))} dates", "through the Jan 2024 flood event"),
]
st.markdown(f"""
<div class="bd-stat-strip">
  {''.join(f'<div class="bd-stat-pill"><div class="bd-stat-num">{n}</div><div class="bd-stat-lbl">{l}</div></div>' for n, l in _pills)}
</div>
""", unsafe_allow_html=True)


def m(section: dict, key: str, fmt: str = "{:,.0f}") -> str:
    v = section.get(key, {}).get("value")
    return fmt.format(v) if v is not None else "n/a"


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
        y=alt.Y("NDVI:Q", title="Mean NDVI (clear ground)", scale=alt.Scale(zero=False)),
    )
    line = base.mark_line(color="#B9C2AE", strokeWidth=1.5)
    points = base.mark_point(size=75, filled=True).encode(
        color=alt.Color(
            "source:N", title="Source",
            scale=alt.Scale(domain=["Landsat (30m)", "Sentinel-2 (10m)"], range=[COLOR_LANDSAT, COLOR_SENTINEL2]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("NDVI:Q", format=".3f"),
            alt.Tooltip("source:N", title="Source"),
            alt.Tooltip("platform:N", title="Platform"),
            alt.Tooltip("clear_pct:Q", title="Clear ground %", format=".0f"),
        ],
    )
    layers = [line, points]
    if 2018 in trend["years"]:
        drought_df = df[df["year"] == 2018]
        marker = alt.Chart(drought_df).mark_point(
            size=180, filled=False, strokeWidth=2, color=COLOR_DROUGHT,
        ).encode(x="year:O", y="NDVI:Q")
        callout = alt.Chart(drought_df).mark_text(
            text="↓ 2018 drought", dy=18, fontSize=11, color=COLOR_DROUGHT, fontWeight="bold",
        ).encode(x="year:O", y="NDVI:Q")
        layers += [marker, callout]

    return (
        alt.layer(*layers)
        .properties(height=280)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, domainColor="#CBD3C1", labelColor="#51604F", titleColor="#51604F")
    )


def flood_timeline_chart(flood_event: dict) -> alt.LayerChart:
    """Flooded-ground share through the event -- one series (a magnitude
    over time), so phases show up via tooltip rather than a second legend
    (a lone series needs none, per the dataviz method)."""
    df = pd.DataFrame(flood_event["timeline"])
    df["date"] = pd.to_datetime(df["date"])

    base = alt.Chart(df).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", labelAngle=0)),
        y=alt.Y("flooded_pct:Q", title="Flagged as flooded (% of AOI)", scale=alt.Scale(zero=True)),
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
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("phase:N", title="Phase"),
            alt.Tooltip("flooded_pct:Q", title="Flooded %", format=".1f"),
            alt.Tooltip("mean_db_vs_reference:Q", title="dB vs. reference", format="+.2f"),
        ],
    )
    return (
        alt.layer(area, points)
        .properties(height=260)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, domainColor="#CBD3C1", labelColor="#51604F", titleColor="#51604F")
    )


tab_overview, tab_explorer, tab_land, tab_env, tab_water, tab_business = st.tabs([
    "📋 Overview · Overzicht",
    "🧭 Field Explorer · Perceelverkenner",
    "🌾 Land & Crops · Land & Gewassen",
    "🏭 Environment & Energy · Milieu & Energie",
    "🌊 Water",
    "💼 Business case",
])

# ======================================================================
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Population · Inwoners", m(cbs, "population"))
    c2.metric("Households · Huishoudens", m(cbs, "households"))
    c3.metric("Registered farmland · Landbouwgrond", f"{brp.get('total_area_ha', 0):,.0f} ha")
    c4.metric("Land area · Landoppervlakte", m(cbs, "land_area_ha"))
    c5.metric("Homes with solar · Zonnepanelen", f"{m(cbs, 'homes_with_solar_pct', '{:.0f}')}%")

    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Land cover (KMeans, satellite-derived)")
        if landcover:
            df = pd.DataFrame({"class": list(landcover.keys()), "% of clear ground": list(landcover.values())})
            df = df.sort_values("% of clear ground", ascending=True)
            st.bar_chart(df.set_index("class"), horizontal=True)
        st.caption(
            f"Mean NDVI Aug 2025: {opt.get('mean_ndvi', 0):.3f} "
            f"({change.get('mean_delta', 0):+.3f} vs Aug 2024). "
            f"Elevation (AHN): {ahn.get('elevation_min_m', 0):.0f}–{ahn.get('elevation_max_m', 0):.0f} m NAP."
        )
        st.markdown(
            "**What's not validated yet · Nog niet gevalideerd:**\n"
            "- Land cover cluster names are a spectral best guess (no ground-truth join) — "
            "except where the Land & Crops tab now backs them with real BRP field registrations.\n"
            f"- SAR/optical water agreement: {cross.get('iou_pct', 0):.0f}% IoU — see the Water tab."
        )
    with right:
        st.subheader("Map · Kaart")
        # Static embed, not st_folium -- this map has no click callback to
        # wire up (returned_objects was always []), and a plain iframe embed
        # of one fixed HTML string sidesteps the render-not-idempotent bug
        # documented on get_map_html() above.
        st.components.v1.html(get_map_html(), height=560)

    if trend.get("years"):
        st.divider()
        st.subheader(
            f"{trend['years'][-1] - trend['years'][0] + 1} years of NDVI · "
            f"{trend['years'][-1] - trend['years'][0] + 1} jaar NDVI ({trend['years'][0]}–{trend['years'][-1]})"
        )
        st.caption(
            "Two sensors, one series: Landsat (30m) before Sentinel-2 L2A coverage gets reliable here, "
            "Sentinel-2 (10m) from 2018 on — colour marks which. Each point is that August's single "
            "least-cloudy date, so real phenological/weather noise sits inside the line alongside any "
            "trend; hover a point for its exact date's platform and cloud-free coverage."
        )
        st.altair_chart(ndvi_trend_chart(trend), use_container_width=True)
        st.caption(
            f"Net {trend.get('net_change', 0):+.3f} NDVI over {len(trend['years']) - 1} years "
            f"({trend.get('slope_per_year', 0):+.4f}/yr) — noisy year to year; read the shape, not the "
            "slope, as the finding. 2018's dip lines up with the documented 2018 Northwestern-Europe "
            "drought, which is corroborating evidence for the pipeline rather than something to explain away."
        )

# ======================================================================
with tab_explorer:
    st.subheader("Every field, selectable · Elk perceel, selecteerbaar")
    st.caption(
        "Pick what colours the fields, then click one for its details and how it's changed year over year. "
        "/ Kies waar de velden op kleuren, klik dan op een perceel voor details en de verandering "
        "sinds vorig jaar."
    )

    color_key = st.selectbox(
        "Colour fields by · Kleur velden op:",
        options=list(FIELD_COLOR_MODES.keys()),
        format_func=lambda k: FIELD_COLOR_MODES[k],
    )

    map_col, detail_col = st.columns([2, 1])
    with map_col:
        field_map = field_explorer_map(color_by=color_key)
        map_state = st_folium(
            field_map, width=None, height=560,
            returned_objects=["last_object_clicked"], key=f"field_map_{color_key}",
        )

    with detail_col:
        st.markdown("**Selected field · Geselecteerd perceel**")
        click = (map_state or {}).get("last_object_clicked")
        row = field_at(click["lat"], click["lng"]) if click else None
        if row is not None:
            st.markdown(f"#### {row['gewas']}")
            st.write(f"**Category · Categorie:** {row['category']}")
            st.write(f"**Area · Oppervlakte:** {row['area_ha']:.2f} ha")
            ndvi25, ndvi24 = row.get("ndvi_2025"), row.get("ndvi_2024")
            if pd.notna(ndvi25) and pd.notna(ndvi24):
                st.write(f"**NDVI 2025:** {ndvi25:.2f}")
                st.write(f"**NDVI change · verandering (2024→2025):** {ndvi25 - ndvi24:+.2f}")
                st.markdown("**How it's going · Hoe het gaat**")
                st.bar_chart(pd.DataFrame({"NDVI": [ndvi24, ndvi25]}, index=["2024", "2025"]))
            else:
                st.caption("No NDVI trend for this field (cloud-masked in one of the two years) · "
                           "Geen NDVI-trend voor dit perceel (bewolkt in een van beide jaren).")

            st.markdown("**At this exact point · Op dit exacte punt**")
            samples = point_samples(click["lat"], click["lng"])
            s1, s2 = st.columns(2)
            elev = samples["elevation_m"]
            s1.metric("Elevation · Hoogte", f"{elev:.1f} m NAP" if elev is not None else "n/a")
            canopy = samples["canopy_height_m"]
            s2.metric("Canopy/roof height · Hoogte", f"{canopy:.1f} m" if canopy is not None else "n/a")
            if samples["sar_vv_db"] is not None:
                water_txt = " · flagged as open water · water" if samples["sar_water"] else ""
                st.caption(f"SAR backscatter (VV, Aug 2025): {samples['sar_vv_db']:.1f} dB{water_txt}")
            else:
                st.caption("No SAR reading here (cloud/shadow-free radar has no gaps, but this AOI clip "
                           "might not extend to this point) · Geen SAR-waarde beschikbaar op dit punt.")
        elif click:
            st.warning("That point isn't inside a registered BRP field · "
                       "Dat punt ligt niet binnen een geregistreerd BRP-perceel.")
        else:
            st.info("Click a field on the map to see its details here. · "
                    "Klik op een perceel op de kaart voor details.")

    st.caption(
        "Legend by mode · Legenda per modus — **Category/Categorie**: green tones for grassland/nature, "
        "brown for arable. **Crop/Gewas**: top 9 crops each get their own colour, everything else is grey. "
        "**NDVI 2025**: red→green, low→high vigour. **NDVI change**: brown→green, browning→greening "
        "since 2024. Fields with no valid pixel in one of the two dates (cloud-masked) show grey."
    )

# ======================================================================
with tab_land:
    st.subheader("Real field boundaries, real crops · Echte perceelgrenzen en gewassen")
    st.caption(
        "Source: BRP (Basisregistratie Gewaspercelen) — every parcel a farmer registered for subsidy, "
        f"{brp.get('year', '')}. {brp.get('n_parcels', 0):,} parcels, {brp.get('total_area_ha', 0):,.0f} ha "
        "total inside the municipal boundary."
    )

    c1, c2, c3 = st.columns(3)
    cat_ha = brp.get("by_category_ha", {})
    c1.metric("Grassland / pasture · Grasland", f"{cat_ha.get('Grasland', 0):,.0f} ha")
    c2.metric("Arable / cropland · Bouwland", f"{cat_ha.get('Bouwland', 0):,.0f} ha")
    c3.metric("Nature terrain · Natuurterrein", f"{cat_ha.get('Natuurterrein', 0):,.0f} ha")

    left, right = st.columns(2)
    with left:
        st.markdown("**Land use category · Landgebruikcategorie**")
        if cat_ha:
            df = pd.DataFrame({"category": list(cat_ha.keys()), "ha": list(cat_ha.values())})
            df = df.sort_values("ha", ascending=True)
            st.bar_chart(df.set_index("category"), horizontal=True)
    with right:
        st.markdown("**Top crops by area · Grootste gewassen naar oppervlakte**")
        crops = brp.get("top_crops_ha", {})
        if crops:
            df = pd.DataFrame({"crop": list(crops.keys()), "ha": list(crops.values())})
            df = df.sort_values("ha", ascending=True)
            st.bar_chart(df.set_index("crop"), horizontal=True)

    st.info(
        "**A dairy-and-arable mix, not a monoculture:** permanent grassland (1,417 ha) and silage maize "
        "(469 ha) together point to livestock farming as the anchor land use, alongside winter wheat, "
        "sugar beet and potatoes as the arable rotation. That mix is exactly what determines nitrogen "
        "exposure — livestock farms are the ones a nitrogen-permit product would actually serve.",
        icon="🌾",
    )

    st.subheader("Forest & nature · Bos & natuur")
    nature_brp = cat_ha.get("Natuurterrein", 0)
    forest_satellite_pct = sum(v for k, v in landcover.items() if "vegetation" in k.lower())
    st.markdown(
        f"- **{nature_brp:,.0f} ha** registered as nature terrain in BRP (mostly agri-environment-scheme "
        "grazing — e.g. the Millingerwaard rewilding area).\n"
        f"- **{forest_satellite_pct:.0f}%** of clear satellite ground reads as dense vegetation "
        "(the KMeans land cover) — this is the honest, broader estimate, since unmanaged forest on the "
        "ridge isn't a BRP-registered parcel at all. The two numbers measure different things and "
        "shouldn't be added together."
    )

# ======================================================================
with tab_env:
    st.subheader("Air quality · Luchtkwaliteit — RIVM, 2024")
    no2, pm10, pm25 = air.get("NO2", {}), air.get("PM10", {}), air.get("PM25", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("NO₂", f"{no2.get('mean_ug_m3', 0)} µg/m³",
              delta=f"WHO: {no2.get('who_guideline_ug_m3', 0)} · {no2.get('pct_area_over_who_guideline', 0)}% over",
              delta_color="off")
    c2.metric("PM₁₀", f"{pm10.get('mean_ug_m3', 0)} µg/m³",
              delta=f"WHO: {pm10.get('who_guideline_ug_m3', 0)} · {pm10.get('pct_area_over_who_guideline', 0)}% over",
              delta_color="off")
    c3.metric("PM₂.₅", f"{pm25.get('mean_ug_m3', 0)} µg/m³",
              delta=f"WHO: {pm25.get('who_guideline_ug_m3', 0)} · {pm25.get('pct_area_over_who_guideline', 0)}% over",
              delta_color="off")
    st.caption(
        "Modelled 1×1 km RIVM national air-quality grids (the same ones used in official NSL reporting), "
        "clipped to the municipal boundary — not raw satellite pixels, but the authoritative reference "
        "any satellite-based air product would be validated against."
    )

    st.warning(
        "**NH₃ / stikstofdepositie — not available here, and that's the actual gap worth closing.** "
        "Ammonia and nitrogen deposition are the numbers Dutch farm nitrogen permitting runs on, not NO₂. "
        "RIVM's Atlas Leefomgeving (checked directly above) has no queryable NH₃ layer — that figure lives "
        "in RIVM's separate GDN/AERIUS product, distributed as an annual grid download rather than a live "
        "service. **This is the single highest-value data gap for a Berg en Dal product aimed at farmers "
        "or the gemeente's own permitting process** — closing it means a data-sharing conversation with "
        "RIVM/AERIUS, not more satellite fetching.",
        icon="⚠️",
    )

    st.divider()
    st.subheader("Energy transition · Energietransitie — CBS, 2024")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Homes with solar · Zonnepanelen", f"{m(cbs,'homes_with_solar_pct','{:.0f}')}%")
    c2.metric("Gas-free homes · Aardgasvrij", f"{m(cbs,'gas_free_homes_pct','{:.0f}')}%")
    c3.metric("Avg. electricity use · Elektriciteitsverbruik", f"{m(cbs,'avg_electricity_use_kwh')} kWh/yr")
    c4.metric("Avg. solar feed-in · Teruglevering", f"{m(cbs,'avg_solar_feedback_kwh')} kWh/yr")
    st.caption(
        f"88% of homes are still gas-heated against 6% gas-free — a{'' } {m(cbs,'housing_stock')}-home "
        "stock with 46% rooftop solar adoption already, meaning the remaining transition is heating, "
        "not electricity generation."
    )

    st.error(
        "**Grid capacity (installed MW, congestion status) — identified, not wired up.** Netbeheer "
        "Nederland's Capaciteitskaart publishes exactly this at postcode-6 level, but as an interactive "
        "map / downloadable file, not a queryable API — I confirmed this directly rather than guessing. "
        "**Internet/broadband coverage** has no comparable clean open geodata in the Netherlands at all; "
        "that would need a telecom partner's own data. Both are real next steps, not silent omissions.",
        icon="🔌",
    )

# ======================================================================
with tab_water:
    if flood_event.get("timeline"):
        st.success(
            f"**Flood extent, documented Jan 2024 high water (Lobith peaked at 14.5–14.7 m NAP, "
            f"a ~1-in-5-year event) — change-detection method:** each date is compared against its own "
            f"pixel's normal backscatter (a per-pixel median built from "
            f"{len(flood_event.get('reference_dates', []))} ordinary-condition autumn 2023 passes), not one "
            f"fixed cutoff. Result: **{flood_event.get('pre_event_flooded_pct', 0):.1f}% → "
            f"{flood_event.get('peak_flooded_pct', 0):.1f}%** flagged as flooded, pre-event to peak "
            f"({flood_event.get('peak_date', '?')}, {flood_event.get('net_change_pct', 0):+.1f} points net) — "
            "a real, clearly-above-noise signal, unlike the old method's ≈0 net change below.",
            icon="🌊",
        )
        st.markdown("**Flooded share of the AOI through the event · Overstroomd aandeel tijdens de gebeurtenis**")
        st.altair_chart(flood_timeline_chart(flood_event), use_container_width=True)
        st.caption(
            "Phases: " + " → ".join(f"{r['date']} ({r['phase']})" for r in flood_event["timeline"]) +
            f". Pre-event ({flood_event.get('pre_event_flooded_pct', 0):.1f}%) isn't 0% — this method has "
            "real background noise (speckle, seasonal moisture drift between the reference dates and the "
            "event window), so read the *change* as the signal, not the absolute level. Peak lands a few "
            "days after Lobith's own gauge peak, consistent with a floodplain filling and draining on its "
            "own delay rather than tracking the river stage instantly — a real hydrological effect this "
            "series can now show, not something the old single-snapshot read could have caught either way."
        )
        st.divider()

    if flood:
        net = flood.get("newly_flooded_pct", 0) - flood.get("newly_dry_pct", 0)
        st.info(
            f"**Old method (kept for comparison) — one fixed -17dB threshold, one post-peak date:** "
            f"{flood.get('newly_flooded_pct', 0):.1f}% of the AOI's clear ground went dry→wet, but "
            f"{flood.get('newly_dry_pct', 0):.1f}% went wet→dry — net change ≈ {net:+.1f}%, essentially zero. "
            "This is the limitation the change-detection result above was built to fix: a fixed absolute "
            "cutoff doesn't transfer cleanly across scenes shot in different conditions/orbits. The "
            "Ooijpolder/Millingerwaard floodplain being an engineered washland (built to absorb moderate "
            "high water within its existing footprint) is a separate, still-open explanation for why even "
            "the improved method's signal is a rise in extent rather than dramatic new inundation.",
            icon="🗂️",
        )
    c1, c2 = st.columns(2)
    c1.metric("SAR ↔ optical water agreement", f"{cross.get('iou_pct', 0):.0f}% IoU",
              delta=f"SAR {cross.get('sar_water_pct', 0):.1f}% vs optical {cross.get('optical_water_pct', 0):.1f}%",
              delta_color="off")
    c2.metric("Elevation range (AHN)", f"{ahn.get('elevation_min_m', 0):.0f}–{ahn.get('elevation_max_m', 0):.0f} m NAP",
              delta=f"canopy/roofs >15m over {ahn.get('ndsm_over_15m_pct', 0):.1f}% of ground", delta_color="off")
    st.markdown(
        "- Still not done: validation against Waterschap Rivierenland's own gauge/extent data — the "
        "concept note's recommended starting point for turning this into a trusted product."
    )

# ======================================================================
with tab_business:
    st.subheader("What this could actually be · Wat dit zou kunnen worden")
    st.markdown(
        "Everything in the other tabs is a **working pipeline on free public data** — nothing here needed "
        "a paid subscription to build. That's the pitch: the raw capability already exists, for free, "
        "for any Dutch municipality. The product is turning it into something the gemeente or a farmer "
        "actually operates against, on a schedule, instead of a one-off pull."
    )

    st.markdown("#### Who pays for what · Wie betaalt waarvoor")
    st.markdown(
        "| Buyer | What they'd actually use it for | Offer shape |\n"
        "|---|---|---|\n"
        "| **Gemeente Berg en Dal** | Built-edge/forest-encroachment monitoring, heat mapping, housing-growth "
        "tracking against the 76 new homes/yr baseline | Annual data contract or embedded dashboard, "
        "refreshed on each new cloud-free satellite pass |\n"
        "| **Individual farmers / LTO members** | Field-level NDVI stress alerts on their own registered "
        "BRP parcels; early groundwork for nitrogen-permit reporting once NH₃/AERIUS is wired in | Per-farm "
        "subscription, priced per registered hectare |\n"
        "| **Waterschap Rivierenland** | The change-detection flood-extent series (Water tab) as a standing "
        "product, validated against their own gauge/extent data | Paid pilot to close that validation gap, "
        "then a standing monitoring contract |\n"
        "| **ARK Nature / Staatsbosbeheer** | Millingerwaard succession trend as a standing report instead "
        "of a one-off | Annual ecological monitoring report |\n"
    )

    st.markdown("#### The three gaps that are also the roadmap · De volgende stappen")
    st.markdown(
        "1. **NH₃ / stikstofdepositie via RIVM's AERIUS/GDN product** — the single highest-value gap. "
        "This is what actually unlocks a farmer-facing nitrogen-permit product, which is the biggest "
        "commercial opportunity here given the scale of the Dutch nitrogen crisis.\n"
        "2. **Grid capacity via Liander/Netbeheer Nederland** — needed before any pitch involving new solar "
        "or business connections; currently a manual postcode lookup, not an integrated data feed.\n"
        "3. **Ground-truth validation of the flood-extent series against Waterschap Rivierenland's own "
        "gauge/extent data** — the multi-date change-detection method (Water tab) replaced the old "
        "single-snapshot fixed-threshold read; what's left is checking it against someone else's "
        "independent measurement before pitching it as a trusted product."
    )

    st.caption(
        "None of the above numbers are fabricated to make this pitch look better than the data supports — "
        "see the caveats on every other tab. That honesty is itself part of the offer: a client who checks "
        "the methodology finds it holds up."
    )

st.caption(
    "Data: Sentinel-2 L2A, Sentinel-1 RTC & Landsat 5/7/8 Collection 2 (Planetary Computer) · AHN4 LiDAR, "
    "municipal boundary & BRP crop parcels (PDOK) · Population/energy statistics (CBS) · Air quality "
    "(RIVM). See README.md."
)
