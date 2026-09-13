"""
Turns the processed rasters into things a person can actually look at:
static PNG maps (true colour, NDVI, land cover, year-over-year change) and
one interactive Leaflet map (via folium) with all of them as toggleable
layers over real OpenStreetMap basemap, framed on Berg en Dal.
"""
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import rasterio
from matplotlib import colormaps
from matplotlib.colors import Normalize, ListedColormap
from PIL import Image

from aoi import geometry_wgs84
from statsutil import load_stats

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANDCOVER_COLORS = {
    "Water": "#3B6E8A",
    "Dense vegetation, dark canopy": "#12331F",
    "Dense vegetation, mid canopy": "#1E4D34",
    "Dense vegetation, bright canopy": "#3E7A4F",
    "Dense vegetation (forest)": "#1E4D34",
    "Grass / farmland": "#A2712F",
    "Built-up / bare": "#B4A38A",
    "Sparse / transitional": "#D8CBAE",
}


def _bounds_wgs84(profile) -> list:
    """[[south,west],[north,east]] for folium.raster_layers.ImageOverlay,
    reprojected from the raster's own CRS (UTM 31N) to WGS84."""
    import geopandas as gpd
    from shapely.geometry import box

    b = rasterio.transform.array_bounds(profile["height"], profile["width"], profile["transform"])
    geom = gpd.GeoSeries([box(*b)], crs=profile["crs"]).to_crs(4326).iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    return [[miny, minx], [maxy, maxx]]


def true_color_png(label: str) -> tuple[Path, list]:
    with rasterio.open(RAW_DIR / f"{label}.tif") as src:
        blue, green, red = src.read(1), src.read(2), src.read(3)
        profile = src.profile.copy()

    def stretch(band, lo=2, hi=98):
        band = band.astype(np.float32)
        p_lo, p_hi = np.percentile(band[band > 0], [lo, hi]) if (band > 0).any() else (0, 1)
        return np.clip((band - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)

    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])
    alpha = np.where((red + green + blue) > 0, 255, 0).astype(np.uint8)
    rgba = np.dstack([(rgb * 255).astype(np.uint8), alpha])

    out_path = OUT_DIR / f"{label}_truecolor.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


def ndvi_png(label: str) -> tuple[Path, list]:
    with rasterio.open(PROC_DIR / f"{label}_indices.tif") as src:
        ndvi = src.read(1)
        valid = src.read(7)
        profile = src.profile.copy()

    norm = Normalize(vmin=-0.2, vmax=0.9)
    rgba = (colormaps["RdYlGn"](norm(ndvi)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid > 0, 220, 0).astype(np.uint8)

    out_path = OUT_DIR / f"{label}_ndvi.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


def landcover_png(label: str) -> tuple[Path, list, dict]:
    with rasterio.open(PROC_DIR / f"{label}_landcover.tif") as src:
        classes = src.read(1)
        profile = src.profile.copy()

    legend_path = PROC_DIR / f"{label}_landcover_legend.txt"
    id_to_name = {}
    for line in legend_path.read_text().strip().splitlines():
        cid, name = line.split("\t")
        id_to_name[int(cid)] = name

    palette = [LANDCOVER_COLORS.get(id_to_name.get(i, ""), "#999999") for i in range(len(id_to_name))]
    cmap = ListedColormap(palette)

    rgba = np.zeros((*classes.shape, 4), dtype=np.uint8)
    valid = classes != 255
    idx = np.clip(classes, 0, len(palette) - 1)
    colored = (cmap(idx) * 255).astype(np.uint8)
    rgba[..., :3] = colored[..., :3]
    rgba[..., 3] = np.where(valid, 210, 0).astype(np.uint8)

    out_path = OUT_DIR / f"{label}_landcover.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile), id_to_name


def ndvi_change_png(label_old: str, label_new: str) -> tuple[Path, list]:
    path = PROC_DIR / f"ndvi_change_{label_old}_to_{label_new}.tif"
    with rasterio.open(path) as src:
        delta = src.read(1)
        profile = src.profile.copy()

    norm = Normalize(vmin=-0.3, vmax=0.3)
    valid = ~np.isnan(delta)
    rgba = (colormaps["BrBG"](norm(np.nan_to_num(delta))) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 220, 0).astype(np.uint8)

    out_path = OUT_DIR / f"ndvi_change_{label_old}_to_{label_new}.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


def sar_png(label: str = "sar_2025") -> tuple[Path, list]:
    """Grayscale VV backscatter -- bright = rough/urban, dark = smooth/water."""
    with rasterio.open(PROC_DIR / f"{label}_indices.tif") as src:
        vv_db = src.read(1)
        valid = src.read(4)
        profile = src.profile.copy()

    norm = Normalize(vmin=-22, vmax=-2)
    rgba = (colormaps["gray"](norm(np.nan_to_num(vv_db, nan=-22))) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid > 0, 230, 0).astype(np.uint8)

    out_path = OUT_DIR / f"{label}_vv.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


def sar_water_png(label: str = "sar_2025") -> tuple[Path, list]:
    with rasterio.open(PROC_DIR / f"{label}_indices.tif") as src:
        water = src.read(3)
        valid = src.read(4)
        profile = src.profile.copy()

    rgba = np.zeros((*water.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0x3B, 0x6E, 0x8A
    rgba[..., 3] = np.where((water > 0) & (valid > 0), 220, 0).astype(np.uint8)

    out_path = OUT_DIR / f"{label}_water.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


def flood_extent_png(highwater_label: str = "sar_highwater_2024", baseline_label: str = "sar_2025") -> tuple[Path, list]:
    with rasterio.open(PROC_DIR / f"flood_extent_{highwater_label}.tif") as src:
        classes = src.read(1)
        profile = src.profile.copy()

    colors = {0: (0, 0, 0, 0), 1: (0x3B, 0x6E, 0x8A, 210), 2: (0xC0, 0x3B, 0x2E, 235), 3: (0xC9, 0xB4, 0x7A, 160)}
    rgba = np.zeros((*classes.shape, 4), dtype=np.uint8)
    for cid, rgba_val in colors.items():
        rgba[classes == cid] = rgba_val

    out_path = OUT_DIR / f"flood_extent_{highwater_label}.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


def flood_changedetect_png(event_label: str) -> tuple[Path, list]:
    """The improved flood product: per-pixel dB drop vs. the reference
    composite, thresholded -- see flood_event.py. `event_label` is one of
    the sar_event_YYYYMMDD labels flood_event.py writes."""
    with rasterio.open(PROC_DIR / f"flood_changedetect_{event_label}.tif") as src:
        flooded = src.read(1)
        profile = src.profile.copy()

    rgba = np.zeros((*flooded.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0xC0, 0x3B, 0x2E
    rgba[..., 3] = np.where(flooded == 1, 230, 0).astype(np.uint8)

    out_path = OUT_DIR / f"flood_changedetect_{event_label}.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


BRP_COLORS = {
    "Grasland": "#8CB369",
    "Bouwland": "#A2712F",
    "Natuurterrein": "#1E4D34",
    "Landschapselement": "#3E7A4F",
    "Braakland": "#C9B47A",
    "Overige": "#999999",
}


FIELD_COLOR_MODES = {
    "category": "Category · Categorie",
    "crop": "Crop · Gewas",
    "ndvi_2025": "NDVI 2025 (health · gezondheid)",
    "ndvi_change": "NDVI change 2024→2025 · verandering",
}


def _hex_from_cmap(value, vmin, vmax, cmap_name):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "#BBBBBB"
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    r, g, b, _ = colormaps[cmap_name](norm(value))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def field_explorer_map(color_by: str = "category", center: list | None = None):
    """One clickable field per BRP parcel, recoloured by whichever attribute
    the dashboard's selector is set to -- category, crop, or either date's
    NDVI. Built fresh per selection (cheap: ~3,700 features, simplified
    geometry, canvas renderer) rather than baking one fixed styling in, so
    "select what colors it" is a real Python-side rebuild, not a client-side
    trick layered on top of a static file.

    Geometry is simplified for *display only* -- the precise polygons (and
    their exact zonal-stat NDVI values) stay in data/raw/brp_parcels.geojson
    untouched; this only thins the copy that gets drawn.
    """
    import geopandas as gpd

    gdf = gpd.read_file(RAW_DIR / "brp_parcels.geojson")
    gdf_rd = gdf.to_crs(28992)
    gdf_rd["geometry"] = gdf_rd.geometry.simplify(4, preserve_topology=True)
    gdf = gdf_rd.to_crs(4326)

    top_crops = gdf["gewas"].value_counts().nlargest(9).index.tolist()
    gdf["crop_group"] = gdf["gewas"].where(gdf["gewas"].isin(top_crops), "Other · Overig")
    crop_palette = {crop: c for crop, c in zip(top_crops, [
        "#8CB369", "#A2712F", "#1E4D34", "#3E7A4F", "#C9B47A",
        "#6E9887", "#B4A38A", "#5B7B9A", "#D2A466",
    ])}
    crop_palette["Other · Overig"] = "#999999"

    def color_for(row):
        if color_by == "category":
            return BRP_COLORS.get(row["category"], "#999999")
        if color_by == "crop":
            return crop_palette.get(row["crop_group"], "#999999")
        if color_by == "ndvi_2025":
            return _hex_from_cmap(row.get("ndvi_2025"), 0.2, 0.9, "RdYlGn")
        if color_by == "ndvi_change":
            return _hex_from_cmap(row.get("ndvi_change"), -0.15, 0.15, "BrBG")
        return "#999999"

    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    if center is None:
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    # Precompute per-feature color + display text as plain properties, so
    # the whole layer can be added in ONE folium.GeoJson call with a
    # style_function that just reads feature["properties"], instead of one
    # Leaflet layer object per parcel (which is what actually hung the
    # browser the first time this was tried).
    gdf["_color"] = gdf.apply(color_for, axis=1)
    gdf["_ndvi25_txt"] = gdf["ndvi_2025"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    gdf["_ndvi_chg_txt"] = gdf["ndvi_change"].apply(lambda v: f"{v:+.2f}" if pd.notna(v) else "n/a")
    gdf["_area_txt"] = gdf["area_ha"].round(2)
    # Drop everything except what the layer actually renders/displays --
    # smaller payload, and NaN floats (unset ndvi on cloud-masked parcels)
    # never reach the embedded JSON at all.
    gdf = gdf[["geometry", "gewas", "category", "_area_txt", "_ndvi25_txt", "_ndvi_chg_txt", "_color"]]

    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap",
                    control_scale=True, prefer_canvas=True)

    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda f: {
            "fillColor": f["properties"]["_color"], "color": "#33332a",
            "weight": 0.4, "fillOpacity": 0.75,
        },
        highlight_function=lambda f: {"weight": 2, "color": "#16221C", "fillOpacity": 0.9},
        tooltip=folium.GeoJsonTooltip(
            fields=["gewas", "category", "_area_txt", "_ndvi25_txt", "_ndvi_chg_txt"],
            aliases=["Crop · Gewas", "Category · Categorie", "Area (ha) · Oppervlakte",
                     "NDVI 2025", "NDVI Δ 2024→25"],
            sticky=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["gewas", "category", "_area_txt", "_ndvi25_txt", "_ndvi_chg_txt"],
            aliases=["Crop · Gewas", "Category · Categorie", "Area (ha) · Oppervlakte",
                     "NDVI 2025", "NDVI Δ 2024→25"],
        ),
        name="fields",
    ).add_to(m)

    folium.GeoJson(
        geometry_wgs84().__geo_interface__, name="Municipal boundary",
        style_function=lambda x: {"fillOpacity": 0, "color": "#16221C", "weight": 2},
    ).add_to(m)
    return m


def brp_png(reference_label: str = "summer_2025") -> tuple[Path, list, dict]:
    """Real field boundaries, coloured by BRP category, rasterized onto the
    Sentinel-2 grid.

    An earlier version of this rendered the raw vector layer -- 3,787
    individual parcel polygons, each with its own tooltip -- straight into
    the page. That's the kind of thing that looks fine on a fast dev machine
    and then hangs the browser tab for whoever actually opens the file, so
    it's a raster overlay here instead, the same pattern as every other
    layer on this map. The per-crop detail lives in the dashboard's own
    crop-area chart instead of an in-map hover.
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    gdf = gpd.read_file(RAW_DIR / "brp_parcels.geojson")
    with rasterio.open(PROC_DIR / f"{reference_label}_indices.tif") as ref:
        ref_profile = ref.profile.copy()
    gdf = gdf.to_crs(ref_profile["crs"])

    categories = sorted(gdf["category"].unique())
    cat_to_id = {cat: i + 1 for i, cat in enumerate(categories)}  # 0 reserved for "no parcel here"
    shapes = [(geom, cat_to_id[cat]) for geom, cat in zip(gdf.geometry, gdf["category"])]

    classes = rasterize(
        shapes, out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"], fill=0, dtype="uint8",
    )

    rgba = np.zeros((*classes.shape, 4), dtype=np.uint8)
    for cat, cid in cat_to_id.items():
        hexcolor = BRP_COLORS.get(cat, "#999999").lstrip("#")
        r, g, b = (int(hexcolor[i:i+2], 16) for i in (0, 2, 4))
        mask = classes == cid
        rgba[mask] = (r, g, b, 200)

    out_path = OUT_DIR / "brp_categories.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(ref_profile), {cat: BRP_COLORS.get(cat, "#999") for cat in categories}


def air_quality_png(pollutant: str = "no2") -> tuple[Path, list]:
    with rasterio.open(RAW_DIR / f"{pollutant}.tif") as src:
        conc = src.read(1)
        nodata = src.nodata
        profile = src.profile.copy()

    valid = conc != nodata
    lo, hi = np.percentile(conc[valid], [2, 98]) if valid.any() else (0, 40)
    norm = Normalize(vmin=lo, vmax=hi)
    rgba = (colormaps["YlOrRd"](norm(np.where(valid, conc, lo))) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 190, 0).astype(np.uint8)

    out_path = OUT_DIR / f"{pollutant}.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


def elevation_png(coverage_id: str = "dtm_05m") -> tuple[Path, list]:
    with rasterio.open(RAW_DIR / f"{coverage_id}.tif") as src:
        dtm = src.read(1)
        nodata = src.nodata
        profile = src.profile.copy()

    valid = dtm != nodata
    norm = Normalize(vmin=8, vmax=95)  # matches the AOI's real NAP range, river to ridge crest
    rgba = (colormaps["terrain"](norm(np.where(valid, dtm, 8))) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 230, 0).astype(np.uint8)

    out_path = OUT_DIR / "ahn_elevation.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


def ndsm_png() -> tuple[Path, list]:
    """DSM - DTM: canopy height in the forest, roughly building height in
    the villages, near-zero over open ground."""
    with rasterio.open(RAW_DIR / "ndsm_05m.tif") as src:
        ndsm = src.read(1)
        nodata = src.nodata
        profile = src.profile.copy()

    valid = ndsm != nodata
    norm = Normalize(vmin=0, vmax=25)
    rgba = (colormaps["YlGn"](norm(np.clip(np.where(valid, ndsm, 0), 0, 25))) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid & (ndsm > 1.5), 220, 0).astype(np.uint8)  # hide bare-ground noise

    out_path = OUT_DIR / "ahn_ndsm.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


def make_map(label_new: str = "summer_2025", label_old: str = "summer_2024") -> folium.Map:
    """Builds the folium.Map object without saving it -- used directly by
    build_map() below and embedded live in dashboard.py via streamlit-folium."""
    tc_path, bounds = true_color_png(label_new)
    ndvi_path, _ = ndvi_png(label_new)
    lc_path, _, id_to_name = landcover_png(label_new)
    chg_path, _ = ndvi_change_png(label_old, label_new)
    vv_path, vv_bounds = sar_png()
    water_path, _ = sar_water_png()
    flood_path, flood_bounds = flood_extent_png()
    flood_cd_path = flood_cd_bounds = None
    peak_event = load_stats().get("flood_event", {}).get("peak_date")
    if peak_event and (PROC_DIR / f"flood_changedetect_sar_event_{peak_event.replace('-', '')}.tif").exists():
        flood_cd_path, flood_cd_bounds = flood_changedetect_png(f"sar_event_{peak_event.replace('-', '')}")
    elev_path, elev_bounds = elevation_png()
    ndsm_path, _ = ndsm_png()
    no2_path, no2_bounds = air_quality_png("no2")
    brp_path, brp_bounds, brp_colors = brp_png()

    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)

    folium.raster_layers.ImageOverlay(
        str(tc_path), bounds=bounds, name="True colour (Aug 2025)", opacity=1.0
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        str(ndvi_path), bounds=bounds, name="NDVI (Aug 2025)", opacity=0.85, show=False
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        str(lc_path), bounds=bounds, name="Land cover (KMeans, Aug 2025)", opacity=0.8, show=False
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        str(chg_path), bounds=bounds, name="NDVI change, Aug 2024 → 2025", opacity=0.85, show=False
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        str(vv_path), bounds=vv_bounds, name="SAR backscatter, VV (Aug 2025)", opacity=0.9, show=False
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        str(water_path), bounds=vv_bounds, name="SAR water mask (Aug 2025)", opacity=0.75, show=False
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        str(flood_path), bounds=flood_bounds,
        name="Flood extent (old method) — fixed threshold, Jan 2024 vs Aug 2025", opacity=0.85, show=False
    ).add_to(m)
    if flood_cd_path is not None:
        folium.raster_layers.ImageOverlay(
            str(flood_cd_path), bounds=flood_cd_bounds,
            name=f"Flood extent (change detection) — peak {peak_event}", opacity=0.85, show=False
        ).add_to(m)
    folium.raster_layers.ImageOverlay(
        str(elev_path), bounds=elev_bounds, name="Elevation, AHN DTM (m NAP)", opacity=0.85, show=False
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        str(ndsm_path), bounds=elev_bounds, name="Canopy / building height, AHN nDSM", opacity=0.85, show=False
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        str(no2_path), bounds=no2_bounds, name="Air quality — NO2 (RIVM, 2024)", opacity=0.75, show=False
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        str(brp_path), bounds=brp_bounds, name="Field boundaries & crops (BRP)", opacity=0.8, show=False
    ).add_to(m)

    folium.GeoJson(
        geometry_wgs84().__geo_interface__,
        name="Municipal boundary",
        style_function=lambda x: {"fillOpacity": 0, "color": "#16221C", "weight": 2},
    ).add_to(m)

    legend_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
        f'<span style="width:12px;height:12px;background:{LANDCOVER_COLORS.get(name,"#999")};'
        f'display:inline-block;border-radius:2px;"></span>{name}</div>'
        for name in sorted(set(id_to_name.values()))
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,.2); font-family: sans-serif; font-size: 12px;">
      <div style="font-weight:600; margin-bottom:4px;">Land cover (KMeans)</div>
      {legend_rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    flood_stats = load_stats().get("flood_event", {})
    net_change_txt = f"{flood_stats['net_change_pct']:+.1f}" if "net_change_pct" in flood_stats else "?"
    flood_legend_html = f"""
    <div style="position: fixed; bottom: 24px; right: 24px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,.2); font-family: sans-serif; font-size: 12px; max-width: 240px;">
      <div style="font-weight:600; margin-bottom:4px;">Flood extent (SAR) — two methods</div>
      <div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
        <span style="width:12px;height:12px;background:#3B6E8A;display:inline-block;border-radius:2px;"></span>
        Permanent water (old method, both dates)</div>
      <div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
        <span style="width:12px;height:12px;background:#C03B2E;display:inline-block;border-radius:2px;"></span>
        Newly flooded (either method)</div>
      <div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
        <span style="width:12px;height:12px;background:#C9B47A;display:inline-block;border-radius:2px;"></span>
        Old method: baseline water missed (threshold noise)</div>
      <div style="margin-top:6px; color:#666; font-size:10.5px;">
        Old method (one fixed -17dB cutoff, one post-peak date): ~0 net change.<br/>
        New method (per-pixel change detection vs. a 4-date normal-condition composite,
        {len(flood_stats.get('timeline', []))} dates through the event):
        {flood_stats.get('pre_event_flooded_pct', '?')}% &rarr; {flood_stats.get('peak_flooded_pct', '?')}%
        at peak ({net_change_txt} pts net) — see README/Water tab.</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(flood_legend_html))

    brp_legend_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
        f'<span style="width:12px;height:12px;background:{color};display:inline-block;border-radius:2px;"></span>{name}</div>'
        for name, color in brp_colors.items()
    )
    brp_legend_html = f"""
    <div style="position: fixed; bottom: 300px; left: 24px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,.2); font-family: sans-serif; font-size: 12px;">
      <div style="font-weight:600; margin-bottom:4px;">Field boundaries (BRP)</div>
      {brp_legend_rows}
      <div style="margin-top:4px; color:#666; font-size:10.5px;">Per-crop breakdown is on the dashboard.</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(brp_legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_map(label_new: str = "summer_2025", label_old: str = "summer_2024") -> Path:
    m = make_map(label_new, label_old)
    out_path = OUT_DIR / "bergendal_map.html"
    m.save(str(out_path))
    print(f"saved interactive map: {out_path}")
    return out_path


if __name__ == "__main__":
    build_map()
