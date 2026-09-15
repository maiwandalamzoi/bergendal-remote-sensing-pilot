"""
Turns the processed rasters into things a person can actually look at:
static PNG maps (true colour, NDVI, land cover, year-over-year change) and
one interactive Leaflet map (via folium) with all of them as toggleable
layers over real OpenStreetMap basemap, framed on Berg en Dal.
"""
import datetime
import unicodedata
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import rasterio
from folium.plugins import Fullscreen, GroupedLayerControl, MarkerCluster
from matplotlib import colormaps
from matplotlib.colors import Normalize, ListedColormap
from PIL import Image

from aoi import geometry_wgs84
from statsutil import load_stats

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Redrawn from a first pass where "Built-up / bare" (#B4A38A) and "Grass /
# farmland" (#A2712F) sat too close in hue -- both tan/brown -- to read
# apart at a glance on the map or in a bar chart. Now each class gets a
# colour from its own distinct family, matching the convention a person
# already expects from any land-cover map (water=blue, built-up=neutral
# grey/stone -- never green or brown, which both already mean vegetation
# here -- vegetation=green, farmland=gold/tan): water stays this app's own
# --river blue; built-up moves to a clearly neutral warm grey; farmland
# moves to a real gold, distinct from both grey and the greens; the three
# canopy tiers use this app's own --forest green as their middle anchor,
# darker and lighter either side; sparse/transitional gets a pale sage
# that reads as "thin vegetation," not as farmland or bare ground.
LANDCOVER_COLORS = {
    "Water": "#3B6E8A",
    "Built-up / bare": "#8D8879",
    "Grass / farmland": "#C9A227",
    "Dense vegetation, dark canopy": "#12331F",
    "Dense vegetation, mid canopy": "#2E5943",
    "Dense vegetation, bright canopy": "#5B8F6E",
    "Dense vegetation (forest)": "#2E5943",
    "Sparse / transitional": "#B7C9A8",
}

# Dutch labels for the fixed English cluster names landcover_ml.py's own
# _label_clusters() produces (that pipeline stage has no bilingual
# convention -- it labels clusters from their own spectral signature, not
# UI text) -- kept here, next to the colours, as the one bilingual lookup
# every legend/chart that shows these names reads from, so a label and its
# translation can never drift apart across the map legend and the bar
# chart on the Overview tab.
LANDCOVER_LABELS_NL = {
    "Water": "Water",
    "Built-up / bare": "Bebouwd / kaal",
    "Grass / farmland": "Grasland / landbouw",
    "Dense vegetation, dark canopy": "Dicht groen, donker bladerdak",
    "Dense vegetation, mid canopy": "Dicht groen, middelhoog bladerdak",
    "Dense vegetation, bright canopy": "Dicht groen, licht bladerdak",
    "Dense vegetation (forest)": "Dicht groen (bos)",
    "Sparse / transitional": "IJl / overgangsgebied",
}


def landcover_label(name: str, lang: str = "en") -> str:
    return name if lang == "en" else LANDCOVER_LABELS_NL.get(name, name)


# Same hand-drawn Feather/Lucide-style line-icon convention as
# CROP_ICON_SVGS below -- plain inline SVG paths, no icon font/CDN/photo,
# each shape checked for what it actually reads as (a tree, not a shrub; a
# building, not a generic square) rather than picked for looking busy.
# "Dense vegetation" reuses one tree glyph across all three canopy tiers
# on purpose -- the tiers are the same *kind* of cover at different
# brightness, which the colour ramp already carries; three different tree
# shapes would imply a difference in kind that isn't there.
LANDCOVER_ICON_SVGS: dict[str, str] = {
    "Water": '<path d="M2 14c2-3 4-3 6 0s4 3 6 0 4-3 6 0"/><path d="M2 18c2-3 4-3 6 0s4 3 6 0 4-3 6 0"/>',
    "Built-up / bare": '<path d="M4 21V9l5-4 5 4v12"/><path d="M14 21v-8h6v8"/><line x1="2" y1="21" x2="22" y2="21"/>'
                        '<line x1="7" y1="13" x2="7" y2="13.01"/><line x1="7" y1="17" x2="7" y2="17.01"/>',
    "Grass / farmland": '<path d="M4 20c0-5 1-8 0-13"/><path d="M9 20c0-7 2-11 1-16"/>'
                         '<path d="M15 20c0-7-1-11 1-16"/><path d="M20 20c1-5 0-8 2-13"/>',
    "Dense vegetation, dark canopy": '<path d="M12 2 8 8h2l-3 5h2.5L6 19h12l-3.5-6H17l-3-5h2z"/><line x1="12" y1="19" x2="12" y2="22"/>',
    "Dense vegetation, mid canopy": '<path d="M12 2 8 8h2l-3 5h2.5L6 19h12l-3.5-6H17l-3-5h2z"/><line x1="12" y1="19" x2="12" y2="22"/>',
    "Dense vegetation, bright canopy": '<path d="M12 2 8 8h2l-3 5h2.5L6 19h12l-3.5-6H17l-3-5h2z"/><line x1="12" y1="19" x2="12" y2="22"/>',
    "Dense vegetation (forest)": '<path d="M12 2 8 8h2l-3 5h2.5L6 19h12l-3.5-6H17l-3-5h2z"/><line x1="12" y1="19" x2="12" y2="22"/>',
    "Sparse / transitional": '<path d="M6 20c-2-8 2-15 13-16 1 11-6 15-13 16z"/><path d="M7 19c3-4 6-7 11-13"/>',
}


def landcover_icon_svg(name: str, size: int = 14, stroke: str = "currentColor") -> str:
    body = LANDCOVER_ICON_SVGS.get(name, "")
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


def _bounds_wgs84(profile) -> list:
    """[[south,west],[north,east]] for folium.raster_layers.ImageOverlay,
    reprojected from the raster's own CRS (UTM 31N) to WGS84."""
    import geopandas as gpd
    from shapely.geometry import box

    b = rasterio.transform.array_bounds(profile["height"], profile["width"], profile["transform"])
    geom = gpd.GeoSeries([box(*b)], crs=profile["crs"]).to_crs(4326).iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    return [[miny, minx], [maxy, maxx]]


def _load_village(village: str | None):
    """Returns (geom_wgs84, bounds_wgs84) for a name from
    data/raw/villages.geojson (CBS's own "wijken"), or (None, None) if no
    village was given or it isn't found on disk. Centralizes the lookup
    that used to be duplicated inline in make_map()."""
    if not village:
        return None, None
    import geopandas as gpd

    villages_path = RAW_DIR / "villages.geojson"
    if not villages_path.exists():
        return None, None
    villages_gdf = gpd.read_file(villages_path)
    match = villages_gdf[villages_gdf["wijknaam"] == village]
    if not len(match):
        return None, None
    geom = match.geometry.iloc[0]
    vminx, vminy, vmaxx, vmaxy = match.total_bounds
    return geom, [[vminy, vminx], [vmaxy, vmaxx]]


def _slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name)


def _clip_to_village(png_path: Path, bounds: list, village_geom_wgs84, village_slug: str) -> Path:
    """"Snap and cut": zeroes the alpha channel of an already-rendered
    overlay PNG everywhere outside the village polygon, instead of leaving
    the whole municipality's imagery visible under just a zoom + outline.
    Works directly against the WGS84 `bounds` rectangle every *_png()
    function already returns -- the exact rectangle
    folium.raster_layers.ImageOverlay draws the PNG onto -- so it needs no
    access to the source raster's own CRS/transform, and the cut lines up
    with what's on screen by construction.

    Writes a village-suffixed sibling file rather than overwriting the
    whole-municipality PNG, so switching the sidebar back to "All of Berg
    en Dal" still has the uncut original to fall back to (and different
    villages don't clobber each other's cached file)."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    out_path = png_path.with_name(f"{png_path.stem}_{village_slug}{png_path.suffix}")
    img = np.array(Image.open(png_path).convert("RGBA"))
    h, w = img.shape[:2]
    (south, west), (north, east) = bounds
    transform = from_bounds(west, south, east, north, w, h)
    inside = rasterize(
        [(village_geom_wgs84, 1)], out_shape=(h, w), transform=transform, fill=0, dtype="uint8",
    ).astype(bool)
    img[..., 3] = np.where(inside, img[..., 3], 0)
    Image.fromarray(img).save(out_path)
    return out_path


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


# Class ids from landcover_trend.py's CHANGE_* constants, kept as plain
# ints here (not imported) so visualize.py doesn't need landcover_trend's
# own heavier dependency chain just for five colours.
LANDCOVER_CHANGE_COLORS = {
    0: (0, 0, 0, 0),          # no data
    1: (0, 0, 0, 0),          # no change -- transparent on purpose, so
                               # only the real change stands out rather
                               # than a solid colour wash over 80% of the
                               # municipality that didn't change class.
    2: (0xC0, 0x3B, 0x2E, 230),  # forest loss -- same red used for
                               # "newly flooded" elsewhere on this map;
                               # both mean "this got worse."
    3: (0x1B, 0xAF, 0x7A, 230),  # forest gain -- CROP_FAMILIES' own
                               # grassland green, reused for consistency.
    4: (0xED, 0xA1, 0x00, 220),  # built-up growth
    5: (0x9A, 0x8F, 0xC2, 180),  # other change -- muted, deliberately
                               # less prominent than the three categories
                               # this map exists to answer.
}


def landcover_change_png(first_year: int, last_year: int) -> tuple[Path, list]:
    """The spatial "where did the forest actually go" map -- see
    landcover_trend.build_change_map()'s own docstring for the method and
    why this is restricted to the Sentinel-2 era (2018-present), not the
    fuller 2005-present window the NDVI trend chart covers elsewhere in
    this dashboard."""
    path = PROC_DIR / f"landcover_change_{first_year}_{last_year}.tif"
    with rasterio.open(path) as src:
        classes = src.read(1)
        profile = src.profile.copy()

    rgba = np.zeros((*classes.shape, 4), dtype=np.uint8)
    for cid, rgba_val in LANDCOVER_CHANGE_COLORS.items():
        rgba[classes == cid] = rgba_val

    out_path = OUT_DIR / f"landcover_change_{first_year}_{last_year}.png"
    Image.fromarray(rgba).save(out_path)
    return out_path, _bounds_wgs84(profile)


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
    "category": ("Category", "Categorie"),
    "crop_family": ("Crop family", "Gewasfamilie"),
    "ndvi_2025": ("NDVI 2025 (health)", "NDVI 2025 (vitaliteit)"),
    "ndvi_change": ("NDVI change 2024→2025", "NDVI-verandering 2024→2025"),
    "predicted_next_family": ("Predicted next crop (ML)", "Voorspeld volgend gewas (ML)"),
}


# Every one of the 102 distinct BRP crop names gets sorted into one of eight
# families (plus an "other" catch-all) rather than a top-N-plus-grey scheme:
# with 102 real crop names, no palette can give each its own distinguishable
# hue (the dataviz method's hard cap for a choropleth-style all-pairs use is
# 3-4 slots), so colour here encodes the *family* -- validated CVD-safe hues,
# adjacent-pair-checked -- and the icon + tooltip carry the literal crop
# identity as a secondary encoding, per the same method's own escape valve
# for exactly this many-categories case. (color, icon, EN label, NL label)
CROP_FAMILIES: dict[str, tuple[str, str, str, str]] = {
    "grassland":  ("#1baf7a", "🌱", "Grassland & pasture", "Grasland"),
    "maize":      ("#eda100", "🌽", "Maize", "Mais"),
    "cereals":    ("#eb6834", "🌾", "Cereals & grains", "Granen"),
    "root":       ("#e34948", "🥔", "Root, bulb & tuber crops", "Wortel- en knolgewassen"),
    "vegetables": ("#e87ba4", "🥬", "Vegetables & legumes", "Groenten & peulvruchten"),
    "fruit":      ("#008300", "🍎", "Fruit, orchards & nuts", "Fruit & boomgaarden"),
    "cover":      ("#4a3aa7", "🍀", "Cover, fodder & oilseed crops", "Groenbemesters & oliehoudende gewassen"),
    "nature":     ("#2a78d6", "🌳", "Nature, landscape & water", "Natuur, landschap & water"),
    "other":      ("#999999", "❔", "Other / unclassified", "Overig"),
}


# Hand-drawn line icons (Feather/Lucide-style: single stroke, rounded
# caps, plain geometric shapes -- the same system dashboard.py's
# icon_svg() uses) for map contexts where real HTML renders: the crop
# marker pins and this file's own map legends. The CROP_FAMILIES emoji
# above stay as-is for plain-text contexts (Altair chart labels, Leaflet
# GeoJsonTooltip field values) where a custom SVG can't render at all --
# only Unicode text does there, so switching those to SVG would silently
# break them rather than look nicer.
# v2 -- redrawn after the first pass read as too abstract to tell apart
# at a glance (a plain circle-on-a-stick for "nature" was, fairly, read
# as a map pin -- the exact icon already used for villages elsewhere on
# this map). Verified by rendering all nine at both a large legibility
# check size and the real on-map 15px/white-on-colour badge size before
# committing to any of them (see the git history of this file for the
# throwaway preview page used to check).
CROP_ICON_SVGS: dict[str, str] = {
    "grassland":  '<path d="M4 20c0-5 1-8 0-13"/><path d="M9 20c0-7 2-11 1-16"/>'
                  '<path d="M15 20c0-7-1-11 1-16"/><path d="M20 20c1-5 0-8 2-13"/>',
    "maize":      '<path d="M12 3c2.5 0 4 2.5 4 6v9c0 3-1.8 5-4 5s-4-2-4-5V9c0-3.5 1.5-6 4-6z"/>'
                  '<circle cx="10" cy="8" r=".8" fill="currentColor"/><circle cx="14" cy="8" r=".8" fill="currentColor"/>'
                  '<circle cx="10" cy="11" r=".8" fill="currentColor"/><circle cx="14" cy="11" r=".8" fill="currentColor"/>'
                  '<circle cx="10" cy="14" r=".8" fill="currentColor"/><circle cx="14" cy="14" r=".8" fill="currentColor"/>'
                  '<circle cx="10" cy="17" r=".8" fill="currentColor"/><circle cx="14" cy="17" r=".8" fill="currentColor"/>'
                  '<path d="M9 5c-2-1-4-1-6 1"/>',  # cob + kernel rows + a husk leaf
    "cereals":    '<path d="M12 21V7"/><path d="M12 7l-3-3M12 7l3-3"/><path d="M12 10l-3-3M12 10l3-3"/>'
                  '<path d="M12 13l-3-3M12 13l3-3"/><path d="M12 16l-2.5-2.5M12 16l2.5-2.5"/>',  # wheat-ear fishbone
    "root":       '<path d="M7 11c-1-3 1-6 4-6s4 1 6 3 2 5 0 7-2 4-5 4-4-1-5-3-1-3 0-5z"/>'
                  '<circle cx="9.5" cy="10.5" r=".6" fill="currentColor"/><circle cx="14" cy="9.5" r=".6" fill="currentColor"/>'
                  '<circle cx="10.5" cy="15" r=".6" fill="currentColor"/>',  # irregular tuber + eyes
    "vegetables": '<path d="M6 20c-2-8 2-15 13-16 1 11-6 15-13 16z"/><path d="M7 19c3-4 6-7 11-13"/>',  # a leaf
    "fruit":      '<path d="M12 8c-1-2-3-3-5-2-3 1-4 5-2 9 1.5 3 4 5 7 5s5.5-2 7-5c2-4 1-8-2-9-2-1-4 0-5 2z"/>'
                  '<path d="M12 8V4"/><path d="M12 5c1.2-1.2 2.5-1.2 3.5-.3" stroke-width="1.3"/>',  # apple + stem/leaf
    "cover":      '<circle cx="9" cy="9" r="3"/><circle cx="15" cy="9" r="3"/><circle cx="12" cy="14" r="3"/>'
                  '<line x1="12" y1="17" x2="12" y2="21"/>',  # 3-leaflet clover + stem
    "nature":     '<path d="M12 2 8 8h2l-3 5h2.5L6 19h12l-3.5-6H17l-3-5h2z"/><line x1="12" y1="19" x2="12" y2="22"/>',
                  # a tiered conifer silhouette -- deliberately NOT a
                  # circle-on-a-stick, which reads as a map pin
    "other":      '<circle cx="12" cy="12" r="8" stroke-dasharray="3,3"/>'
                  '<path d="M9.5 9.5a2.5 2.5 0 0 1 4.6-1.4c.6.9.4 1.7-.4 2.4-.8.7-1.2 1.1-1.2 2" stroke-width="1.6"/>'
                  '<circle cx="12" cy="16.3" r=".4" fill="currentColor"/>',  # dashed circle + "?"
}


def crop_icon_svg(family: str, size: int = 16, stroke: str = "currentColor") -> str:
    body = CROP_ICON_SVGS.get(family, CROP_ICON_SVGS["other"])
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{stroke}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


def classify_crop(name: str, category: str) -> str:
    """Maps a raw BRP `gewas` name (Dutch, free-text-ish but drawn from a
    fixed registry vocabulary) to a CROP_FAMILIES key, by keyword rather than
    a 102-entry lookup table -- new crop names the registry adds later still
    land somewhere sensible instead of silently falling into "other".
    Checked against every one of this AOI's 102 actual crop names: ~97% of
    parcels get a specific family, the rest ("Groene braak" / fallow and a
    handful of unnamed "overige ..." catch-alls) are genuinely other.

    Accent-normalized before matching (NFKD-decompose, drop combining
    marks) -- a real bug found while building crop-rotation forecasting:
    BRP's historical GeoPackage archive spells maize "Maïs" (diaeresis)
    where the live WFS drops it ("Mais"; same fix crop_rotation.py's own
    normalize_crop() already applies for a different reason). Without
    this, `has("mais")` never matches "maïs, snij-" -- every archive-year
    maize parcel silently fell into "other" instead of "maize"."""
    decomposed = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()

    def has(*words: str) -> bool:
        return any(w in n for w in words)

    if "koolzaad" in n or "mosterd" in n or "deder" in n:
        return "cover"
    if has("mais"):
        return "maize"
    if has("tarwe", "gerst", "rogge, korrel", "granen", "spelt", "triticale", "boekweit", "haver"):
        return "cereals"
    if has("aardappel", "bieten", "biet,", "uien,", "wortel", "waspeen", "kroten", "cichorei"):
        return "root"
    if has("groenbemesting", "vanggewas", "lupine", "lupinen", "klaver", "luzerne", "miscanthus",
           "graszaad", "bloemzaden", "sierconiferen", "kerstbomen", "drachtplanten"):
        return "cover"
    if has("groente", "bonen", "boon,", "erwt", "peulen", "asperge", "prei,", "sla,", "pompoen",
           "bloemkwekerij", "droogbloemen"):
        return "vegetables"
    if has("appel", "peren.", "peren,", "pruim", "druiven", "bessen", "kleinfruit", "notenbomen",
           "boomgaard", "voedselbos"):
        return "fruit"
    if has("grasland"):
        return "grassland"
    if has("natuur", "bos", "hout", "struweel", "heg,", "haag", "sloot", "water,", "poel",
           "ruigte", "schurveling", "graften", "schouwpad", "bossingel", "wilgenhakhout",
           "woudbomen", "sorghum"):
        return "nature"
    if category == "Grasland":
        return "grassland"
    if category in ("Natuurterrein", "Landschapselement"):
        return "nature"
    return "other"


def _hex_from_cmap(value, vmin, vmax, cmap_name):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "#BBBBBB"
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    r, g, b, _ = colormaps[cmap_name](norm(value))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _report_header_element(lang: str, village: str | None) -> folium.Element:
    """The QGIS-print-composer-style title/scope/date/source strip shared
    by both maps -- a small pill on screen, a full title block on
    @media print (see the embedded <style>). Baked into the map's own
    HTML (not the surrounding Streamlit page) so it's part of what
    actually prints from inside the st.components.v1 iframe."""
    i = 0 if lang == "en" else 1
    scope_text = village or ("Berg en Dal (whole municipality)", "Berg en Dal (hele gemeente)")[i]
    report_title = ("Berg en Dal — remote sensing pilot", "Berg en Dal — aardobservatie-pilot")[i]
    source_line = "Sentinel-1/2 · Landsat · AHN LiDAR · RIVM · KNMI · BRP · CBS · ISRIC SoilGrids"
    report_date = datetime.date.today().strftime("%d %B %Y")
    return folium.Element(f"""
    <style>
      .bd-map-report {{
        position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
        z-index: 9998; background: rgba(255,255,255,.92); padding: 4px 14px;
        border-radius: 999px; box-shadow: 0 1px 4px rgba(0,0,0,.18);
        font-family: sans-serif; font-size: 11px; color: #16221C; white-space: nowrap;
      }}
      .bd-map-report .bd-mr-sub {{ display: none; }}
      @media print {{
        .bd-map-report {{
          position: static; transform: none; width: 100%; text-align: left;
          border-radius: 0; box-shadow: none; background: #fff; white-space: normal;
          padding: 0 0 10px; border-bottom: 2px solid #16221C; margin-bottom: 8px;
        }}
        .bd-map-report .bd-mr-title {{ font-size: 17px; font-weight: 700; }}
        .bd-map-report .bd-mr-sub {{ display: block; color: #444; font-size: 12px; margin-top: 3px; }}
        .leaflet-control-zoom, .leaflet-control-fullscreen {{ display: none !important; }}
      }}
    </style>
    <div class="bd-map-report">
      <span class="bd-mr-title">🛰️ {report_title}</span>
      <span class="bd-mr-sub">{scope_text} · {report_date} · {source_line}</span>
    </div>
    """)


def _north_arrow_element() -> folium.Element:
    """A static 'N ↑' badge -- both maps' basemap tiles (OpenStreetMap via
    Leaflet) are always plain north-up, so a fixed arrow is cartographically
    correct here without computing a real bearing."""
    return folium.Element("""
    <div style="position: fixed; top: 112px; left: 10px; z-index: 9998;
                background: white; width: 34px; height: 34px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,.25); display: flex; align-items: center;
                justify-content: center; flex-direction: column; font-family: sans-serif;">
      <div style="font-size: 14px; line-height: 1;">&#8593;</div>
      <div style="font-size: 9px; font-weight: 700; line-height: 1; margin-top: 1px;">N</div>
    </div>
    """)


def field_explorer_map(color_by: str = "category", center: list | None = None, lang: str = "en",
                        village: str | None = None):
    """One clickable field per BRP parcel, recoloured by whichever attribute
    the dashboard's selector is set to -- category, crop, or either date's
    NDVI. Built fresh per selection (cheap: ~3,700 features, simplified
    geometry, canvas renderer) rather than baking one fixed styling in, so
    "select what colors it" is a real Python-side rebuild, not a client-side
    trick layered on top of a static file.

    Geometry is simplified for *display only* -- the precise polygons (and
    their exact zonal-stat NDVI values) stay in data/raw/brp_parcels.geojson
    untouched; this only thins the copy that gets drawn.

    `village`, when given a name from data/raw/villages.geojson (CBS's own
    "wijken"), clips the fields shown to that village's real administrative
    boundary rather than a hand-drawn radius.

    `color_by="predicted_next_family"` colours each field by the most
    likely crop family a first-order Markov transition matrix (built once
    by forecast.py from every field's own real multi-year BRP rotation
    history) predicts for next year -- looked up here, not refit per
    request. See forecast.py's module docstring for the method and its
    honest limits.
    """
    import geopandas as gpd

    i = 0 if lang == "en" else 1
    gdf = gpd.read_file(RAW_DIR / "brp_parcels.geojson")
    village_geom = None
    if village:
        villages_path = RAW_DIR / "villages.geojson"
        if villages_path.exists():
            villages = gpd.read_file(villages_path)
            match = villages[villages["wijknaam"] == village]
            if len(match):
                village_geom = match.geometry.iloc[0]
                # Centroid-in-village test done in RD New (a projected,
                # metric CRS) -- the same fix already applied elsewhere in
                # this file (crop-icon markers, soil sampling) for the same
                # "plain lon/lat centroid is geometrically off" reason.
                gdf_for_filter = gdf.to_crs(28992)
                village_geom_rd = gpd.GeoSeries([village_geom], crs=4326).to_crs(28992).iloc[0]
                gdf = gdf[gdf_for_filter.geometry.centroid.within(village_geom_rd).values]
    gdf_rd = gdf.to_crs(28992)
    gdf_rd["geometry"] = gdf_rd.geometry.simplify(4, preserve_topology=True)

    # One emoji marker per field, on top of the colour fill -- the fill
    # carries the family, the icon carries the literal crop. Centroid taken
    # here, still in RD New (a projected, metric CRS), rather than after
    # reprojecting to WGS84 below, where a plain lat/lon centroid would be
    # geometrically off. Skipped below ~0.3ha (over half the registry --
    # mostly ditches, hedgerows, single trees) so the map reads as field
    # icons, not an icon soup.
    gdf_rd["_family"] = gdf_rd.apply(lambda r: classify_crop(r["gewas"], r["category"]), axis=1)
    gdf_rd["_icon"] = gdf_rd["_family"].map(lambda f: CROP_FAMILIES[f][1])

    # Predicted next crop family (ML): a first-order Markov transition
    # matrix over crop families, built once by forecast.py from every
    # field's own real multi-year BRP rotation history -- looked up here,
    # not refit per request. Falls back to "stays the same family, no
    # probability claimed" for any family with zero observed transitions
    # in that history (e.g. "other").
    _transition_matrix = load_stats().get("forecast", {}).get("crop_rotation", {}).get("transition_matrix", {})

    def _predict_next(fam: str) -> tuple[str, float | None]:
        row_probs = _transition_matrix.get(fam)
        if not row_probs:
            return fam, None
        best_fam = max(row_probs, key=row_probs.get)
        return best_fam, row_probs[best_fam]

    _pred = gdf_rd["_family"].map(_predict_next)
    gdf_rd["_pred_family"] = _pred.map(lambda p: p[0])
    gdf_rd["_pred_prob"] = _pred.map(lambda p: p[1])

    icon_points_rd = gdf_rd[gdf_rd["area_ha"] >= 0.3][["geometry", "_icon", "_family", "gewas", "area_ha"]].copy()
    icon_points_rd["geometry"] = icon_points_rd.geometry.centroid
    icon_gdf = icon_points_rd.to_crs(4326)
    icon_gdf["_lat"] = icon_gdf.geometry.y
    icon_gdf["_lon"] = icon_gdf.geometry.x

    gdf = gdf_rd.to_crs(4326)
    # Icon + crop name in one field so the tooltip/popup shows "🌽 Mais,
    # snij-" without needing a second aliased row just for the icon.
    gdf["_crop_display"] = gdf["_icon"] + " " + gdf["gewas"]
    _no_data_txt = ("no observed transitions", "geen waargenomen overgangen")[i]
    gdf["_pred_display"] = gdf.apply(
        lambda r: (
            f"{CROP_FAMILIES[r['_pred_family']][1]} {CROP_FAMILIES[r['_pred_family']][2 + i]}"
            + (f" ({r['_pred_prob'] * 100:.0f}%)" if pd.notna(r["_pred_prob"]) else f" ({_no_data_txt})")
        ),
        axis=1,
    )

    def color_for(row):
        if color_by == "category":
            return BRP_COLORS.get(row["category"], "#999999")
        if color_by == "crop_family":
            return CROP_FAMILIES[row["_family"]][0]
        if color_by == "predicted_next_family":
            return CROP_FAMILIES[row["_pred_family"]][0]
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
    gdf = gdf[["geometry", "_crop_display", "category", "_area_txt", "_ndvi25_txt", "_ndvi_chg_txt",
               "_pred_display", "_color"]]

    m = folium.Map(location=center, zoom_start=14 if village else 13, tiles="OpenStreetMap",
                    control_scale=True, prefer_canvas=True)
    Fullscreen(position="topleft",
               title="Fullscreen" if lang == "en" else "Volledig scherm",
               title_cancel="Exit fullscreen" if lang == "en" else "Volledig scherm sluiten").add_to(m)

    tooltip_aliases = [
        ("Crop", "Gewas"), ("Category", "Categorie"), ("Area (ha)", "Oppervlakte (ha)"),
        ("NDVI 2025", "NDVI 2025"), ("NDVI Δ 2024→25", "NDVI Δ 2024→25"),
        ("Predicted next crop", "Voorspeld volgend gewas"),
    ][:]
    aliases = [pair[i] for pair in tooltip_aliases]
    _feature_fields = ["_crop_display", "category", "_area_txt", "_ndvi25_txt", "_ndvi_chg_txt", "_pred_display"]

    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda f: {
            "fillColor": f["properties"]["_color"], "color": "#33332a",
            "weight": 0.4, "fillOpacity": 0.75,
        },
        highlight_function=lambda f: {"weight": 2, "color": "#16221C", "fillOpacity": 0.9},
        tooltip=folium.GeoJsonTooltip(fields=_feature_fields, aliases=aliases, sticky=True),
        popup=folium.GeoJsonPopup(fields=_feature_fields, aliases=aliases),
        name=("Field boundaries (BRP)", "Perceelgrenzen (BRP)")[i],
    ).add_to(m)

    if color_by == "crop_family":
        # Clustered so 1,500+ individual markers don't turn into an
        # unreadable pile at municipality-wide zoom -- Leaflet groups them
        # into a numbered bubble below zoom 15 and expands to real icons
        # above it, the standard pattern for this many point markers.
        cluster = MarkerCluster(
            disable_clustering_at_zoom=15, max_cluster_radius=38,
            name="Crop icons" if lang == "en" else "Gewas-iconen",
        ).add_to(m)
        for _, r in icon_gdf.iterrows():
            fam_color = CROP_FAMILIES[r["_family"]][0]
            folium.Marker(
                location=[r["_lat"], r["_lon"]],
                # A colour badge (the field's own family colour, so the
                # pin and the fill it sits on always agree) with a white
                # line-icon inside -- the professional-icon-set
                # replacement for a bare emoji + white halo.
                icon=folium.DivIcon(html=(
                    f'<div style="width:24px;height:24px;border-radius:50%;background:{fam_color};'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'box-shadow:0 1px 3px rgba(0,0,0,.45),0 0 0 1.5px #fff;">'
                    f'{crop_icon_svg(r["_family"], size=15, stroke="#fff")}</div>'
                ), icon_anchor=(12, 12)),
                tooltip=f'{r["gewas"]} · {r["area_ha"]:.1f} ha',
            ).add_to(cluster)

        rows = "".join(
            f'<div style="display:flex;align-items:center;gap:7px;margin:3px 0;">'
            f'<span style="width:20px;height:20px;border-radius:50%;background:{color};flex-shrink:0;'
            f'display:flex;align-items:center;justify-content:center;">{crop_icon_svg(fam, size=12, stroke="#fff")}</span>'
            f'<span>{label[i]}</span></div>'
            for fam, (color, _icon, *label) in CROP_FAMILIES.items()
        )
        legend_title = "Crop family" if lang == "en" else "Gewasfamilie"
        m.get_root().html.add_child(folium.Element(f"""
        <div style="position: fixed; bottom: 24px; right: 24px; z-index: 9999;
                    background: white; padding: 10px 14px; border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,.2); font-family: sans-serif; font-size: 11.5px;
                    max-width: 250px;">
          <div style="font-weight:600; margin-bottom:4px; font-size:12px;">{legend_title}</div>
          {rows}
        </div>
        """))
    elif color_by == "category":
        # Every color_by mode now carries a real legend -- this one used to
        # be the one silent exception (crop_family had one, the field's
        # default coloring didn't).
        rows = "".join(
            f'<div style="display:flex;align-items:center;gap:7px;margin:3px 0;">'
            f'<span style="width:12px;height:12px;background:{color};display:inline-block;'
            f'border-radius:2px;flex-shrink:0;"></span><span>{cat}</span></div>'
            for cat, color in BRP_COLORS.items()
        )
        legend_title = ("Category (BRP)", "Categorie (BRP)")[i]
        m.get_root().html.add_child(folium.Element(f"""
        <div style="position: fixed; bottom: 24px; right: 24px; z-index: 9999;
                    background: white; padding: 10px 14px; border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,.2); font-family: sans-serif; font-size: 11.5px;
                    max-width: 220px;">
          <div style="font-weight:600; margin-bottom:4px; font-size:12px;">{legend_title}</div>
          {rows}
        </div>
        """))
    elif color_by in ("ndvi_2025", "ndvi_change"):
        # A sequential/diverging gradient swatch with labeled endpoints --
        # the same vmin/vmax/colormap color_for() itself uses, so the
        # legend and the fill are guaranteed to agree.
        if color_by == "ndvi_2025":
            vmin, vmax, cmap_name = 0.2, 0.9, "RdYlGn"
            legend_title = ("NDVI 2025 (health)", "NDVI 2025 (vitaliteit)")[i]
            lo_txt, hi_txt = (("0.2 · bare / stressed", "0.9 · dense / healthy") if lang == "en"
                              else ("0,2 · kaal / gestrest", "0,9 · dicht / vitaal"))
        else:
            vmin, vmax, cmap_name = -0.15, 0.15, "BrBG"
            legend_title = ("NDVI change 2024→2025", "NDVI-verandering 2024→2025")[i]
            lo_txt, hi_txt = (("-0.15 · decline", "+0.15 · growth") if lang == "en"
                              else ("-0,15 · afname", "+0,15 · toename"))
        stops = [_hex_from_cmap(vmin + f * (vmax - vmin), vmin, vmax, cmap_name) for f in (0, .25, .5, .75, 1)]
        gradient_css = ",".join(stops)
        m.get_root().html.add_child(folium.Element(f"""
        <div style="position: fixed; bottom: 24px; right: 24px; z-index: 9999;
                    background: white; padding: 10px 14px; border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,.2); font-family: sans-serif; font-size: 11.5px;
                    max-width: 220px;">
          <div style="font-weight:600; margin-bottom:6px; font-size:12px;">{legend_title}</div>
          <div style="height:12px; border-radius:3px; background: linear-gradient(to right, {gradient_css});"></div>
          <div style="display:flex; justify-content:space-between; margin-top:4px; color:#555; font-size:10.5px;">
            <span>{lo_txt}</span><span>{hi_txt}</span>
          </div>
        </div>
        """))
    elif color_by == "predicted_next_family":
        # Same swatch rows as crop_family (same palette, same meaning of
        # colour) -- the caveat is what's different: this is a predicted
        # *next* family, with a real sample size behind it, not the
        # observed current one.
        rows = "".join(
            f'<div style="display:flex;align-items:center;gap:7px;margin:3px 0;">'
            f'<span style="width:20px;height:20px;border-radius:50%;background:{color};flex-shrink:0;'
            f'display:flex;align-items:center;justify-content:center;">{crop_icon_svg(fam, size=12, stroke="#fff")}</span>'
            f'<span>{label[i]}</span></div>'
            for fam, (color, _icon, *label) in CROP_FAMILIES.items()
        )
        legend_title = ("Predicted next crop family (ML)", "Voorspeld volgend gewasfamilie (ML)")[i]
        _n_obs = load_stats().get("forecast", {}).get("crop_rotation", {}).get("n_transitions_observed", 0)
        legend_note = (
            f"Each field's most likely next-year family, from a transition matrix built on {_n_obs:,} "
            "real year-to-year transitions in this pipeline's own matched BRP history — a probability, "
            "not a guarantee. Hover a field for its own predicted probability.",
            f"De meest waarschijnlijke gewasfamilie van volgend jaar per perceel, uit een overgangsmatrix "
            f"op basis van {_n_obs:,} echte jaar-op-jaar overgangen in de eigen gematchte BRP-geschiedenis "
            "van deze pipeline — een kans, geen garantie. Beweeg over een perceel voor de eigen "
            "voorspelde kans.",
        )[i]
        m.get_root().html.add_child(folium.Element(f"""
        <div style="position: fixed; bottom: 24px; right: 24px; z-index: 9999;
                    background: white; padding: 10px 14px; border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,.2); font-family: sans-serif; font-size: 11.5px;
                    max-width: 260px;">
          <div style="font-weight:600; margin-bottom:4px; font-size:12px;">{legend_title}</div>
          {rows}
          <div style="margin-top:6px; color:#666; font-size:10px; line-height:1.4;">{legend_note}</div>
        </div>
        """))

    if village_geom is not None:
        folium.GeoJson(
            village_geom.__geo_interface__, name=f"📍 {village}",
            style_function=lambda x: {"fillOpacity": 0, "color": "#4a3aa7", "weight": 3},
        ).add_to(m)

    folium.GeoJson(
        geometry_wgs84().__geo_interface__, name="Municipal boundary",
        style_function=lambda x: {"fillOpacity": 0, "color": "#16221C", "weight": 2},
    ).add_to(m)

    # Same QGIS-print-composer-style title/scope/date/source strip and
    # north arrow as the Overview map (see _report_header_element /
    # _north_arrow_element) -- Field Explorer is printable too.
    m.get_root().html.add_child(_report_header_element(lang, village))
    m.get_root().html.add_child(_north_arrow_element())

    folium.LayerControl(collapsed=False).add_to(m)
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


def make_map(label_new: str = "summer_2025", label_old: str = "summer_2024", lang: str = "en",
             village: str | None = None) -> folium.Map:
    """Builds the folium.Map object without saving it -- used directly by
    build_map() below and embedded live in dashboard.py. `lang` ("en"/"nl")
    switches every layer name and legend on the map itself, not just the
    Streamlit chrome around it. `village`, a name from
    data/raw/villages.geojson, zooms to that place, clips every raster
    layer to its real administrative boundary (see `_clip_to_village` --
    alpha zeroed outside the polygon, not just a zoom + outline over the
    whole-municipality imagery), and draws the boundary itself as its own
    outlined layer -- the same village the sidebar's selector already
    filters Field Explorer to."""
    i = 0 if lang == "en" else 1
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

    # "Where did the forest actually go" -- a real spatial pixel-change
    # map, not just the aggregate hectare trend on the Trends & Climate/
    # Forecast tabs. Restricted to whatever Sentinel-2-era window
    # landcover_trend.build_change_map() actually built (2018-present in
    # practice) -- see that function's own docstring for why this can't
    # extend back to the 2005-era Landsat years without reprojecting to a
    # common grid first, a real resolution limit, not an oversight.
    lc_change_stats = load_stats().get("landcover_change_map", {})
    lcc_first, lcc_last = lc_change_stats.get("first_year"), lc_change_stats.get("last_year")
    lcc_path = lcc_bounds = None
    if lcc_first and lcc_last:
        lcc_path, lcc_bounds = landcover_change_png(lcc_first, lcc_last)

    village_geom_wgs84, village_bounds = _load_village(village)

    # "Snap and cut": once a village is picked, every raster layer above
    # gets clipped to its real boundary (alpha zeroed outside it) instead
    # of the whole municipality staying visible under a zoom + outline --
    # picking a place now visibly cuts the map to it, not just centers on
    # it. Vector layers (BRP field boundaries within Field Explorer,
    # the municipal/village outlines below) are already place-accurate by
    # construction and don't need this.
    if village_geom_wgs84 is not None:
        vslug = _slug(village)
        tc_path = _clip_to_village(tc_path, bounds, village_geom_wgs84, vslug)
        ndvi_path = _clip_to_village(ndvi_path, bounds, village_geom_wgs84, vslug)
        lc_path = _clip_to_village(lc_path, bounds, village_geom_wgs84, vslug)
        chg_path = _clip_to_village(chg_path, bounds, village_geom_wgs84, vslug)
        vv_path = _clip_to_village(vv_path, vv_bounds, village_geom_wgs84, vslug)
        water_path = _clip_to_village(water_path, vv_bounds, village_geom_wgs84, vslug)
        flood_path = _clip_to_village(flood_path, flood_bounds, village_geom_wgs84, vslug)
        if flood_cd_path is not None:
            flood_cd_path = _clip_to_village(flood_cd_path, flood_cd_bounds, village_geom_wgs84, vslug)
        elev_path = _clip_to_village(elev_path, elev_bounds, village_geom_wgs84, vslug)
        ndsm_path = _clip_to_village(ndsm_path, elev_bounds, village_geom_wgs84, vslug)
        no2_path = _clip_to_village(no2_path, no2_bounds, village_geom_wgs84, vslug)
        brp_path = _clip_to_village(brp_path, brp_bounds, village_geom_wgs84, vslug)
        if lcc_path is not None:
            lcc_path = _clip_to_village(lcc_path, lcc_bounds, village_geom_wgs84, vslug)

    if village_bounds:
        center = [(village_bounds[0][0] + village_bounds[1][0]) / 2, (village_bounds[0][1] + village_bounds[1][1]) / 2]
        zoom_start = 14
    else:
        center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
        zoom_start = 12
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap", control_scale=True)
    Fullscreen(position="topleft",
               title="Fullscreen" if lang == "en" else "Volledig scherm",
               title_cancel="Exit fullscreen" if lang == "en" else "Volledig scherm sluiten").add_to(m)
    if village_bounds:
        m.fit_bounds(village_bounds)

    layer_names = [
        ("True colour (Aug 2025)", "Ware kleur (aug. 2025)"),
        ("NDVI (Aug 2025)", "NDVI (aug. 2025)"),
        ("Land cover (KMeans, Aug 2025)", "Landgebruik (KMeans, aug. 2025)"),
        ("NDVI change, Aug 2024 → 2025", "NDVI-verandering, aug. 2024 → 2025"),
        ("SAR backscatter, VV (Aug 2025)", "SAR-terugkaatsing, VV (aug. 2025)"),
        ("SAR water mask (Aug 2025)", "SAR-watermasker (aug. 2025)"),
        ("Flood extent (old method) — fixed threshold, Jan 2024 vs Aug 2025",
         "Overstromingsgebied (oude methode) — vaste drempel, jan. 2024 vs aug. 2025"),
        (f"Flood extent (change detection) — peak {peak_event}",
         f"Overstromingsgebied (verandering-detectie) — piek {peak_event}"),
        ("Elevation, AHN DTM (m NAP)", "Hoogte, AHN DTM (m NAP)"),
        ("Canopy / building height, AHN nDSM", "Bladerdak-/gebouwhoogte, AHN nDSM"),
        ("Air quality — NO2 (RIVM, 2024)", "Luchtkwaliteit — NO2 (RIVM, 2024)"),
        ("Field boundaries & crops (BRP)", "Perceelgrenzen & gewassen (BRP)"),
        ("Municipal boundary", "Gemeentegrens"),
    ]
    names = [pair[i] for pair in layer_names]

    l_tc = folium.raster_layers.ImageOverlay(str(tc_path), bounds=bounds, name=names[0], opacity=1.0).add_to(m)
    l_ndvi = folium.raster_layers.ImageOverlay(str(ndvi_path), bounds=bounds, name=names[1], opacity=0.85, show=False).add_to(m)
    l_lc = folium.raster_layers.ImageOverlay(str(lc_path), bounds=bounds, name=names[2], opacity=0.8, show=False).add_to(m)
    l_chg = folium.raster_layers.ImageOverlay(str(chg_path), bounds=bounds, name=names[3], opacity=0.85, show=False).add_to(m)
    l_vv = folium.raster_layers.ImageOverlay(str(vv_path), bounds=vv_bounds, name=names[4], opacity=0.9, show=False).add_to(m)
    l_water = folium.raster_layers.ImageOverlay(str(water_path), bounds=vv_bounds, name=names[5], opacity=0.75, show=False).add_to(m)
    l_flood = folium.raster_layers.ImageOverlay(str(flood_path), bounds=flood_bounds, name=names[6], opacity=0.85, show=False).add_to(m)
    l_flood_cd = None
    if flood_cd_path is not None:
        l_flood_cd = folium.raster_layers.ImageOverlay(
            str(flood_cd_path), bounds=flood_cd_bounds, name=names[7], opacity=0.85, show=False
        ).add_to(m)
    l_elev = folium.raster_layers.ImageOverlay(str(elev_path), bounds=elev_bounds, name=names[8], opacity=0.85, show=False).add_to(m)
    l_ndsm = folium.raster_layers.ImageOverlay(str(ndsm_path), bounds=elev_bounds, name=names[9], opacity=0.85, show=False).add_to(m)
    l_no2 = folium.raster_layers.ImageOverlay(str(no2_path), bounds=no2_bounds, name=names[10], opacity=0.75, show=False).add_to(m)
    l_brp = folium.raster_layers.ImageOverlay(str(brp_path), bounds=brp_bounds, name=names[11], opacity=0.8, show=False).add_to(m)
    l_lcc = None
    if lcc_path is not None:
        lcc_name = (f"Forest & land cover change, {lcc_first}→{lcc_last}",
                    f"Bos & landgebruikverandering, {lcc_first}→{lcc_last}")[i]
        l_lcc = folium.raster_layers.ImageOverlay(
            str(lcc_path), bounds=lcc_bounds, name=lcc_name, opacity=0.9, show=False
        ).add_to(m)

    l_muni = folium.GeoJson(
        geometry_wgs84().__geo_interface__,
        name=names[12],
        style_function=lambda x: {"fillOpacity": 0, "color": "#16221C", "weight": 2},
    ).add_to(m)

    l_village = None
    if village_geom_wgs84 is not None:
        village_layer_name = f"📍 {village}" if lang == "en" else f"📍 {village}"
        l_village = folium.GeoJson(
            village_geom_wgs84.__geo_interface__,
            name=village_layer_name,
            style_function=lambda x: {"fillColor": "#4a3aa7", "fillOpacity": 0.08, "color": "#4a3aa7", "weight": 3},
        ).add_to(m)

    # Grouped, collapsible layer control (leaflet-groupedlayercontrol) --
    # a flat 14-15-checkbox list ("all of NDVI/land cover/SAR/forest
    # change/field boundaries on screen at once") is genuinely hard to
    # scan; grouping by what kind of data each layer actually is (optical,
    # land cover, radar, elevation, environment, registry) is the standard
    # GIS-software fix, not a cosmetic one. exclusive_groups=False keeps
    # every group a checkbox list (multiple layers within or across
    # groups can still be shown together), matching the plain
    # LayerControl's existing behaviour exactly -- only the layout changes.
    group_titles = {
        "optical": ("Optical & vegetation", "Optisch & vegetatie"),
        "landcover": ("Land cover & change", "Landgebruik & verandering"),
        "sar": ("Radar (SAR)", "Radar (SAR)"),
        "elevation": ("Elevation", "Hoogte"),
        "environment": ("Environment", "Milieu"),
        "registry": ("Registry & boundaries", "Registratie & grenzen"),
    }
    groups = {
        group_titles["optical"][i]: [l_tc, l_ndvi, l_chg],
        group_titles["landcover"][i]: [l_lc] + ([l_lcc] if l_lcc is not None else []),
        group_titles["sar"][i]: [l_vv, l_water, l_flood] + ([l_flood_cd] if l_flood_cd is not None else []),
        group_titles["elevation"][i]: [l_elev, l_ndsm],
        group_titles["environment"][i]: [l_no2],
        group_titles["registry"][i]: [l_brp, l_muni] + ([l_village] if l_village is not None else []),
    }
    GroupedLayerControl(groups=groups, exclusive_groups=False, collapsed=False).add_to(m)

    legend_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
        f'<span style="width:16px;height:16px;border-radius:50%;background:{LANDCOVER_COLORS.get(name,"#999")};'
        f'display:flex;align-items:center;justify-content:center;flex:none;">'
        f'{landcover_icon_svg(name, size=10, stroke="#fff")}</span>'
        f'<span>{landcover_label(name, "en" if i == 0 else "nl")}</span></div>'
        for name in sorted(set(id_to_name.values()))
    )
    legend_title = ("Land cover (KMeans)", "Landgebruik (KMeans)")[i]
    _method_ref_lc = ("ℹ️ Method: unsupervised KMeans, 6 clusters — see \"Land cover (KMeans)\" on the "
                       "Methodology section.",
                       "ℹ️ Methode: ongestuurde KMeans, 6 clusters — zie \"Landgebruik (KMeans)\" op het "
                       "onderdeel Methodologie.")[i]
    land_cover_legend_block = f"""
    <details class="bd-legend" open>
      <summary>{legend_title}</summary>
      <div class="bd-legend-body">{legend_rows}
        <div class="bd-legend-method">{_method_ref_lc}</div>
      </div>
    </details>
    """
    # Shared style for every <details class="bd-legend"> block on this map
    # (land cover, BRP, forest-change, flood -- built at various points
    # below) -- added once, here, since land_cover_legend_block above is
    # always built regardless of what else is on the map.
    m.get_root().html.add_child(folium.Element("""
    <style>
      details.bd-legend { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,.2);
        font-family: sans-serif; font-size: 12px; }
      details.bd-legend summary { padding: 10px 14px; font-weight: 600; cursor: pointer; list-style: none; }
      details.bd-legend summary::-webkit-details-marker { display: none; }
      details.bd-legend summary::after { content: '▾'; float: right; margin-left: 14px; color: #999; font-weight: 400; }
      details.bd-legend[open] summary::after { content: '▴'; }
      details.bd-legend .bd-legend-body { padding: 0 14px 12px; }
      .bd-legend-method { margin-top: 6px; padding-top: 6px; border-top: 1px dashed #ddd;
        color: #7a8177; font-size: 10px; line-height: 1.4; }
      details.bd-legend {
        background: rgba(255,255,255,.96); border: 1px solid #c7d0c4; border-radius: 6px;
        box-shadow: 0 10px 26px rgba(16,40,29,.18); color: #172019;
        font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif; font-size: 12px;
        backdrop-filter: blur(2px);
      }
      details.bd-legend summary {
        padding: 10px 12px; font-weight: 700; border-bottom: 1px solid #e4e8df; color:#10281d;
      }
      details.bd-legend summary::after { content: '+'; color: #2d6f86; font-weight: 700; }
      details.bd-legend[open] summary::after { content: '-'; }
      details.bd-legend .bd-legend-body { padding: 8px 12px 12px; }
      .bd-legend-method {
        margin-top: 8px; padding-top: 7px; border-top: 1px solid #e4e8df;
        color: #526157; font-size: 10.5px; line-height: 1.35;
      }
      @media print {
        .leaflet-control-layers { display:none !important; }
        details.bd-legend { box-shadow:none; border:1px solid #999; }
      }
    </style>
    """))

    flood_stats = load_stats().get("flood_event", {})
    net_change_txt = f"{flood_stats['net_change_pct']:+.1f}" if "net_change_pct" in flood_stats else "?"
    flood_legend_text = [
        {
            "title": "Flood extent (SAR) — two methods",
            "permanent": "Permanent water (old method, both dates)",
            "newly": "Newly flooded (either method)",
            "missed": "Old method: baseline water missed (threshold noise)",
            "note": (f"Old method (one fixed -17dB cutoff, one post-peak date): ~0 net change.<br/>"
                     f"New method (per-pixel change detection vs. a 4-date normal-condition composite, "
                     f"{len(flood_stats.get('timeline', []))} dates through the event): "
                     f"{flood_stats.get('pre_event_flooded_pct', '?')}% &rarr; "
                     f"{flood_stats.get('peak_flooded_pct', '?')}% at peak ({net_change_txt} pts net) — "
                     f"see README/Water section."),
            "method_ref": "ℹ️ Method: see \"Flood extent — fixed threshold (old method)\" and "
                          "\"Flood extent — change detection (new method)\" in the Methodology section.",
        },
        {
            "title": "Overstromingsgebied (SAR) — twee methodes",
            "permanent": "Permanent water (oude methode, beide data)",
            "newly": "Nieuw overstroomd (beide methodes)",
            "missed": "Oude methode: basiswater gemist (drempelruis)",
            "note": (f"Oude methode (één vaste -17dB-drempel, één datum na de piek): ~0 netto verandering.<br/>"
                     f"Nieuwe methode (verandering-detectie per pixel t.o.v. een 4-datums normaalcomposiet, "
                     f"{len(flood_stats.get('timeline', []))} data door de gebeurtenis): "
                     f"{flood_stats.get('pre_event_flooded_pct', '?')}% &rarr; "
                     f"{flood_stats.get('peak_flooded_pct', '?')}% op de piek ({net_change_txt} pt netto) — "
                     f"zie README/onderdeel Water."),
            "method_ref": "ℹ️ Methode: zie \"Overstromingsgebied — vaste drempel (oude methode)\" en "
                          "\"Overstromingsgebied — verandering-detectie (nieuwe methode)\" op het "
                          "onderdeel Methodologie.",
        },
    ][i]
    flood_legend_block = f"""
    <details class="bd-legend" style="max-width: 240px;">
      <summary>{flood_legend_text['title']}</summary>
      <div class="bd-legend-body">
        <div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
          <span style="width:12px;height:12px;background:#3B6E8A;display:inline-block;border-radius:2px;"></span>
          {flood_legend_text['permanent']}</div>
        <div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
          <span style="width:12px;height:12px;background:#C03B2E;display:inline-block;border-radius:2px;"></span>
          {flood_legend_text['newly']}</div>
        <div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
          <span style="width:12px;height:12px;background:#C9B47A;display:inline-block;border-radius:2px;"></span>
          {flood_legend_text['missed']}</div>
        <div style="margin-top:6px; color:#666; font-size:10.5px;">{flood_legend_text['note']}</div>
        <div class="bd-legend-method">{flood_legend_text['method_ref']}</div>
      </div>
    </details>
    """
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position: fixed; bottom: 24px; right: 24px; z-index: 9999;">{flood_legend_block}</div>
    """))

    brp_legend_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
        f'<span style="width:12px;height:12px;background:{color};display:inline-block;border-radius:2px;"></span>{name}</div>'
        for name, color in brp_colors.items()
    )
    brp_legend_title = ("Field boundaries (BRP)", "Perceelgrenzen (BRP)")[i]
    brp_legend_note = ("Per-crop breakdown is on the dashboard.", "Uitsplitsing per gewas staat op het dashboard.")[i]
    _method_ref_brp = ("Info: see \"Registered farmland (BRP)\" in the Methodology section.",
                        "Info: zie \"Landbouwgrond (BRP)\" in het onderdeel Methodologie.")[i]
    brp_legend_block = f"""
    <details class="bd-legend">
      <summary>{brp_legend_title}</summary>
      <div class="bd-legend-body">
        {brp_legend_rows}
        <div style="margin-top:4px; color:#666; font-size:10.5px;">{brp_legend_note}</div>
        <div class="bd-legend-method">{_method_ref_brp}</div>
      </div>
    </details>
    """

    lcc_legend_block = ""
    if lcc_path is not None:
        _lcc_ha = lc_change_stats.get("ha", {})
        _lcc_rows_spec = [
            (2, ("Forest loss", "Bosverlies")),
            (3, ("Forest gain", "Bosaanwas")),
            (4, ("Built-up growth", "Bebouwingsgroei")),
            (5, ("Other change", "Overige verandering")),
        ]
        lcc_legend_rows = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
            f'<span style="width:12px;height:12px;background:rgb{LANDCOVER_CHANGE_COLORS[cid][:3]};'
            f'display:inline-block;border-radius:2px;"></span>{label[i]}'
            f'<span style="color:#888; margin-left:auto; padding-left:8px;">'
            f'{_lcc_ha.get(label[0], 0):,.0f} ha</span></div>'  # `ha` dict keys are always the English name
            for cid, label in _lcc_rows_spec
        )
        lcc_legend_title = (f"Forest & land cover change, {lcc_first}→{lcc_last}",
                             f"Bos & landgebruikverandering, {lcc_first}→{lcc_last}")[i]
        lcc_legend_note = (
            "Forest loss/gain are the more spectrally stable, more trustworthy categories here (same "
            "caveat as the hectare trend elsewhere in this dashboard); built-up growth and other change "
            "inherit that series' real classifier year-to-year wobble. Unchanged ground (the large "
            "majority) is left transparent on purpose, so only real change draws the eye.",
            "Bosverlies/-aanwas zijn hier de spectraal stabielere, betrouwbaardere categorieën (dezelfde "
            "kanttekening als bij de hectaretrend elders in dit dashboard); bebouwingsgroei en overige "
            "verandering erven de echte jaar-op-jaar classifier-wiebel van die reeks. Ongewijzigde grond "
            "(de grote meerderheid) is bewust transparant gelaten, zodat alleen echte verandering opvalt.",
        )[i]
        _method_ref_lcc = ("ℹ️ Method: same KMeans classifier as \"Land cover (KMeans)\" on the "
                           "Methodology section, compared pixel-for-pixel across years.",
                           "ℹ️ Methode: dezelfde KMeans-classifier als \"Landgebruik (KMeans)\" op het "
                           "onderdeel Methodologie, pixel-voor-pixel vergeleken over de jaren.")[i]
        lcc_legend_block = f"""
        <details class="bd-legend" style="max-width: 260px;">
          <summary>{lcc_legend_title}</summary>
          <div class="bd-legend-body">
            {lcc_legend_rows}
            <div style="margin-top:6px; color:#666; font-size:10px; line-height:1.4;">{lcc_legend_note}</div>
            <div class="bd-legend-method">{_method_ref_lcc}</div>
          </div>
        </details>
        """

    # One collapsible stack, bottom-left, instead of four separate boxes
    # scattered at hand-picked pixel offsets -- "a real map report" is
    # also "not a wall of always-open text boxes." Land cover opens by
    # default (the most commonly toggled-on layer alongside true colour);
    # the rest are one click away rather than always taking up space.
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; display: flex;
                flex-direction: column; gap: 8px; max-width: 260px;">
      {land_cover_legend_block}
      {brp_legend_block}
      {lcc_legend_block}
    </div>
    """))

    # Print/report block: a QGIS-print-composer-style title/scope/date/
    # source strip and a north arrow (see _report_header_element /
    # _north_arrow_element), baked into the map's own HTML so they're
    # part of what actually prints -- the map lives in an
    # st.components.v1 iframe, and anything drawn only in the surrounding
    # Streamlit page isn't guaranteed to appear in the printed output.
    # Together with the always-on legends above, the scale bar
    # (`control_scale=True` on the folium.Map itself) and this block, the
    # printed page is a real map report -- title, scope, date, sources,
    # north arrow, scale, legend -- not just a screenshot of the widget.
    m.get_root().html.add_child(_report_header_element(lang, village))
    m.get_root().html.add_child(_north_arrow_element())

    # GroupedLayerControl (added earlier, right after the layers
    # themselves) replaces the plain LayerControl here -- see its own
    # comment above for why.
    return m


def build_map(label_new: str = "summer_2025", label_old: str = "summer_2024", lang: str = "en") -> Path:
    m = make_map(label_new, label_old, lang=lang)
    out_path = OUT_DIR / "bergendal_map.html"
    m.save(str(out_path))
    print(f"saved interactive map: {out_path}")
    return out_path


if __name__ == "__main__":
    build_map()
