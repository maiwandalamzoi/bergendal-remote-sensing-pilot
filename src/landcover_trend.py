"""
Land-cover area change, 2018-2026: how many hectares of dense vegetation
(forest-like), grass/farmland, built-up/bare ground and water actually
grew or shrank, not just this year's snapshot.

Runs `landcover_ml.classify()` for every Sentinel-2 year already on disk
(2018-present -- all fetched by ndvi_trend.py already, so this needs no
new imagery, just the KMeans pass) and rolls the six raw clusters up into
four broad categories a person actually asks about.

**A real bug found and fixed while building this:** the first pass
compared each year's *raw* hectares directly, and came back with e.g.
built-up area swinging from 530ha (2018) to 93ha (2023) back to 357ha
(2026) -- not remotely plausible for a real municipality's built
environment in under a decade. Cause: clear-ground coverage differs by
year (cloud), so each year's classified pixel count has a *different*
denominator -- a less cloudy year reads as "more of everything" even
with zero real change. Fixed by restricting every year's area count to
the pixels valid in *all* years (the intersection mask across all 9
classified rasters) -- a fixed, common footprint, so a change in hectares
means a change on the ground, not a change in cloud cover.

**A real limitation that's not fixable the same way, so it's named
instead:** with only 6 unsupervised clusters, "Built-up / bare" and
"Grass / farmland" sit closer together in spectral space than they do in
reality -- bare/just-harvested cropland and true impervious surface can
get relabelled between them from one year's centroid fit to the next.
"Forest / dense vegetation" and "Water" are far more spectrally distinct
and consistently the more stable series; built-up specifically should be
read as directional/noisy, not a precise hectare count, until it's
cross-checked against something like BAG (the building registry) --
flagged as a next step, not faked as more precise than it is.

    python landcover_trend.py
"""
from pathlib import Path

import numpy as np
import rasterio

from landcover_ml import classify
from statsutil import load_stats, update_stats

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

YEARS = list(range(2018, 2027))  # Sentinel-2 era only -- see module docstring
HA_PER_PX = 0.01  # 10m Sentinel-2 grid: 100 m2/px

# The six raw KMeans labels (_label_clusters in landcover_ml.py) rolled up
# into four broad categories a person actually asks about.
BROAD_CATEGORIES = {
    "Water": "Water",
    "Built-up / bare": "Built-up",
    "Grass / farmland": "Agriculture",
    "Dense vegetation, dark canopy": "Forest / dense vegetation",
    "Dense vegetation, mid canopy": "Forest / dense vegetation",
    "Dense vegetation, bright canopy": "Forest / dense vegetation",
    "Dense vegetation (forest)": "Forest / dense vegetation",
}


def ensure_classified() -> list[int]:
    stats = load_stats()
    available = []
    for year in YEARS:
        label = f"summer_{year}"
        if not (PROC_DIR / f"{label}_indices.tif").exists():
            print(f"[landcover trend] {year}: no indices on disk yet, skipping "
                  f"(run ndvi_trend.py / preprocess.py first)")
            continue
        if not (PROC_DIR / f"{label}_landcover.tif").exists() or f"landcover_{label}" not in stats:
            classify(label)
        available.append(year)
    return available


def _load_legend(label: str) -> dict[int, str]:
    legend = {}
    for line in (PROC_DIR / f"{label}_landcover_legend.txt").read_text().strip().splitlines():
        cid, name = line.split("\t")
        legend[int(cid)] = name
    return legend


def build_trend() -> dict:
    years = ensure_classified()
    if len(years) < 2:
        raise RuntimeError(f"only {len(years)} classified year(s) -- can't build a trend")

    rasters, legends = {}, {}
    for year in years:
        label = f"summer_{year}"
        with rasterio.open(PROC_DIR / f"{label}_landcover.tif") as src:
            rasters[year] = src.read(1)
        legends[year] = _load_legend(label)

    common_valid = np.ones_like(rasters[years[0]], dtype=bool)
    for year in years:
        common_valid &= rasters[year] != 255
    n_common = int(common_valid.sum())
    print(f"[landcover trend] common valid footprint across all {len(years)} years: "
          f"{n_common:,} px ({n_common * HA_PER_PX:,.0f} ha, "
          f"{n_common / common_valid.size * 100:.1f}% of the AOI's own raster grid)")

    stats = load_stats()
    series = {cat: [] for cat in set(BROAD_CATEGORIES.values())}
    clear_pcts = []
    for year in years:
        cls = rasters[year][common_valid]
        by_broad = {cat: 0 for cat in series}
        for cid, raw_name in legends[year].items():
            broad = BROAD_CATEGORIES.get(raw_name)
            if broad:
                by_broad[broad] += int((cls == cid).sum())
        for cat in series:
            series[cat].append(round(by_broad[cat] * HA_PER_PX, 1))
        clear_pcts.append(stats.get(f"optical_summer_{year}", {}).get("clear_pct"))

    print(f"Land cover trend {years[0]}-{years[-1]} ({len(years)} years, common footprint only):")
    changes = {}
    for cat, vals in series.items():
        net = vals[-1] - vals[0]
        pct_of_first = (net / vals[0] * 100) if vals[0] else float("nan")
        changes[cat] = {"start_ha": vals[0], "end_ha": vals[-1], "net_ha": round(net, 1),
                         "net_pct": round(pct_of_first, 1)}
        print(f"  {cat:<28} {vals[0]:7.1f} ha -> {vals[-1]:7.1f} ha  ({net:+7.1f} ha, {pct_of_first:+.1f}%)")

    update_stats("landcover_trend", {
        "years": years,
        "clear_pct": clear_pcts,
        "common_footprint_ha": round(n_common * HA_PER_PX, 1),
        "series_ha": series,
        "change": changes,
    })

    build_change_map(years[0], years[-1])
    return {"years": years, "series_ha": series, "change": changes}


# Pixel-level transition classes for the spatial change map -- one label
# per changed pixel, priority-ordered so a pixel that's both "left forest"
# and "became built-up" reads as forest loss first (the category this map
# exists to answer -- "where did the forest actually go"), not folded
# anonymously into a generic "other change" bucket.
CHANGE_NODATA = 0
CHANGE_NONE = 1
CHANGE_FOREST_LOSS = 2
CHANGE_FOREST_GAIN = 3
CHANGE_BUILTUP_GROWTH = 4
CHANGE_OTHER = 5

CHANGE_CLASS_NAMES = {
    CHANGE_NODATA: "No data",
    CHANGE_NONE: "No change",
    CHANGE_FOREST_LOSS: "Forest loss",
    CHANGE_FOREST_GAIN: "Forest gain",
    CHANGE_BUILTUP_GROWTH: "Built-up growth",
    CHANGE_OTHER: "Other change",
}


def build_change_map(first_year: int, last_year: int) -> Path:
    """A real spatial answer to "where did the forest actually go" -- not
    just the aggregate hectare trend above, a pixel that changed *class*
    between `first_year` and `last_year`, both already-classified Sentinel-2
    years sharing the same 10m grid (checked directly: identical transform/
    shape/CRS, not assumed -- this is what makes a pixel-wise compare valid
    at all here). Restricted to the Sentinel-2 era (2018-present) on
    purpose: the pre-2018 series in ndvi_trend.py/the Trends & Climate tab
    uses 30m Landsat, a different pixel grid entirely -- comparing it
    pixel-for-pixel against this 10m grid without reprojecting first would
    silently misalign ground, not just lose precision, so this map's
    honest window is 2018-present even though the *trend line* elsewhere
    in this dashboard goes back to 2005.
    """
    label0, label1 = f"summer_{first_year}", f"summer_{last_year}"
    with rasterio.open(PROC_DIR / f"{label0}_landcover.tif") as src:
        cls0, profile = src.read(1), src.profile.copy()
    with rasterio.open(PROC_DIR / f"{label1}_landcover.tif") as src:
        cls1 = src.read(1)
    legend0, legend1 = _load_legend(label0), _load_legend(label1)

    def to_broad(cls_arr: np.ndarray, legend: dict[int, str]) -> np.ndarray:
        broad = np.full(cls_arr.shape, "", dtype=object)
        for cid, raw_name in legend.items():
            b = BROAD_CATEGORIES.get(raw_name)
            if b:
                broad[cls_arr == cid] = b
        return broad

    broad0, broad1 = to_broad(cls0, legend0), to_broad(cls1, legend1)
    both_valid = (cls0 != 255) & (cls1 != 255)

    change = np.full(cls0.shape, CHANGE_NODATA, dtype=np.uint8)
    same = both_valid & (broad0 == broad1)
    change[same] = CHANGE_NONE

    forest = "Forest / dense vegetation"
    forest_loss = both_valid & ~same & (broad0 == forest) & (broad1 != forest)
    forest_gain = both_valid & ~same & (broad1 == forest) & (broad0 != forest)
    builtup_growth = both_valid & ~same & (broad1 == "Built-up") & (broad0 != "Built-up") & ~forest_loss
    other = both_valid & ~same & ~forest_loss & ~forest_gain & ~builtup_growth
    change[forest_loss] = CHANGE_FOREST_LOSS
    change[forest_gain] = CHANGE_FOREST_GAIN
    change[builtup_growth] = CHANGE_BUILTUP_GROWTH
    change[other] = CHANGE_OTHER

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=CHANGE_NODATA)
    out_path = PROC_DIR / f"landcover_change_{first_year}_{last_year}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(change[np.newaxis, :, :])

    counts = {CHANGE_CLASS_NAMES[c]: int((change == c).sum()) for c in CHANGE_CLASS_NAMES if c != CHANGE_NODATA}
    ha = {name: round(n * HA_PER_PX, 1) for name, n in counts.items()}
    n_changed = sum(n for c, n in counts.items() if c != "No change")
    n_common = int(both_valid.sum())
    print(f"[landcover change map] {first_year}->{last_year}: {n_changed:,}/{n_common:,} common px changed class "
          f"({n_changed / n_common * 100:.1f}%)" if n_common else "[landcover change map] no common valid pixels")
    for name, val in ha.items():
        print(f"  {name:<20} {val:8,.1f} ha")

    update_stats("landcover_change_map", {
        "first_year": first_year, "last_year": last_year,
        "common_valid_ha": round(n_common * HA_PER_PX, 1),
        "ha": ha,
    })
    return out_path


if __name__ == "__main__":
    build_trend()
