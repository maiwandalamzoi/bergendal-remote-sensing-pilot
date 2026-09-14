"""
Real modelled air-quality concentrations for Berg en Dal from RIVM's Atlas
Leefomgeving WCS -- NO2, PM10, PM2.5 at 1x1 km resolution, the official
Dutch national air-quality maps (the "NSL monitoring" grids used in actual
policy reporting, not a satellite product itself but the ground-truth layer
a satellite-based air-quality pitch would be judged against). Plus NH3
concentration (see _fetch_nh3 below) from RIVM's separate GCN download.

NOT included here: NH3 / nitrogen *deposition* (mol N/ha/yr -- the number
Dutch agricultural permitting actually runs on, via AERIUS). RIVM's Atlas
Leefomgeving WCS -- checked directly -- carries NO2/PM10/PM2.5/EC but no
NH3 or deposition layer at all, confirmed again against its live
GetCapabilities. NH3 *concentration* (ug/m3, a different, real, and
directly downloadable product -- see _fetch_nh3) is not the same number:
it is a real proxy for ammonia in the air, not a deposition-on-nature-area
figure, and cannot substitute for an AERIUS calculation in an actual
permitting decision -- that distinction is carried through into the
dashboard/methodology text, not glossed over.

Source: https://data.rivm.nl/geo/alo/wcs (RIVM, no key)
"""
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.mask import mask as rio_mask

from statsutil import update_stats

WCS_URL = "https://data.rivm.nl/geo/alo/wcs"
# RIVM's GCN (Grootschalige Concentratiekaarten Nederland) download -- a
# real, open, directly-downloadable 1x1km RD-New ASCII grid, no WCS/API and
# no key needed. Confirmed live via data.rivm.nl/data/gcn/ directory
# listing (conc_NH3_<year>.zip exists there alongside NO2/PM10/PM2.5/EC/etc,
# same RIVM product family as the WCS coverages above, just distributed as
# a static file instead of a queryable service).
GCN_NH3_URL = "https://data.rivm.nl/data/gcn/conc_NH3_2024.zip"
RD_BOUNDARY_PATH = Path(__file__).resolve().parent.parent / "data" / "bergendal_boundary_rd.geojson"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# pollutant -> (WCS coverage id, WHO 2021 annual guideline, ug/m3)
COVERAGES = {
    "NO2": ("alo__rivm_nsl_20260401_gm_NO22024", 10),
    "PM10": ("alo__rivm_nsl_20260401_gm_PM102024", 15),
    "PM25": ("alo__rivm_nsl_20260401_gm_PM252024", 5),
}


def _fetch_one(pollutant: str, coverage_id: str, who_guideline: float) -> dict:
    gdf = gpd.read_file(RD_BOUNDARY_PATH)
    minx, miny, maxx, maxy = gdf.total_bounds

    params = {
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage",
        "coverageId": coverage_id, "format": "image/tiff",
        "subset": [f"X({minx},{maxx})", f"Y({miny},{maxy})"],
    }
    r = requests.get(WCS_URL, params=params, timeout=60)
    r.raise_for_status()
    raw_path = RAW_DIR / f"{pollutant.lower()}_raw.tif"
    raw_path.write_bytes(r.content)

    with rasterio.open(raw_path) as src:
        clipped, transform = rio_mask(src, gdf.geometry, crop=True, nodata=src.nodata)
        profile = src.profile.copy()
        profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=transform)

    out_path = RAW_DIR / f"{pollutant.lower()}.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clipped)
    raw_path.unlink()

    # A real bug found in fetch_air_quality_trend.py's older-year coverages
    # (checked directly): some declare nodata=0.0 but actually fill nodata
    # pixels with float32's most-negative value (-3.4e38), which corrupts a
    # mean computed by excluding only the *declared* nodata. A physical
    # sanity bound (no real concentration is negative or above 1000 ug/m3)
    # guards this coverage the same way regardless of which sentinel it
    # turns out to actually use.
    band = clipped[0]
    valid = band[(band != profile["nodata"]) & (band > -1000) & (band < 1000)]
    result = {
        "mean_ug_m3": round(float(valid.mean()), 1),
        "min_ug_m3": round(float(valid.min()), 1),
        "max_ug_m3": round(float(valid.max()), 1),
        "who_guideline_ug_m3": who_guideline,
        "pct_area_over_who_guideline": round(float((valid > who_guideline).mean() * 100), 1),
        "year": 2024,
    }
    print(f"{pollutant}: mean={result['mean_ug_m3']} ug/m3 (WHO guideline {who_guideline}), "
          f"{result['pct_area_over_who_guideline']}% of area over guideline")
    return result


# RIVM's GCN only publishes NH3 for two real (non-projected) years --
# 2024 and 2025; 2030/2035/2040 are policy-scenario projections, not
# history, and are deliberately not fetched here as if they were real
# measured/modelled years. That means no 12-year trend is possible for NH3
# the way there is for NO2/PM10/PM2.5/EC -- a real data-availability limit,
# not a pipeline gap, and documented as such rather than worked around.
GCN_NH3_YEARS = [2024, 2025]


def _fetch_nh3(year: int) -> dict:
    """RIVM's GCN NH3 concentration grid -- a real, open download (not the
    WCS above, which genuinely has no NH3 coverage). Same RIVM
    concentration-map family and resolution as NO2/PM10/PM2.5, produced by
    RIVM's OPS-pro atmospheric dispersion model, calibrated against real
    LML/MAN station measurements (per the product's own metadata PDF,
    downloaded and read directly, not assumed). Explicitly ammonia
    *concentration* in air (ug/m3) -- not nitrogen *deposition* on nature
    areas (mol N/ha/yr, the AERIUS/GDN figure farm permitting runs on),
    which this grid does not contain and which stays a real, separate gap."""
    url = f"https://data.rivm.nl/data/gcn/conc_NH3_{year}.zip"
    zip_path = RAW_DIR / f"conc_nh3_{year}.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    zip_path.write_bytes(r.content)

    with zipfile.ZipFile(zip_path) as zf:
        asc_name = next(n for n in zf.namelist() if n.endswith(".asc"))
        zf.extract(asc_name, RAW_DIR)
    asc_path = RAW_DIR / asc_name

    gdf = gpd.read_file(RD_BOUNDARY_PATH)
    with rasterio.open(asc_path) as src:
        # The .asc carries no CRS of its own (Esri ASCII grid format has no
        # such field) -- RIVM's own metadata PDF for this product states
        # RD-New explicitly, and the boundary file this pipeline already
        # uses for every other RIVM/PDOK layer is the same EPSG:28992, so
        # this is read from the product's documentation, not guessed.
        profile = src.profile.copy()
        profile["crs"] = "EPSG:28992"
        data = src.read()
        nodata = src.nodata

    tmp_tif = RAW_DIR / f"nh3_{year}_rd.tif"
    profile.update(driver="GTiff")
    with rasterio.open(tmp_tif, "w", **profile) as dst:
        dst.write(data)

    with rasterio.open(tmp_tif) as src:
        clipped, _ = rio_mask(src, gdf.geometry, crop=True, nodata=nodata)

    for p in (zip_path, asc_path, tmp_tif):
        p.unlink(missing_ok=True)
    prj_path = RAW_DIR / asc_name.replace(".asc", ".prj")
    prj_path.unlink(missing_ok=True)

    band = clipped[0]
    valid = band[(band != nodata) & (band > -100) & (band < 1000)]
    result = {
        "mean_ug_m3": round(float(valid.mean()), 2),
        "min_ug_m3": round(float(valid.min()), 2),
        "max_ug_m3": round(float(valid.max()), 2),
        "year": year,
        "resolution_m": 1000,
        "model": "OPS-pro 5.3.1.0",
        "uncertainty_pct": "20-25",  # RIVM's own stated sigma for this product
        "note": "Ammonia (NH3) concentration in air, not nitrogen deposition on nature areas "
                "(that figure -- mol N/ha/yr, via AERIUS/GDN -- is a separate RIVM product not "
                "included in this download and still not fetched here). No WHO guideline exists "
                "for NH3 in the way there is for NO2/PM10/PM2.5.",
    }
    print(f"NH3 {year}: mean={result['mean_ug_m3']} ug/m3 (concentration, RIVM GCN, "
          f"OPS-pro model, +/-{result['uncertainty_pct']}% per RIVM's own stated uncertainty)")
    return result


def fetch_all() -> None:
    results = {}
    for pollutant, (coverage_id, guideline) in COVERAGES.items():
        results[pollutant] = _fetch_one(pollutant, coverage_id, guideline)
    try:
        by_year = {year: _fetch_nh3(year) for year in GCN_NH3_YEARS}
        latest_year = max(by_year)
        results["NH3"] = by_year[latest_year]
        results["NH3"]["prior_years"] = {
            str(y): {"mean_ug_m3": v["mean_ug_m3"]} for y, v in by_year.items() if y != latest_year
        }
    except Exception as exc:  # a real network/format failure, not silently swallowed
        results["_not_available"] = {
            "NH3": f"RIVM's GCN NH3 concentration download failed at fetch time ({exc}) -- "
                   "the file normally exists at data.rivm.nl/data/gcn/, confirmed reachable "
                   "directly in this pipeline's own development; re-run this script to retry.",
        }
    results.setdefault("_not_available", {})["NH3_deposition"] = (
        "Nitrogen *deposition* (mol N/ha/yr, via RIVM's AERIUS/GDN product) is still not "
        "included -- it is a separate product from the NH3 concentration figure above, "
        "distributed through aerius.nl rather than a queryable service or the GCN download "
        "used here. Concentration is a real, useful proxy but not a substitute for an actual "
        "AERIUS deposition calculation in a permitting decision."
    )
    update_stats("air_quality", results)


if __name__ == "__main__":
    fetch_all()
