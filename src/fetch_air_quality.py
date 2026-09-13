"""
Real modelled air-quality concentrations for Berg en Dal from RIVM's Atlas
Leefomgeving WCS -- NO2, PM10, PM2.5 at 1x1 km resolution, the official
Dutch national air-quality maps (the "NSL monitoring" grids used in actual
policy reporting, not a satellite product itself but the ground-truth layer
a satellite-based air-quality pitch would be judged against).

NOT included here: NH3 / nitrogen deposition (the number Dutch agricultural
permitting actually runs on). RIVM's Atlas Leefomgeving WCS -- checked
directly, see below -- carries NO2/PM10/PM2.5/EC but no NH3 or stikstof
deposition layer. That figure comes from RIVM's separate GDN/AERIUS product
(aerius.nl), distributed as an annual grid download rather than a queryable
WCS/WFS -- wiring that in is a real next step, not something to fake a
number for here.

Source: https://data.rivm.nl/geo/alo/wcs (RIVM, no key)
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.mask import mask as rio_mask

from statsutil import update_stats

WCS_URL = "https://data.rivm.nl/geo/alo/wcs"
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


def fetch_all() -> None:
    results = {}
    for pollutant, (coverage_id, guideline) in COVERAGES.items():
        results[pollutant] = _fetch_one(pollutant, coverage_id, guideline)
    results["_not_available"] = {
        "NH3": "No queryable WCS/WFS from RIVM's Atlas Leefomgeving for ammonia or nitrogen "
               "deposition -- the figure Dutch farm nitrogen permitting actually uses comes from "
               "RIVM's separate GDN/AERIUS product (aerius.nl), an annual grid download. "
               "Confirmed absent from this service, not simply unfetched.",
    }
    update_stats("air_quality", results)


if __name__ == "__main__":
    fetch_all()
