"""
Multi-year air quality, 2013-2024: real annual-mean NO2/PM10/PM2.5/EC grids
from RIVM's Atlas Leefomgeving WCS, the same service fetch_air_quality.py
already uses for the current year -- checked directly against its own
GetCapabilities and confirmed to carry a full run of annual coverages back
to 2013, not just the latest year.

EC (elemental carbon) is included as the closest real proxy this service
has to a combustion/GHG signal: it's soot -- a direct marker of fossil-fuel
and biomass combustion -- not a CO2 or NH3 number itself. That's a real
limit, not smoothed over: this pipeline could not find a free, keyless,
municipality-resolution API for either CO2/GHG or NH3/nitrogen deposition
(both checked directly -- see the module docstrings on
fetch_air_quality.py and README.md's findings -- Klimaatmonitor's own
municipal CO2 figures sit behind a login-gated OData API, confirmed by a
401 "Guest user group not found" response, not merely unfetched; RIVM's
GDN/AERIUS nitrogen-deposition product has no public WCS/WFS at all).
NO2 and EC trends are the honest, real substitute this pass can offer.

    python fetch_air_quality_trend.py
"""
import re
from pathlib import Path

import geopandas as gpd
import rasterio
import requests
from rasterio.mask import mask as rio_mask

from statsutil import update_stats

WCS_URL = "https://data.rivm.nl/geo/alo/wcs"
RD_BOUNDARY_PATH = Path(__file__).resolve().parent.parent / "data" / "bergendal_boundary_rd.geojson"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2013, 2025))
POLLUTANTS = {
    "NO2": re.compile(r"(?:^|_)no2(\d{4})", re.I),
    "PM10": re.compile(r"(?:^|_)pm10(\d{4})", re.I),
    "PM25": re.compile(r"(?:^|_)pm25(\d{4})", re.I),
    "EC": re.compile(r"(?:^|_)ec(\d{4})", re.I),
}


def _coverage_ids_by_year() -> dict:
    r = requests.get(WCS_URL, params={
        "service": "WCS", "version": "2.0.1", "request": "GetCapabilities",
    }, timeout=30)
    r.raise_for_status()
    ids = re.findall(r"<wcs:CoverageId>([^<]+)</wcs:CoverageId>", r.text)
    by_pollutant = {p: {} for p in POLLUTANTS}
    for cov_id in ids:
        for pollutant, pat in POLLUTANTS.items():
            m = pat.search(cov_id)
            if m:
                by_pollutant[pollutant][int(m.group(1))] = cov_id
    return by_pollutant


def _fetch_coverage(coverage_id: str) -> float | None:
    gdf = gpd.read_file(RD_BOUNDARY_PATH)
    minx, miny, maxx, maxy = gdf.total_bounds
    r = requests.get(WCS_URL, params={
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage",
        "coverageId": coverage_id, "format": "image/tiff",
        "subset": [f"X({minx},{maxx})", f"Y({miny},{maxy})"],
    }, timeout=60)
    r.raise_for_status()
    tmp_path = RAW_DIR / "_air_trend_tmp.tif"
    tmp_path.write_bytes(r.content)
    with rasterio.open(tmp_path) as src:
        clipped, _ = rio_mask(src, gdf.geometry, crop=True, nodata=src.nodata)
        nodata = src.nodata
    tmp_path.unlink()
    band = clipped[0]
    # A real bug found here: some of the older coverages (checked directly,
    # e.g. 2014's NO2) declare nodata=0.0 in their own GDAL profile but
    # actually fill nodata pixels with float32's most-negative value
    # (-3.4e38) instead -- excluding only the *declared* nodata let that
    # sentinel through into the mean, corrupting it to -inf. A physical
    # sanity bound (no real pollutant concentration is negative or above
    # 1000 ug/m3) catches this regardless of which sentinel a given
    # coverage actually uses.
    valid = band[(band != nodata) & (band > -1000) & (band < 1000)]
    return float(valid.mean()) if valid.size else None


def run() -> dict:
    by_pollutant = _coverage_ids_by_year()
    series = {p: {"years": [], "mean_ug_m3": []} for p in POLLUTANTS}

    for pollutant, by_year in by_pollutant.items():
        for year in YEARS:
            if year not in by_year:
                continue
            mean_val = _fetch_coverage(by_year[year])
            if mean_val is None:
                continue
            series[pollutant]["years"].append(year)
            series[pollutant]["mean_ug_m3"].append(round(mean_val, 1))
            print(f"[air trend] {pollutant} {year}: {mean_val:.1f} ug/m3")

    for pollutant, s in series.items():
        if len(s["years"]) >= 2:
            net = s["mean_ug_m3"][-1] - s["mean_ug_m3"][0]
            print(f"[air trend] {pollutant} net change {s['years'][0]}-{s['years'][-1]}: "
                  f"{s['mean_ug_m3'][0]:.1f} -> {s['mean_ug_m3'][-1]:.1f} ug/m3 ({net:+.1f})")

    update_stats("air_quality_trend", {
        "series": series,
        "note": ("EC (elemental carbon / soot) is the closest real combustion proxy this free "
                 "service offers -- not a CO2 or NH3 number. Municipal CO2/GHG (Klimaatmonitor) "
                 "requires an authenticated login (confirmed: 401 on its OData API); NH3/nitrogen "
                 "deposition (RIVM GDN/AERIUS) has no public WCS/WFS at all -- both checked "
                 "directly, not assumed."),
    })
    return series


if __name__ == "__main__":
    run()
