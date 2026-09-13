"""
Historical BRP crop registrations for the AOI, one snapshot per year -- the
real data crop_rotation.py needs to show what a field grew in 2021, 2022,
2023, 2024 before the boundaries as registered in 2025 (fetch_brp.py).

Source: checked directly against the BRP WFS's own GetCapabilities --
`brpgewaspercelen:BrpGewas` is the only feature type, and every feature
in it carries `"jaar": 2025` (the current year only). There is no
per-year WFS. PDOK's ATOM download feed
(https://service.pdok.nl/rvo/gewaspercelen/atom/index.xml) has the
historical years, but only as whole-Netherlands GeoPackages (2.5-3GB
each for 2020+) -- multi-gigabyte downloads for a ~9km-municipality AOI.

Rather than pulling a whole year's file, this reads each GeoPackage over
HTTP via GDAL's /vsicurl/ virtual filesystem with a bounding-box filter,
which uses the GeoPackage's own spatial (rtree) index to fetch only the
AOI's rows instead of the whole multi-GB file -- confirmed directly
(2,126 features for one test year in ~3 minutes, not a multi-GB
download). Still slow enough (HTTP range requests, not a local file) that
this is meant to be run once, standalone, not as part of every
`run_pipeline.py` pass -- each year is cached to disk and skipped on a
rerun.

    python fetch_brp_history.py
"""
from pathlib import Path

import geopandas as gpd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RD_BOUNDARY_PATH = Path(__file__).resolve().parent.parent / "data" / "bergendal_boundary_rd.geojson"

GPKG_URL_TMPL = ("/vsicurl/https://service.pdok.nl/rvo/gewaspercelen/atom/downloads/"
                  "brpgewaspercelen_definitief_{year}.gpkg")

# 2025 is already covered by fetch_brp.py (the live WFS, current year).
# Further back than 2020 starts hitting noticeably larger/slower files and
# category definitions drift more from the current schema -- 2020-2024
# gives a real 6-year rotation window (2020-2025) without that.
HISTORY_YEARS = [2020, 2021, 2022, 2023, 2024]


def fetch_year(year: int) -> Path:
    out_path = DATA_DIR / f"brp_parcels_{year}.geojson"
    if out_path.exists():
        print(f"[brp {year}] already on disk, skipping")
        return out_path

    boundary = gpd.read_file(RD_BOUNDARY_PATH)
    minx, miny, maxx, maxy = boundary.total_bounds
    url = GPKG_URL_TMPL.format(year=year)
    print(f"[brp {year}] reading {url}")
    print(f"           (bbox-filtered over HTTP via the GeoPackage's own spatial index -- "
          f"not a {year} full-country download; takes a couple of minutes)")
    gdf = gpd.read_file(url, bbox=(minx, miny, maxx, maxy))
    if len(gdf) == 0:
        raise RuntimeError(f"[brp {year}] 0 features returned -- check the URL/year still exists")
    gdf = gpd.clip(gdf, boundary)
    gdf["area_ha"] = gdf.geometry.area / 10_000
    gdf.to_crs(4326).to_file(out_path, driver="GeoJSON")
    print(f"[brp {year}] {len(gdf):,} parcels, {gdf['area_ha'].sum():,.0f} ha -> saved {out_path}")
    return out_path


def fetch_all() -> None:
    for year in HISTORY_YEARS:
        fetch_year(year)


if __name__ == "__main__":
    fetch_all()
