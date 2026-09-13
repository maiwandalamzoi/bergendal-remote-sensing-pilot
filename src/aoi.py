"""
Area of interest for the Berg en Dal pilot.

Loads the official municipal boundary (CBS/Kadaster "bestuurlijkegebieden"
dataset, gemeentecode GM1945) served by PDOK, reprojects it from RD New
(EPSG:28992) to WGS84 (EPSG:4326), and exposes both the polygon and its
bounding box for use by the satellite-data fetchers.

Source: https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0
"""
from pathlib import Path
import json

import geopandas as gpd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RD_PATH = DATA_DIR / "bergendal_boundary_rd.geojson"
WGS84_PATH = DATA_DIR / "bergendal_boundary_wgs84.geojson"

GEMEENTECODE = "GM1945"
NAAM = "Berg en Dal"


def load_boundary() -> gpd.GeoDataFrame:
    """Return the Berg en Dal municipal boundary as a GeoDataFrame in EPSG:4326.

    Reprojects and caches a WGS84 copy next to the original RD New download
    so downstream scripts don't pay the reprojection cost twice.
    """
    if WGS84_PATH.exists():
        return gpd.read_file(WGS84_PATH)

    gdf = gpd.read_file(RD_PATH)  # native CRS: EPSG:28992 (RD New)
    gdf = gdf.set_crs(epsg=28992, allow_override=True).to_crs(epsg=4326)
    gdf.to_file(WGS84_PATH, driver="GeoJSON")
    return gdf


def bbox_wgs84(margin_deg: float = 0.0) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) in EPSG:4326, optionally padded by margin_deg."""
    gdf = load_boundary()
    minx, miny, maxx, maxy = gdf.total_bounds
    return (minx - margin_deg, miny - margin_deg, maxx + margin_deg, maxy + margin_deg)


def geometry_wgs84():
    """Shapely (multi)polygon of the municipal boundary, EPSG:4326."""
    gdf = load_boundary()
    return gdf.geometry.iloc[0]


if __name__ == "__main__":
    gdf = load_boundary()
    print(f"{NAAM} ({GEMEENTECODE})")
    print("area (approx, deg^2 for a quick sanity check):", gdf.geometry.area.iloc[0])
    print("bbox WGS84:", bbox_wgs84())
    print("saved:", WGS84_PATH)
