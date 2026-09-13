"""
Real field boundaries and what's actually growing in them: BRP
(Basisregistratie Gewaspercelen) is the Dutch government's registry of every
subsidised agricultural parcel, filed by the farmer each year -- exact
polygon plus a specific crop code ("Aardappelen, poot NAK", "Grasland,
blijvend", "Natuurterreinen (incl. heide)", ...). This is the ground-truth
join the concept note's Phase 2 asked for: it doesn't just estimate "grass/
farmland" from spectral signature the way the KMeans land cover does, it
names the actual crop.

Source: https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0 (RVO/PDOK, no key)

Coverage caveat: BRP only contains parcels a farmer registered for
subsidy -- it does not cover unmanaged forest, so it complements the
satellite-derived land cover rather than replacing it (see stats["brp"]
vs stats["landcover_summer_2025"] in the dashboard).
"""
import json
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import shape

from statsutil import update_stats

WFS_URL = "https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0"
TYPENAME = "brpgewaspercelen:BrpGewas"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RD_BOUNDARY_PATH = Path(__file__).resolve().parent.parent / "data" / "bergendal_boundary_rd.geojson"

# Dutch category -> a short English gloss for the dashboard, not a translation
# of every crop name (there are hundreds) -- just the top-level grouping.
CATEGORY_EN = {
    "Grasland": "Grassland / pasture",
    "Bouwland": "Arable / cropland",
    "Natuurterrein": "Nature terrain",
    "Braakland": "Fallow land",
    "Overige": "Other",
    "Sloot": "Ditch / waterway",
    "Landschapselement": "Landscape feature",
}


def fetch_all() -> Path:
    boundary = gpd.read_file(RD_BOUNDARY_PATH)
    minx, miny, maxx, maxy = boundary.total_bounds

    features, start = [], 0
    page_size = 1000
    while True:
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeName": TYPENAME, "outputFormat": "application/json",
            "bbox": f"{minx},{miny},{maxx},{maxy},urn:ogc:def:crs:EPSG::28992",
            "count": page_size, "startIndex": start,
        }
        r = requests.get(WFS_URL, params=params, timeout=60)
        r.raise_for_status()
        page = r.json()["features"]
        features.extend(page)
        if len(page) < page_size:
            break
        start += page_size

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:28992")
    gdf = gpd.clip(gdf, boundary)  # bbox over-fetched; clip to the real polygon
    gdf["area_ha"] = gdf.geometry.area / 10_000

    out_path = DATA_DIR / "brp_parcels.geojson"
    gdf.to_crs(4326).to_file(out_path, driver="GeoJSON")

    total_ha = float(gdf["area_ha"].sum())
    by_category = gdf.groupby("category")["area_ha"].sum().sort_values(ascending=False)
    by_crop = gdf.groupby("gewas")["area_ha"].sum().sort_values(ascending=False)

    print(f"BRP parcels: {len(gdf):,} covering {total_ha:.0f} ha inside the municipal boundary")
    print("by category:")
    for cat, ha in by_category.items():
        print(f"  {cat:<20} {ha:7.0f} ha  ({ha/total_ha*100:4.1f}%)  [{CATEGORY_EN.get(cat, cat)}]")
    print("top 8 crops by area:")
    for crop, ha in by_crop.head(8).items():
        print(f"  {crop:<45} {ha:6.0f} ha")

    update_stats("brp", {
        "year": int(gdf["jaar"].iloc[0]) if len(gdf) else None,
        "n_parcels": len(gdf),
        "total_area_ha": round(total_ha, 1),
        "by_category_ha": {cat: round(float(ha), 1) for cat, ha in by_category.items()},
        "top_crops_ha": {crop: round(float(ha), 1) for crop, ha in by_crop.head(10).items()},
    })
    return out_path


if __name__ == "__main__":
    fetch_all()
