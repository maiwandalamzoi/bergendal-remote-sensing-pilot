"""
Berg en Dal isn't one place -- it's a 2015 merger of the former
municipalities Groesbeek, Millingen aan de Rijn and Ubbergen, and reads
as 13 real villages/hamlets day to day, not one blob. This fetches CBS's
own "wijken" (district) boundaries and per-district population/area --
the same PDOK WFS fetch_cbs.py already uses for the municipality-wide
row, one level down.

`wijken` (13 features here) rather than the finer `buurten` (41 features,
splits each village into its built-up core vs. surrounding
"Buitengebied" countryside): wijken is the level that actually matches a
place name someone would say out loud ("Ooij", "Leuth", "Groesbeek"),
which is what a village filter/selector needs.

Source: https://service.pdok.nl/cbs/wijkenbuurten/2024/wfs/v1_0
(same service, same year, as fetch_cbs.py)

    python fetch_villages.py
"""
from pathlib import Path

import geopandas as gpd
import requests

from statsutil import update_stats

WFS_URL = "https://service.pdok.nl/cbs/wijkenbuurten/2024/wfs/v1_0"
GEMEENTECODE = "GM1945"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_all() -> Path:
    features, start, page_size = [], 0, 1000
    while True:
        r = requests.get(WFS_URL, params={
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeName": "wijkenbuurten:wijken", "outputFormat": "application/json",
            "count": page_size, "startIndex": start,
        }, timeout=30)
        r.raise_for_status()
        page = r.json()["features"]
        features.extend(page)
        if len(page) < page_size:
            break
        start += page_size

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:28992")
    gdf = gdf[gdf["gemeentecode"] == GEMEENTECODE].copy()
    gdf = gdf[gdf["water"] != "JA"]  # drop the "Groot water" / river-only wijk, not a place someone lives
    gdf = gdf.sort_values("aantalInwoners", ascending=False)

    out_path = DATA_DIR / "villages.geojson"
    gdf.to_crs(4326).to_file(out_path, driver="GeoJSON")

    villages = []
    for _, row in gdf.iterrows():
        villages.append({
            "name": row["wijknaam"],
            "population": None if row["aantalInwoners"] == -99997 else int(row["aantalInwoners"]),
            "households": None if row["aantalHuishoudens"] == -99997 else int(row["aantalHuishoudens"]),
            "land_area_ha": None if row["oppervlakteLandInHa"] == -99997 else round(float(row["oppervlakteLandInHa"]), 1),
            "density_per_km2": None if row["bevolkingsdichtheidInwonersPerKm2"] == -99997 else int(row["bevolkingsdichtheidInwonersPerKm2"]),
        })

    print(f"Villages in Berg en Dal ({len(villages)}):")
    for v in villages:
        pop = f"{v['population']:,}" if v["population"] is not None else "n/a"
        area = f"{v['land_area_ha']:,.0f}" if v["land_area_ha"] is not None else "n/a"
        print(f"  {v['name']:<24} {pop:>7} inhabitants  {area:>7} ha")

    update_stats("villages", {"year": 2024, "list": villages})
    return out_path


if __name__ == "__main__":
    fetch_all()
