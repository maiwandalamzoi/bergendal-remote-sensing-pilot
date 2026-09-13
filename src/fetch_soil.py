"""
Real soil properties for Berg en Dal's farmland: organic carbon, pH,
nitrogen, texture (clay/sand/silt), bulk density and cation exchange
capacity, from ISRIC's SoilGrids -- a global, free, no-key soil property
map (250m) built from real profile observations plus environmental
covariates, the same real source zaminai's own /soil endpoint already
uses for Afghanistan.

Checked directly rather than assumed reachable: PDOK/BRO's own Dutch soil
map (Bodemkaart) has no discoverable public WFS/WMS under any of the
naming patterns PDOK otherwise uses (bzk/wenr/wur -- all checked, all
404), and its data.overheid.nl listings only turn up per-province
services (Utrecht, Zeeland, Zuid-Holland) that don't cover Gelderland.
SoilGrids is the fallback that's actually reachable, not the first
choice -- a real methodology note, not a quiet substitution.

SoilGrids answers one point at a time, and a single point can land on
open water in this river-adjacent AOI and come back null (checked
directly) -- so this samples at real BRP agricultural parcel centroids
(guaranteed to be on farmed land, and directly relevant to "how healthy
is the soil under this cropland") rather than a blind grid, and averages
whatever comes back non-null.

    python fetch_soil.py
"""
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests

from statsutil import update_stats

API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
BRP_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "brp_parcels.geojson"

# property -> (unit conversion divisor to get the target_units SoilGrids
# itself documents, human label, unit)
PROPERTIES = {
    "soc": (10, "Soil organic carbon", "g/kg"),       # dg/kg -> g/kg
    "phh2o": (10, "pH (in water)", "pH"),              # pHx10 -> pH
    "nitrogen": (100, "Total nitrogen", "g/kg"),        # cg/kg -> g/kg
    "clay": (10, "Clay content", "%"),                  # g/kg -> % (permille->percent /10)
    "sand": (10, "Sand content", "%"),
    "silt": (10, "Silt content", "%"),
    "bdod": (100, "Bulk density", "kg/dm3"),            # cg/cm3 -> kg/dm3
    "cec": (10, "Cation exchange capacity", "cmol(c)/kg"),
}
DEPTH = "0-5cm"  # topsoil -- most relevant to current land use / farming
N_SAMPLE_POINTS = 40


def _sample_points() -> list[tuple[float, float]]:
    """Centroids of a systematic sample of real BRP parcels -- guaranteed
    on farmed land, not open water, and directly tied to the AOI's actual
    agricultural footprint rather than an arbitrary grid."""
    gdf = gpd.read_file(BRP_PATH)
    farmland = gdf[gdf["category"].isin(["Bouwland", "Grasland"])]
    if len(farmland) > N_SAMPLE_POINTS:
        farmland = farmland.iloc[np.linspace(0, len(farmland) - 1, N_SAMPLE_POINTS).astype(int)]
    # Centroid in RD New (a projected, metric CRS) -- doing this in plain
    # lon/lat would be geometrically off, the same fix already applied to
    # the crop-icon markers in visualize.py.
    pts = farmland.to_crs(28992).geometry.centroid
    pts_wgs = gpd.GeoSeries(pts, crs=28992).to_crs(4326)
    return list(zip(pts_wgs.x, pts_wgs.y))


def _query_point(lon: float, lat: float, retries: int = 3) -> dict:
    for attempt in range(retries):
        r = requests.get(API_URL, params={
            "lon": lon, "lat": lat, "property": list(PROPERTIES.keys()), "depth": DEPTH, "value": "mean",
        }, timeout=30)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 15))
            print(f"[soil] rate-limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        out = {}
        for layer in r.json().get("properties", {}).get("layers", []):
            val = layer["depths"][0]["values"]["mean"]
            out[layer["name"]] = val
        return out
    raise RuntimeError(f"SoilGrids still rate-limiting after {retries} retries")


def run() -> dict:
    points = _sample_points()
    print(f"[soil] sampling {len(points)} farmland-parcel centroids via ISRIC SoilGrids ({DEPTH})...")

    readings = {prop: [] for prop in PROPERTIES}
    n_null = 0
    for i, (lon, lat) in enumerate(points):
        vals = _query_point(lon, lat)
        if all(v is None for v in vals.values()):
            n_null += 1
        else:
            for prop, v in vals.items():
                if v is not None:
                    readings[prop].append(v)
        if i < len(points) - 1:
            time.sleep(5)  # ISRIC's free-tier fair-use limit is tight enough to need real pacing

    result = {"depth": DEPTH, "n_points_sampled": len(points), "n_points_null": n_null, "properties": {}}
    print(f"[soil] {len(points) - n_null}/{len(points)} points returned data "
          f"({n_null} landed on water/no-coverage)")
    for prop, (divisor, label, unit) in PROPERTIES.items():
        vals = readings[prop]
        if not vals:
            result["properties"][prop] = {"label": label, "unit": unit, "mean": None}
            continue
        mean_native = float(np.mean(vals))
        mean_converted = round(mean_native / divisor, 2)
        result["properties"][prop] = {
            "label": label, "unit": unit, "mean": mean_converted, "n": len(vals),
        }
        print(f"  {label:<26} {mean_converted:8.2f} {unit}  (n={len(vals)})")

    update_stats("soil", result)
    return result


if __name__ == "__main__":
    run()
