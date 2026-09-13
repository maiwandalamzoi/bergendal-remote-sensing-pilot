"""
Per-field NDVI — not just "this municipality's mean NDVI dropped 0.022",
but a real number for each of the 3,787 BRP parcels individually, for both
dates. This is what lets the dashboard answer "how is *this* field doing"
instead of only "how is the municipality doing".

Zonal stats done as one vectorised pass rather than 3,787 individual
polygon masks: rasterize every parcel's integer id onto the same 10m grid
the NDVI rasters already live on, then bincount-average NDVI per id. Orders
of magnitude faster than looping crop-per-polygon, and it's the same
rasterize() call visualize.py already uses for the BRP category layer.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
BRP_PATH = RAW_DIR / "brp_parcels.geojson"


def _zonal_mean_ndvi(parcel_id_raster: np.ndarray, n_parcels: int, label: str) -> np.ndarray:
    with rasterio.open(PROC_DIR / f"{label}_indices.tif") as src:
        ndvi = src.read(1)
        valid = src.read(7) > 0

    ids = np.where(valid, parcel_id_raster, 0).ravel()
    vals = np.where(valid, ndvi, 0).ravel()

    sums = np.bincount(ids, weights=vals, minlength=n_parcels + 1)
    counts = np.bincount(ids, minlength=n_parcels + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return means  # index 0 = background (unused), 1..n_parcels = parcel mean NDVI


def compute() -> Path:
    gdf = gpd.read_file(BRP_PATH)
    with rasterio.open(PROC_DIR / "summer_2025_indices.tif") as ref:
        ref_profile = ref.profile.copy()
    gdf_proj = gdf.to_crs(ref_profile["crs"])

    gdf_proj["_pid"] = np.arange(1, len(gdf_proj) + 1)
    shapes = list(zip(gdf_proj.geometry, gdf_proj["_pid"]))
    parcel_id_raster = rasterize(
        shapes, out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"], fill=0, dtype="int32",
    )

    n = len(gdf_proj)
    ndvi_2025 = _zonal_mean_ndvi(parcel_id_raster, n, "summer_2025")
    ndvi_2024 = _zonal_mean_ndvi(parcel_id_raster, n, "summer_2024")

    gdf["ndvi_2025"] = ndvi_2025[gdf_proj["_pid"].values]
    gdf["ndvi_2024"] = ndvi_2024[gdf_proj["_pid"].values]
    gdf["ndvi_change"] = gdf["ndvi_2025"] - gdf["ndvi_2024"]

    # A handful of tiny/sliver parcels fall entirely on masked-cloud or
    # off-grid pixels and come back NaN -- drop those from stats, keep them
    # in the map (they'll just show as "no data" for NDVI coloring).
    n_valid = gdf["ndvi_2025"].notna().sum()

    gdf.to_file(BRP_PATH, driver="GeoJSON")
    print(f"Zonal NDVI computed for {n_valid:,}/{n:,} parcels "
          f"(mean field NDVI 2025={gdf['ndvi_2025'].mean():.3f}, "
          f"mean change={gdf['ndvi_change'].mean():+.3f})")
    return BRP_PATH


if __name__ == "__main__":
    compute()
