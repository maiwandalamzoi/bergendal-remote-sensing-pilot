"""
Pulls real Sentinel-2 L2A imagery for the Berg en Dal AOI from Microsoft's
Planetary Computer STAC catalog (public, no API key required) and writes a
clipped, band-stacked GeoTIFF per date to data/raw/.

Bands: B02 (blue), B03 (green), B04 (red), B08 (NIR) at 10 m, and SCL
(scene classification layer, cloud/shadow/water/vegetation flags) at 20 m,
resampled to the 10 m grid.

Only the AOI window is read off each remote Cloud-Optimized GeoTIFF — this
does not download full 110x110 km tiles.
"""
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.mask import mask as rio_mask
import geopandas as gpd

from aoi import bbox_wgs84, geometry_wgs84

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B08", "SCL"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _best_items(datetime_range: str, max_cloud: float = 10.0):
    """All Sentinel-2 items over the AOI within a date range, cloud-sorted.

    Returns a list because the AOI straddles two MGRS tiles (31UFT / 31UGT)
    -- a single date can need more than one scene to fully cover it.
    """
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox_wgs84(),
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {datetime_range} under {max_cloud}% cloud")
    return sorted(items, key=lambda it: it.properties["eo:cloud_cover"])


def _items_for_single_date(items, date_str: str):
    """Filter a sorted item list down to those acquired on one calendar date."""
    same_day = [it for it in items if it.properties["datetime"].startswith(date_str)]
    return same_day or items[:1]


def _read_band_clipped(band_items, band: str, dst_res=10):
    """Mosaic (all tiles forced onto one 10 m grid) + clip one band to the AOI.

    Both dates and both MGRS tiles (31UFT/31UGT) share UTM zone 31N, so no
    reprojection is needed -- only resampling every band onto the same
    10 m grid (SCL is natively 20 m) and clipping to the same polygon,
    which keeps every band and every date pixel-aligned.
    """
    aoi_geom = geometry_wgs84()
    datasets = [rasterio.open(it.assets[band].href) for it in band_items]

    # Restrict the remote read to the AOI window (+500m buffer) instead of
    # pulling each full ~110x110km tile -- these are windowed HTTP range
    # requests against the COG, not full downloads.
    aoi_in_src_crs = gpd.GeoSeries([aoi_geom], crs="EPSG:4326").to_crs(datasets[0].crs)
    minx, miny, maxx, maxy = aoi_in_src_crs.total_bounds
    buf = 500
    read_bounds = (minx - buf, miny - buf, maxx + buf, maxy + buf)

    mosaic, transform = rio_merge(
        datasets,
        bounds=read_bounds,
        res=dst_res,
        resampling=Resampling.nearest if band == "SCL" else Resampling.bilinear,
    )
    profile = datasets[0].profile.copy()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, count=1)
    for ds in datasets:
        ds.close()

    # Clip to the exact municipal polygon (reprojected into the raster's CRS)
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as tmp:
            tmp.write(mosaic)
            aoi_in_crs = gpd.GeoSeries([aoi_geom], crs="EPSG:4326").to_crs(tmp.crs)
            clipped, clip_transform = rio_mask(tmp, aoi_in_crs.geometry, crop=True, nodata=0)
        profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=clip_transform)

    return clipped[0], profile


def fetch_date(datetime_range: str, label: str, max_cloud: float = 10.0) -> Path:
    """Fetch, mosaic, and clip all BANDS for the least-cloudy date in a range.

    Writes data/raw/{label}.tif (5-band stack: B02,B03,B04,B08,SCL) and
    returns its path.
    """
    items = _best_items(datetime_range, max_cloud)
    target_date = items[0].properties["datetime"][:10]
    same_date_items = _items_for_single_date(items, target_date)
    processing_baseline = same_date_items[0].properties.get("s2:processing_baseline", "")
    print(f"[{label}] date={target_date} tiles={[it.id.split('_')[-2] for it in same_date_items]} "
          f"cloud={same_date_items[0].properties['eo:cloud_cover']:.2f}% "
          f"baseline={processing_baseline}")

    arrays, profile = [], None
    for band in BANDS:
        arr, prof = _read_band_clipped(same_date_items, band)
        arrays.append(arr)
        profile = prof  # identical grid for every band -- last one wins, all match

    stack = np.stack(arrays)
    out_profile = profile.copy()
    out_profile.update(count=len(BANDS), dtype=stack.dtype)
    out_path = DATA_DIR / f"{label}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(stack)
        dst.descriptions = tuple(BANDS)
        # preprocess.py needs this: only baseline >= 04.00 (25 Jan 2022) DN
        # values carry the +1000 reflectance offset. Stamped on the file
        # itself so a year fetched today still preprocesses correctly
        # however long it then sits on disk before that step runs.
        dst.update_tags(s2_processing_baseline=processing_baseline)
    print(f"  saved {out_path}  shape={stack.shape}")
    return out_path


if __name__ == "__main__":
    fetch_date("2025-08-01/2025-08-31", "summer_2025", max_cloud=5)
    fetch_date("2024-08-01/2024-08-31", "summer_2024", max_cloud=5)
