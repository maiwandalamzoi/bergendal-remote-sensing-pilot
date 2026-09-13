"""
Pulls Sentinel-1 RTC (radiometrically terrain-corrected) SAR backscatter for
the Berg en Dal AOI from Planetary Computer -- VV and VH, already calibrated
to linear gamma0, no cloud to wait out. RTC tiles for this AOI are natively
in UTM zone 32N (EPSG:32632) rather than the zone 31N (EPSG:32631) grid our
Sentinel-2 mosaic landed on -- Berg en Dal sits almost exactly on the 6°E
zone boundary. Kept in its native grid here; preprocess_sar.py reprojects
onto the optical grid only where the two are actually compared pixel-to-pixel.
"""
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge

import planetary_computer
import pystac_client
from shapely.geometry import shape

from aoi import bbox_wgs84, geometry_wgs84

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
BANDS = ["vv", "vh"]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _items_for_date(date_str: str):
    """A single scene covering the *whole* AOI on one date.

    IW swaths are ~250km wide, comfortably bigger than the AOI, but Berg en
    Dal sits close to a swath edge on some passes -- a scene can intersect
    the AOI's bounding box while its actual footprint (a diagonal parallelogram,
    not a rectangle) only clips one corner of it. Filtering by bbox overlap
    alone silently returns those partial scenes; checking that the scene's
    real footprint *contains* the AOI polygon catches it before the fetch.
    """
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=[COLLECTION], bbox=bbox_wgs84(), datetime=f"{date_str}/{date_str}")
    items = [it for it in search.items() if shape(it.geometry).contains(geometry_wgs84())]
    if not items:
        raise RuntimeError(f"No Sentinel-1 RTC scene fully covering the AOI on {date_str}")
    return items


def find_covering_date(center_date: str, window_days: int = 10) -> str:
    """Nearest date to center_date with a scene that fully covers the AOI."""
    from datetime import datetime, timedelta

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    center = datetime.fromisoformat(center_date)
    start = (center - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (center + timedelta(days=window_days)).strftime("%Y-%m-%d")
    search = catalog.search(collections=[COLLECTION], bbox=bbox_wgs84(), datetime=f"{start}/{end}")
    aoi = geometry_wgs84()
    candidates = [it for it in search.items() if shape(it.geometry).contains(aoi)]
    if not candidates:
        raise RuntimeError(f"No fully-covering Sentinel-1 scene within {window_days}d of {center_date}")
    best = min(candidates, key=lambda it: abs(
        (datetime.fromisoformat(it.properties["datetime"][:19]) - center).total_seconds()
    ))
    return best.properties["datetime"][:10]


def fetch_date(date_str: str, label: str) -> Path:
    """Fetch VV+VH for the scene(s) covering the AOI on one date.

    Writes data/raw/{label}.tif (2-band stack: VV, VH, linear gamma0,
    native UTM 32N grid) and returns its path.
    """
    items = _items_for_date(date_str)
    print(f"[{label}] date={date_str} scenes={[it.id for it in items]} "
          f"orbit={items[0].properties.get('sat:orbit_state')}")

    aoi_geom = geometry_wgs84()
    arrays, out_profile = [], None
    for band in BANDS:
        datasets = [rasterio.open(it.assets[band].href) for it in items]
        aoi_in_crs = gpd.GeoSeries([aoi_geom], crs="EPSG:4326").to_crs(datasets[0].crs)
        minx, miny, maxx, maxy = aoi_in_crs.total_bounds
        buf = 500
        mosaic, transform = rio_merge(datasets, bounds=(minx - buf, miny - buf, maxx + buf, maxy + buf))
        profile = datasets[0].profile.copy()
        profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, count=1)
        for ds in datasets:
            ds.close()

        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**profile) as tmp:
                tmp.write(mosaic)
                clipped, clip_transform = rio_mask(tmp, aoi_in_crs.geometry, crop=True, nodata=0)
            profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=clip_transform)

        arrays.append(clipped[0])
        out_profile = profile

    import numpy as np
    stack = np.stack(arrays)
    out_profile.update(count=len(BANDS), dtype=stack.dtype)
    out_path = RAW_DIR / f"{label}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(stack)
        dst.descriptions = ("VV", "VH")
    print(f"  saved {out_path}  shape={stack.shape}  crs={out_profile['crs']}")
    return out_path


if __name__ == "__main__":
    date_str = find_covering_date("2025-08-11")  # nearest full-coverage pass to the optical date
    fetch_date(date_str, "sar_2025")
