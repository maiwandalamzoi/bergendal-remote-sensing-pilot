"""
Pulls Landsat Collection 2 Level-2 surface reflectance for the Berg en Dal
AOI from Planetary Computer, for the years before Sentinel-2 L2A coverage
gets reliable here (2005-2017; ndvi_trend.py uses Sentinel-2 from 2018 on).

Planetary Computer's STAC exposes the same common-name asset keys (`red`,
`nir08`, `qa_pixel`) across Landsat 5 (TM), 7 (ETM+) and 8/9 (OLI) even
though the underlying band numbers differ per sensor, so one fetch function
covers all three without a platform switch. One WRS-2 path/row scene fully
covers this AOI (checked directly against the STAC catalogue -- no
multi-tile mosaic needed here, unlike the Sentinel-2 fetch).

Surface reflectance scaling (Collection 2 Level 2, all platforms):
    reflectance = DN * 0.0000275 - 0.2
Cloud/shadow/fill masking uses the QA_PIXEL bit flags Landsat ships
directly (bit 6 = Clear, bit 0 = Fill) rather than a separate scene
classification product.

Landsat 7 scenes from 31 May 2003 onward have the SLC-off diagonal data
gaps (a hardware failure, not a processing choice) -- real, documented
missing stripes, not a bug in this pipeline. They show up as extra
nodata inside the AOI, on top of whatever cloud there is.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape

from aoi import bbox_wgs84, geometry_wgs84

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"
BANDS = ["red", "green", "nir08", "qa_pixel"]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _best_covering_item(datetime_range: str, max_cloud: float = 30.0):
    """Least-cloudy scene in the range whose real footprint contains the
    whole AOI polygon -- same "don't trust bbox overlap alone" check as
    fetch_sentinel1.py, since a WRS-2 scene edge can otherwise clip the AOI
    on some paths."""
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[COLLECTION], bbox=bbox_wgs84(), datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    aoi = geometry_wgs84()
    items = [it for it in search.items() if shape(it.geometry).contains(aoi)]
    if not items:
        raise RuntimeError(f"No Landsat scene fully covering the AOI in {datetime_range} under {max_cloud}% cloud")
    return min(items, key=lambda it: it.properties["eo:cloud_cover"])


def fetch_date(datetime_range: str, label: str, max_cloud: float = 30.0) -> Path:
    """Fetch red/green/nir08/qa_pixel for the least-cloudy fully-covering
    scene in a date range. Writes data/raw/{label}.tif (4-band stack,
    native ~30m grid, DN values -- preprocess_landsat.py applies the
    reflectance scale and derives NDVI + NDWI) and returns its path."""
    item = _best_covering_item(datetime_range, max_cloud)
    platform = item.properties.get("platform", "?")
    print(f"[{label}] date={item.properties['datetime'][:10]} platform={platform} "
          f"cloud={item.properties['eo:cloud_cover']:.1f}%")

    aoi_geom = geometry_wgs84()
    arrays, out_profile = [], None
    for band in BANDS:
        with rasterio.open(item.assets[band].href) as ds:
            aoi_in_crs = gpd.GeoSeries([aoi_geom], crs="EPSG:4326").to_crs(ds.crs)
            clipped, clip_transform = rio_mask(ds, aoi_in_crs.geometry, crop=True, nodata=0)
            profile = ds.profile.copy()
            profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=clip_transform, count=1)
        arrays.append(clipped[0])
        out_profile = profile

    stack = np.stack(arrays)
    out_profile.update(count=len(BANDS), dtype=stack.dtype)
    out_path = RAW_DIR / f"{label}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(stack)
        dst.descriptions = tuple(BANDS)
        dst.update_tags(platform=platform, scale="0.0000275", offset="-0.2")
    print(f"  saved {out_path}  shape={stack.shape}  crs={out_profile['crs']}")
    return out_path


if __name__ == "__main__":
    fetch_date("2013-07-15/2013-09-15", "summer_2013_landsat", max_cloud=30)
