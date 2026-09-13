"""
Pulls real elevation data for Berg en Dal from AHN (Actueel Hoogtebestand
Nederland) -- the Dutch national LiDAR archive -- via PDOK's public WCS.
No key required. Native resolution is 0.5 m; this requests a coarser grid
directly from the service (the `scalesize` WCS parameter) rather than
downloading full-resolution and resampling locally.

Two coverages:
  dtm_05m -- Digital Terrain Model: bare-earth height (buildings/trees
             stripped out) -- the real version of the concept note's
             hand-drawn elevation profile.
  dsm_05m -- Digital Surface Model: first-return height, i.e. whatever the
             laser hit first (treetop, roofline, or ground).

DSM - DTM = a normalized surface model: canopy height in the forest, roughly
building height in the villages, ~0 over open ground. That's the "AHN
canopy / building height" data source the concept note pointed at.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.mask import mask as rio_mask

from statsutil import update_stats

RD_BOUNDARY_PATH = Path(__file__).resolve().parent.parent / "data" / "bergendal_boundary_rd.geojson"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

WCS_URL = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
RESOLUTION_M = 5  # output grid spacing; native archive is 0.5m


def _rd_bounds_padded(pad_m: float = 250):
    gdf = gpd.read_file(RD_BOUNDARY_PATH)  # already EPSG:28992 (RD New)
    minx, miny, maxx, maxy = gdf.total_bounds
    return minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m, gdf


def _fetch_coverage(coverage_id: str) -> Path:
    minx, miny, maxx, maxy, gdf = _rd_bounds_padded()
    width = int((maxx - minx) / RESOLUTION_M)
    height = int((maxy - miny) / RESOLUTION_M)

    params = {
        "service": "WCS", "version": "2.0.1", "request": "GetCoverage",
        "coverageId": coverage_id, "format": "image/tiff",
        "subset": [f"x({minx},{maxx})", f"y({miny},{maxy})"],
        "scalesize": f"x({width}),y({height})",
    }
    r = requests.get(WCS_URL, params=params, timeout=120)
    r.raise_for_status()

    raw_path = RAW_DIR / f"{coverage_id}_raw.tif"
    raw_path.write_bytes(r.content)

    # Clip to the exact municipal polygon (already in the WCS's native CRS)
    with rasterio.open(raw_path) as src:
        clipped, transform = rio_mask(src, gdf.geometry, crop=True, nodata=src.nodata)
        profile = src.profile.copy()
        profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=transform)

    out_path = RAW_DIR / f"{coverage_id}.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clipped)
    raw_path.unlink()

    valid = clipped[0][clipped[0] != profile["nodata"]]
    print(f"{coverage_id}: {out_path.name}  {clipped.shape[1]}x{clipped.shape[2]} @ {RESOLUTION_M}m  "
          f"elevation {valid.min():.1f}-{valid.max():.1f} m NAP")
    return out_path


def fetch_all():
    dtm_path = _fetch_coverage("dtm_05m")
    dsm_path = _fetch_coverage("dsm_05m")

    with rasterio.open(dtm_path) as d:
        dtm, profile, nodata = d.read(1), d.profile.copy(), d.nodata
    with rasterio.open(dsm_path) as s:
        dsm = s.read(1)

    invalid = (dtm == nodata) | (dsm == nodata)
    ndsm = np.clip(dsm - dtm, 0, None)
    ndsm[invalid] = nodata

    out_path = RAW_DIR / "ndsm_05m.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(ndsm[np.newaxis, :, :])

    valid_ndsm = ndsm[~invalid]
    valid_dtm = dtm[dtm != nodata]
    print(f"nDSM (canopy/building height): {out_path.name}  "
          f"mean={valid_ndsm.mean():.1f}m  >15m (mature canopy/tall buildings)={ (valid_ndsm>15).mean()*100:.1f}%")
    update_stats("ahn", {
        "resolution_m": RESOLUTION_M,
        "elevation_min_m": round(float(valid_dtm.min()), 1),
        "elevation_max_m": round(float(valid_dtm.max()), 1),
        "elevation_mean_m": round(float(valid_dtm.mean()), 1),
        "ndsm_mean_m": round(float(valid_ndsm.mean()), 1),
        "ndsm_over_15m_pct": round(float((valid_ndsm > 15).mean() * 100), 1),
    })


if __name__ == "__main__":
    fetch_all()
