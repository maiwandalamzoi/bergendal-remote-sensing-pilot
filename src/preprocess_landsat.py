"""
Turns a raw 4-band Landsat Collection 2 Level-2 stack (red, green, nir08,
qa_pixel) into the same NDVI/NDWI/clear-ground summary preprocess.py
produces for Sentinel-2 -- writes to the same `optical_{label}` stats
section, so ndvi_trend.py can read a Landsat year and a Sentinel-2 year
identically.

Reflectance scale is fixed across Collection 2 Level 2 regardless of
platform (Landsat 4-9): reflectance = DN * 0.0000275 - 0.2. Clear-ground
comes straight off the QA_PIXEL bit flags Landsat ships itself (bit 6 =
Clear, bit 0 = Fill) rather than a separate classification layer.
"""
from pathlib import Path

import numpy as np
import rasterio

from statsutil import update_stats

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

SCALE = 0.0000275
OFFSET = -0.2
QA_FILL_BIT = 1 << 0
QA_CLEAR_BIT = 1 << 6


def _to_reflectance(dn: np.ndarray) -> np.ndarray:
    refl = dn.astype(np.float32) * SCALE + OFFSET
    return np.clip(refl, 0.0, 1.0)


def process(label: str, stats_key: str | None = None) -> Path:
    """Reads data/raw/{label}.tif, writes data/processed/{label}_indices.tif
    (NDVI, NDWI, valid) and an `optical_{stats_key or label}` stats entry.

    `stats_key` lets a Landsat fetch stored under e.g. "summer_2013_landsat"
    (kept distinct on disk so it's never confused with a Sentinel-2 fetch
    for the same year) land in stats.json as "optical_summer_2013" -- the
    same key ndvi_trend.py looks up for every year regardless of source.
    """
    stats_key = stats_key or label
    src_path = RAW_DIR / f"{label}.tif"
    with rasterio.open(src_path) as src:
        red, green, nir, qa = src.read(1), src.read(2), src.read(3), src.read(4)
        profile = src.profile.copy()
        platform = src.tags().get("platform", "?")

    valid = (((qa & QA_CLEAR_BIT) != 0) & ((qa & QA_FILL_BIT) == 0)).astype(np.uint8)

    red_r, green_r, nir_r = _to_reflectance(red), _to_reflectance(green), _to_reflectance(nir)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir_r - red_r) / (nir_r + red_r)
        ndwi = (green_r - nir_r) / (green_r + nir_r)
    ndvi = np.nan_to_num(ndvi, nan=0.0)
    ndwi = np.nan_to_num(ndwi, nan=0.0)

    stack = np.stack([ndvi, ndwi, valid.astype(np.float32)])
    out_profile = profile.copy()
    out_profile.update(count=3, dtype="float32")
    out_path = PROC_DIR / f"{label}_indices.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(stack)
        dst.descriptions = ("NDVI", "NDWI", "valid")

    clear_frac = float(valid.mean() * 100)
    mean_ndvi = float(ndvi[valid == 1].mean()) if valid.any() else float("nan")
    mean_ndwi = float(ndwi[valid == 1].mean()) if valid.any() else float("nan")
    print(f"[{label}] platform={platform} (30m, Landsat)  {valid.size:,} px  "
          f"clear={clear_frac:.1f}%  mean NDVI={mean_ndvi:.3f}  mean NDWI={mean_ndwi:.3f}")
    update_stats(f"optical_{stats_key}", {
        "total_px": int(valid.size), "clear_pct": round(clear_frac, 1),
        "mean_ndvi": round(mean_ndvi, 3), "mean_ndwi": round(mean_ndwi, 3),
        "source": "landsat", "platform": platform,
    })
    return out_path


if __name__ == "__main__":
    process("summer_2013_landsat", stats_key="summer_2013")
