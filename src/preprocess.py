"""
Turns a raw 5-band Sentinel-2 stack (B02,B03,B04,B08,SCL) into analysis-ready
reflectance + vegetation/water indices, with cloud/shadow pixels masked out.

Sentinel-2 L2A since processing baseline 04.00 (25 Jan 2022) stores digital
numbers with a +1000 additive offset so reflectance never goes negative;
this reverses that before computing anything, or NDVI/NDWI ratios would be
subtly wrong (the offset only cancels in a ratio's numerator, not its
denominator). Scenes from *before* that baseline change (anything fetched
for a year earlier than 2022) never had the offset applied in the first
place -- subtracting it anyway silently deflates the reflectance denominator
and inflates NDVI. This was a real bug here: the multi-year trend first
came out of the pipeline with 2018-2021 sitting at ~0.81-0.90 NDVI against
~0.59-0.69 for 2022-2025, a jump that landed exactly on the baseline-04.00
boundary rather than anywhere a real growing season would produce it. Fixed
by reading each scene's actual baseline (stamped into the GeoTIFF by
fetch_sentinel2.py) and only applying the offset when it's >= 04.00.
"""
from pathlib import Path

import numpy as np
import rasterio

from statsutil import update_stats

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

BOA_OFFSET = -1000  # ESA baseline >= 04.00: true reflectance = (DN + offset) / 10000
OFFSET_BASELINE_CUTOFF = 4.0
SCALE = 10000.0

# Scene Classification Layer codes we treat as clear ground truth.
# Excluded: 0 no-data, 1 saturated, 3 cloud shadow, 8/9 cloud, 10 cirrus.
CLEAR_SCL_CODES = {2, 4, 5, 6, 7, 11}


def _to_reflectance(dn: np.ndarray, offset: float) -> np.ndarray:
    refl = (dn.astype(np.float32) + offset) / SCALE
    return np.clip(refl, 0.0, 1.0)


def process(label: str) -> Path:
    """Reads data/raw/{label}.tif, writes data/processed/{label}_indices.tif.

    Output bands: NDVI, NDWI, B04(red,refl), B08(nir,refl), B03(green,refl),
    B02(blue,refl), valid(1=clear ground, 0=cloud/shadow/nodata).
    """
    src_path = RAW_DIR / f"{label}.tif"
    with rasterio.open(src_path) as src:
        blue, green, red, nir, scl = [src.read(i) for i in range(1, 6)]
        profile = src.profile.copy()
        baseline_tag = src.tags().get("s2_processing_baseline", "")

    try:
        baseline = float(baseline_tag)
    except ValueError:
        baseline = None
    offset = BOA_OFFSET if (baseline is not None and baseline >= OFFSET_BASELINE_CUTOFF) else 0.0
    if baseline is None:
        print(f"[{label}] WARNING: no processing-baseline tag on this raster (fetched by an older "
              f"version of fetch_sentinel2.py) -- assuming baseline >= 04.00 and applying the offset. "
              f"Refetch this label if it predates Jan 2022.")
        offset = BOA_OFFSET

    valid = np.isin(scl, list(CLEAR_SCL_CODES)).astype(np.uint8)

    red_r, nir_r, green_r, blue_r = (_to_reflectance(b, offset) for b in (red, nir, green, blue))

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir_r - red_r) / (nir_r + red_r)
        ndwi = (green_r - nir_r) / (green_r + nir_r)
    ndvi = np.nan_to_num(ndvi, nan=0.0)
    ndwi = np.nan_to_num(ndwi, nan=0.0)

    stack = np.stack([ndvi, ndwi, red_r, nir_r, green_r, blue_r, valid.astype(np.float32)])
    out_profile = profile.copy()
    out_profile.update(count=stack.shape[0], dtype="float32")
    out_path = PROC_DIR / f"{label}_indices.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(stack)
        dst.descriptions = ("NDVI", "NDWI", "red", "nir", "green", "blue", "valid")

    clear_frac = valid.mean() * 100
    mean_ndvi = float(ndvi[valid == 1].mean()) if valid.any() else float("nan")
    mean_ndwi = float(ndwi[valid == 1].mean()) if valid.any() else float("nan")
    print(f"[{label}] baseline={baseline_tag or '?'} offset={offset:.0f}  {valid.size:,} px  "
          f"clear={clear_frac:.1f}%  mean NDVI={mean_ndvi:.3f}  mean NDWI={mean_ndwi:.3f}")
    update_stats(f"optical_{label}", {
        "total_px": int(valid.size), "clear_pct": round(float(clear_frac), 1), "mean_ndvi": round(mean_ndvi, 3),
        "mean_ndwi": round(mean_ndwi, 3),
        "source": "sentinel2", "platform": f"baseline {baseline_tag}" if baseline_tag else "sentinel-2",
    })
    return out_path


if __name__ == "__main__":
    process("summer_2025")
    process("summer_2024")
