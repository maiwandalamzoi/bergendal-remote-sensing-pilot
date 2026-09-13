"""
Converts raw Sentinel-1 RTC backscatter (linear gamma0) to dB, derives a
simple open-water mask, and cross-checks that mask against the optical
KMeans land cover's own "Water" cluster -- two independent sensors, two
independent methods, same water bodies or not.

Water threshold: smooth open water is a near-specular reflector at C-band,
so it returns very little energy to the sensor -- open water on this scene
sits below roughly -17 dB VV (see the 5th percentile print in fetch), a
threshold consistent with the ~5-6% water fraction Sentinel-2's NDWI/KMeans
pass already found independently.
"""
from pathlib import Path

import numpy as np
import rasterio

from landcover_ml import _load_indices  # reuse the optical land-cover reader
from statsutil import update_stats

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

VV_WATER_THRESHOLD_DB = -17.0


def process(label: str = "sar_2025") -> Path:
    with rasterio.open(RAW_DIR / f"{label}.tif") as src:
        vv, vh = src.read(1), src.read(2)
        profile = src.profile.copy()

    valid = vv > 0
    with np.errstate(divide="ignore"):
        vv_db = np.where(valid, 10 * np.log10(np.where(valid, vv, 1)), np.nan)
        vh_db = np.where(valid, 10 * np.log10(np.where(valid, vh, 1)), np.nan)

    water = (valid & (vv_db < VV_WATER_THRESHOLD_DB)).astype(np.float32)

    stack = np.stack([vv_db, vh_db, water, valid.astype(np.float32)])
    out_profile = profile.copy()
    out_profile.update(count=4, dtype="float32")
    out_path = PROC_DIR / f"{label}_indices.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(stack)
        dst.descriptions = ("VV_dB", "VH_dB", "water_mask", "valid")

    water_pct = float(water[valid.astype(bool)].mean() * 100)
    print(f"[{label}] {valid.sum():,} valid px, SAR water = {water_pct:.1f}% of AOI clear ground")
    update_stats("sar", {"label": label, "water_pct": round(water_pct, 1), "threshold_db": VV_WATER_THRESHOLD_DB})
    return out_path


def _reproject_to(src_arr, src_profile, dst_profile, resampling):
    from rasterio.warp import reproject

    dst_arr = np.zeros((dst_profile["height"], dst_profile["width"]), dtype=np.float32)
    reproject(
        source=src_arr.astype(np.float32), destination=dst_arr,
        src_transform=src_profile["transform"], src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
        resampling=resampling,
    )
    return dst_arr


def flood_extent(highwater_label: str, baseline_label: str = "sar_2025") -> Path:
    """Where did the high-water scene show open water that the ordinary-flow
    baseline didn't? That "newly flooded" class is the actual flood-extent
    product the concept note pitched -- everything else so far has been a
    single-date snapshot, not a before/after.

    Output raster (on the baseline's grid): 0 = dry both times, 1 = water
    both times (permanent water body), 2 = newly flooded (dry at baseline,
    wet during high water).
    """
    from rasterio.enums import Resampling

    with rasterio.open(PROC_DIR / f"{baseline_label}_indices.tif") as base_src:
        base_water = base_src.read(3)
        base_valid = base_src.read(4) > 0
        base_profile = base_src.profile

    with rasterio.open(PROC_DIR / f"{highwater_label}_indices.tif") as hw_src:
        hw_water = hw_src.read(3)
        hw_valid = hw_src.read(4) > 0
        hw_profile = hw_src.profile

    hw_water_on_base = _reproject_to(hw_water, hw_profile, base_profile, Resampling.nearest) > 0
    hw_valid_on_base = _reproject_to(hw_valid.astype(np.float32), hw_profile, base_profile, Resampling.nearest) > 0

    both_valid = base_valid & hw_valid_on_base
    base_w = base_water.astype(bool)

    # 0 dry->dry, 1 wet->wet (permanent water), 2 dry->wet (newly flooded),
    # 3 wet->dry (baseline water the high-water scene missed -- almost
    # certainly threshold/geometry noise, not real drying during a flood).
    flood_class = np.zeros(base_valid.shape, dtype=np.uint8)
    flood_class[both_valid & base_w & hw_water_on_base] = 1
    flood_class[both_valid & ~base_w & hw_water_on_base] = 2
    flood_class[both_valid & base_w & ~hw_water_on_base] = 3
    flood_class[~both_valid] = 255

    out_profile = base_profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=255)
    out_path = PROC_DIR / f"flood_extent_{highwater_label}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(flood_class[np.newaxis, :, :])

    n_valid = both_valid.sum()
    baseline_pct = float((base_w[both_valid]).mean() * 100)
    highwater_pct = float((hw_water_on_base[both_valid]).mean() * 100)
    newly_flooded_pct = float((flood_class[both_valid] == 2).mean() * 100)
    newly_dry_pct = float((flood_class[both_valid] == 3).mean() * 100)

    print(f"flood extent ({baseline_label} -> {highwater_label}) on {n_valid:,} shared px:")
    print(f"  baseline water = {baseline_pct:.1f}%  |  high-water scene water = {highwater_pct:.1f}%")
    print(f"  newly inundated (dry -> wet)  = {newly_flooded_pct:.1f}%")
    print(f"  baseline water missed (wet -> dry) = {newly_dry_pct:.1f}%")
    if newly_dry_pct > newly_flooded_pct:
        print("  NOTE: more 'dried' than 'flooded' pixels despite this being a documented high-water date --")
        print("  the fixed -17dB threshold doesn't transfer cleanly across scenes shot from different orbit")
        print("  directions/conditions. A proper flood product needs change detection against a multi-date")
        print("  reference composite, not one fixed absolute threshold. Treat this pass as a pipeline proof,")
        print("  not a validated flood map.")
    update_stats("flood_extent", {
        "baseline": baseline_label, "high_water": highwater_label,
        "baseline_water_pct": round(baseline_pct, 1),
        "high_water_water_pct": round(highwater_pct, 1),
        "newly_flooded_pct": round(newly_flooded_pct, 1),
        "newly_dry_pct": round(newly_dry_pct, 1),
        "threshold_reliable": newly_flooded_pct >= newly_dry_pct,
    })
    return out_path


def cross_check_water(sar_label: str = "sar_2025", optical_label: str = "summer_2025") -> None:
    """How much do the SAR water mask and the optical KMeans "Water" cluster
    agree, on the ground they both actually cover?

    The two rasters sit on different grids (SAR's own clipped extent isn't
    pixel-identical to the optical mosaic's), so this resamples the SAR mask
    onto the optical grid with nearest-neighbour before comparing -- exact
    enough for an area-agreement check, not for anything sub-pixel.
    """
    from rasterio.warp import reproject, Resampling

    with rasterio.open(PROC_DIR / f"{sar_label}_indices.tif") as sar_src:
        sar_water = sar_src.read(3)
        sar_valid = sar_src.read(4) > 0
        sar_profile = sar_src.profile

    opt_data, opt_profile = _load_indices(optical_label)
    opt_valid = opt_data["valid"] > 0

    with rasterio.open(PROC_DIR / f"{optical_label}_landcover.tif") as lc_src:
        landcover = lc_src.read(1)
    legend = {}
    for line in (PROC_DIR / f"{optical_label}_landcover_legend.txt").read_text().strip().splitlines():
        cid, name = line.split("\t")
        legend[int(cid)] = name
    water_ids = [cid for cid, name in legend.items() if name == "Water"]
    opt_water = np.isin(landcover, water_ids)

    sar_water_on_opt_grid = np.zeros(opt_valid.shape, dtype=np.float32)
    reproject(
        source=sar_water, destination=sar_water_on_opt_grid,
        src_transform=sar_profile["transform"], src_crs=sar_profile["crs"],
        dst_transform=opt_profile["transform"], dst_crs=opt_profile["crs"],
        resampling=Resampling.nearest,
    )
    sar_valid_on_opt_grid = np.zeros(opt_valid.shape, dtype=np.float32)
    reproject(
        source=sar_valid.astype(np.float32), destination=sar_valid_on_opt_grid,
        src_transform=sar_profile["transform"], src_crs=sar_profile["crs"],
        dst_transform=opt_profile["transform"], dst_crs=opt_profile["crs"],
        resampling=Resampling.nearest,
    )

    both_valid = opt_valid & (sar_valid_on_opt_grid > 0)
    sar_w = sar_water_on_opt_grid[both_valid] > 0
    opt_w = opt_water[both_valid]

    agree_water = (sar_w & opt_w).sum()
    union_water = (sar_w | opt_w).sum()
    iou = agree_water / union_water if union_water else float("nan")

    print(f"cross-check on {both_valid.sum():,} shared clear px:")
    print(f"  SAR water = {sar_w.mean()*100:.1f}% | optical KMeans water = {opt_w.mean()*100:.1f}%")
    print(f"  agreement (IoU) = {iou*100:.1f}%")
    update_stats("sar_optical_cross_check", {
        "shared_px": int(both_valid.sum()),
        "sar_water_pct": round(float(sar_w.mean() * 100), 1),
        "optical_water_pct": round(float(opt_w.mean() * 100), 1),
        "iou_pct": round(float(iou * 100), 1),
    })


if __name__ == "__main__":
    process("sar_2025")
    cross_check_water()
