"""
Multi-date Sentinel-1 flood-event analysis for Berg en Dal -- the fix for
the two gaps preprocess_sar.flood_extent() flagged itself:

  1. "a fixed -17dB absolute threshold doesn't transfer cleanly between
     scenes shot in different conditions/orbits" -- replaced here with
     change detection: each pixel is compared against its *own* normal
     backscatter (a per-pixel median built from several ordinary-condition
     scenes), not one global cutoff.
  2. "only one high-water date fetched, five days past the documented
     peak" -- replaced here with a short series spanning the pre-event
     baseline, the rising limb, and three stages of recession.

Reference dates (REFERENCE_DATES) are four autumn-2023 passes, well before
the documented Jan 2024 Rhine/Waal high water, spaced ~12 days apart so no
single date's local wetness or speckle dominates the per-pixel median.
Event dates (EVENT_DATES) bracket the documented Lobith gauge peak
(14.5-14.7 m NAP, Jan 8-9 2024): no Sentinel-1 scene fully covering the
AOI exists on the peak day itself (checked directly against the STAC
catalogue), so the nearest fully-covering passes on either side stand in
for it. sar_event_20240113 is the same acquisition preprocess_sar.py
already knows as "sar_highwater_2024" -- fetched fresh here under its own
label so this module doesn't depend on that file existing.

    python flood_event.py
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from fetch_sentinel1 import fetch_date
from statsutil import update_stats

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_DATES = ["2023-11-02", "2023-11-14", "2023-11-26", "2023-12-08"]

EVENT_DATES = [
    ("2023-12-23", "pre-event"),
    ("2024-01-04", "rise"),
    ("2024-01-13", "recession (early)"),
    ("2024-01-16", "recession (mid)"),
    ("2024-01-25", "recession (late)"),
]

# A pixel reading more than this many dB darker than its own reference-
# composite level is flagged as flooded. -3dB is the change-detection
# threshold used in the SAR flood-mapping literature (e.g. Twele et al.
# 2016), not a value tuned against this AOI's own data.
CHANGE_THRESHOLD_DB = -3.0


def _ref_label(date_str: str) -> str:
    return f"sar_ref_{date_str.replace('-', '')}"


def _event_label(date_str: str) -> str:
    return f"sar_event_{date_str.replace('-', '')}"


def fetch_all() -> None:
    for d in REFERENCE_DATES:
        label = _ref_label(d)
        if not (RAW_DIR / f"{label}.tif").exists():
            fetch_date(d, label)
    for d, _phase in EVENT_DATES:
        label = _event_label(d)
        if not (RAW_DIR / f"{label}.tif").exists():
            fetch_date(d, label)


def _read_vv_db(label: str):
    with rasterio.open(RAW_DIR / f"{label}.tif") as src:
        vv = src.read(1)
        profile = src.profile.copy()
    valid = vv > 0
    with np.errstate(divide="ignore"):
        vv_db = np.where(valid, 10 * np.log10(np.where(valid, vv, 1)), np.nan)
    return vv_db, valid, profile


def _reproject_to(arr: np.ndarray, src_profile: dict, dst_profile: dict) -> np.ndarray:
    dst = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype=np.float32)
    reproject(
        source=arr.astype(np.float32), destination=dst,
        src_transform=src_profile["transform"], src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
        resampling=Resampling.bilinear,
    )
    return dst


def build_reference_composite():
    """Per-pixel median VV dB across REFERENCE_DATES, on the first
    reference date's grid (all four share the same CRS/resolution/AOI clip,
    so this is a near-identical grid already; reprojecting the other three
    onto it handles any sub-pixel offset between passes)."""
    labels = [_ref_label(d) for d in REFERENCE_DATES]
    base_db, base_valid, base_profile = _read_vv_db(labels[0])
    stack = [np.where(base_valid, base_db, np.nan)]
    for label in labels[1:]:
        db, valid, profile = _read_vv_db(label)
        stack.append(_reproject_to(np.where(valid, db, np.nan), profile, base_profile))
    stack = np.stack(stack)
    n_obs = np.sum(~np.isnan(stack), axis=0)

    # Most of this rectangular grid sits outside the irregular municipal
    # polygon (same reason a Sentinel-2 scene here reads ~47% "clear ground"
    # over its own bbox) -- median coverage over the *whole* array would
    # report 0/4 and look broken. Restrict the diagnostic to pixels at
    # least one reference date actually covers, and compute nanmedian only
    # where there's something to average (nanmedian on an all-NaN slice is
    # harmless here -- it's masked out right after -- but warns every time).
    in_aoi = n_obs > 0
    composite = np.full(in_aoi.shape, np.nan, dtype=np.float32)
    composite[in_aoi] = np.nanmedian(stack[:, in_aoi], axis=0)

    out_profile = base_profile.copy()
    out_profile.update(count=1, dtype="float32", nodata=np.nan)
    out_path = PROC_DIR / "sar_reference_composite_vv_db.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(composite[np.newaxis, :, :])

    print(f"reference composite: {len(labels)} dates ({', '.join(REFERENCE_DATES)}), "
          f"{in_aoi.mean()*100:.1f}% of grid inside the AOI, "
          f"median coverage there = {int(np.median(n_obs[in_aoi]))}/{len(labels)} scenes")
    return composite, base_profile


def classify_event_date(date_str: str, phase: str, composite: np.ndarray, composite_profile: dict) -> dict:
    """Change-detection flood flag for one event-window scene: how far
    below its own pixel's reference-composite level did it drop? A pixel
    that's already dark under normal conditions (e.g. permanent water,
    a shadowed slope) needs to drop *further*, not just read as dark in
    absolute terms -- which is exactly what the old fixed-threshold method
    couldn't tell apart from a pixel that only turned dark now."""
    label = _event_label(date_str)
    db, valid, profile = _read_vv_db(label)
    db_on_ref = _reproject_to(np.where(valid, db, np.nan), profile, composite_profile)

    delta = db_on_ref - composite
    both_valid = ~np.isnan(delta)
    flooded = both_valid & (delta < CHANGE_THRESHOLD_DB)

    flooded_pct = float(flooded[both_valid].mean() * 100) if both_valid.any() else float("nan")
    mean_delta = float(np.nanmean(delta[both_valid])) if both_valid.any() else float("nan")

    out = np.full(delta.shape, 255, dtype=np.uint8)
    out[both_valid] = flooded[both_valid].astype(np.uint8)
    out_profile = composite_profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=255)
    out_path = PROC_DIR / f"flood_changedetect_{_event_label(date_str)}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(out[np.newaxis, :, :])

    print(f"[{date_str}] ({phase}) {both_valid.sum():,} shared px, mean backscatter vs reference = "
          f"{mean_delta:+.1f} dB, flagged as flooded = {flooded_pct:.1f}% of shared ground")
    return {"date": date_str, "phase": phase, "flooded_pct": round(flooded_pct, 1),
            "mean_db_vs_reference": round(mean_delta, 2)}


def run() -> list[dict]:
    fetch_all()
    composite, composite_profile = build_reference_composite()

    timeline = [classify_event_date(d, phase, composite, composite_profile) for d, phase in EVENT_DATES]

    pre_event = timeline[0]
    peak = max(timeline, key=lambda r: r["flooded_pct"])
    net_change = round(peak["flooded_pct"] - pre_event["flooded_pct"], 1)

    print(f"\nchange-detection flood series, pre-event -> peak: "
          f"{pre_event['flooded_pct']:.1f}% -> {peak['flooded_pct']:.1f}% "
          f"(peak on {peak['date']}, {peak['phase']}), net {net_change:+.1f}%")

    update_stats("flood_event", {
        "method": "change_detection_vs_reference_composite",
        "reference_dates": REFERENCE_DATES,
        "change_threshold_db": CHANGE_THRESHOLD_DB,
        "timeline": timeline,
        "pre_event_date": pre_event["date"],
        "pre_event_flooded_pct": pre_event["flooded_pct"],
        "peak_date": peak["date"],
        "peak_phase": peak["phase"],
        "peak_flooded_pct": peak["flooded_pct"],
        "net_change_pct": net_change,
    })
    return timeline


if __name__ == "__main__":
    run()
