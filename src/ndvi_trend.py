"""
20+ year August NDVI + NDWI trend for the Berg en Dal AOI, 2005-present --
as far back as clean-enough coverage goes and as current as the most
recent finished growing season, combining two real sensors rather than
stretching one past where it's reliable:

  - 2005-2017: Landsat 5/7/8 (30m) via fetch_landsat.py + preprocess_landsat.py
  - 2018-present: Sentinel-2 L2A (10m) via fetch_sentinel2.py + preprocess.py

Both write to the same `optical_summer_{year}` stats key (mean_ndvi,
clear_pct, plus a `source`/`platform` tag), so this module just fetches
whichever years are missing with the right tool and reads the series back
in one pass. `LAST_YEAR` is computed from today's date, not hardcoded, so a
future run of this script picks up the newest finished August on its own.

    python ndvi_trend.py

Years before 2005 aren't attempted: Landsat 5/7 coverage and cloud-free
availability over this specific AOI thins out further back, and Landsat 7's
SLC-off gaps (since 2003) already add real missing-stripe noise inside the
window used here.
"""
import datetime as dt
from pathlib import Path

from fetch_landsat import fetch_date as fetch_landsat_date
from fetch_sentinel2 import fetch_date as fetch_sentinel2_date
from preprocess import process as process_sentinel2
from preprocess_landsat import process as process_landsat
from statsutil import load_stats, update_stats

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FIRST_YEAR = 2005
LANDSAT_LAST_YEAR = 2017  # 2018+ uses Sentinel-2 -- see module docstring

# This year's own August may not have happened yet (or may still be
# cloud-only) when this runs -- cap at last year in that case rather than
# raising, so a January run doesn't fail on next-August data that can't exist.
_today = dt.date.today()
LAST_YEAR = _today.year if _today.month > 8 else _today.year - 1

YEARS = list(range(FIRST_YEAR, LAST_YEAR + 1))


def _fetch_and_process_year(year: int) -> None:
    if year <= LANDSAT_LAST_YEAR:
        label = f"summer_{year}_landsat"
        if not (RAW_DIR / f"{label}.tif").exists():
            # August only, same as the Sentinel-2 fetch below -- an earlier
            # version of this used a wider mid-July-to-mid-September window
            # for Landsat, and about half the years it picked landed in
            # September while every Sentinel-2 year was August. That alone
            # produced a misleading step in the trend right at the
            # Landsat/Sentinel-2 handover that looked like a sensor
            # calibration issue; a same-month cross-check (Landsat
            # 2017-08-25 vs Sentinel-2 2017-08-29, 4 days apart) actually
            # agreed closely (0.650 vs 0.640), confirming it was the
            # mismatched window, not the data. Some Landsat years only have
            # one cloud-free-ish August pass under a single-satellite
            # 16-day repeat, hence the relaxed 50% ceiling -- QA_PIXEL's
            # per-pixel Clear bit still excludes the actual cloud from the
            # mean regardless of that whole-scene estimate.
            fetch_landsat_date(f"{year}-08-01/{year}-08-31", label, max_cloud=50)
        process_landsat(label, stats_key=f"summer_{year}")
    else:
        label = f"summer_{year}"
        if not (RAW_DIR / f"{label}.tif").exists():
            fetch_sentinel2_date(f"{year}-08-01/{year}-08-31", label, max_cloud=30)
        process_sentinel2(label)


def fetch_missing_years() -> None:
    for year in YEARS:
        _fetch_and_process_year(year)


def build_trend() -> list[dict]:
    fetch_missing_years()
    stats = load_stats()

    series = []
    for year in YEARS:
        opt = stats.get(f"optical_summer_{year}")
        if opt:
            series.append({
                "year": year, "mean_ndvi": opt["mean_ndvi"], "mean_ndwi": opt.get("mean_ndwi"),
                "clear_pct": opt["clear_pct"],
                "source": opt.get("source", "?"), "platform": opt.get("platform", "?"),
            })

    if len(series) < 2:
        raise RuntimeError(f"only {len(series)} year(s) of NDVI available -- can't build a trend")

    ndvis = [p["mean_ndvi"] for p in series]
    n_years = series[-1]["year"] - series[0]["year"]
    slope_per_year = (ndvis[-1] - ndvis[0]) / n_years if n_years else 0.0

    print(f"NDVI trend {series[0]['year']}-{series[-1]['year']} ({len(series)} years, "
          f"Landsat through {LANDSAT_LAST_YEAR}, Sentinel-2 from {LANDSAT_LAST_YEAR + 1}):")
    print("  " + "  ".join(f"{p['year']}={p['mean_ndvi']:.3f}" for p in series))
    print(f"  net change {ndvis[0]:+.3f} -> {ndvis[-1]:+.3f} over {n_years} years "
          f"({slope_per_year:+.4f}/yr)")

    update_stats("ndvi_trend", {
        "years": [p["year"] for p in series],
        "mean_ndvi": [p["mean_ndvi"] for p in series],
        "mean_ndwi": [p["mean_ndwi"] for p in series],
        "clear_pct": [p["clear_pct"] for p in series],
        "source": [p["source"] for p in series],
        "platform": [p["platform"] for p in series],
        "landsat_last_year": LANDSAT_LAST_YEAR,
        "slope_per_year": round(slope_per_year, 4),
        "net_change": round(ndvis[-1] - ndvis[0], 3),
        "updated": _today.isoformat(),
    })
    return series


if __name__ == "__main__":
    build_trend()
