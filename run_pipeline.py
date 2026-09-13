"""
End-to-end pilot pipeline for Berg en Dal: fetch real Sentinel-1/2 and
Landsat imagery, AHN LiDAR, BRP field boundaries, CBS municipality
statistics, RIVM air quality and KNMI daily weather, run the unsupervised
ML + correlation steps, and render the outputs. See src/ for each stage;
this just runs them in order with one command.

    python run_pipeline.py

Outputs land in outputs/ (PNGs per layer + the interactive
bergendal_map.html) and data/processed/stats.json (every stage's summary
numbers, read directly by dashboard.py -- `streamlit run dashboard.py`).

Every fetch step is idempotent (skips a date/label already on disk in
data/raw/), so re-running this after the first pass is fast except for
whatever's genuinely new. The heaviest steps -- ndvi_trend (13 extra
Landsat years), flood_event (9 Sentinel-1 scenes), and fetch_weather (one
20-year daily pull from KNMI) -- can also be run standalone
(`python src/ndvi_trend.py`, `python src/flood_event.py`,
`python src/fetch_weather.py`) without repeating everything else.

`fetch_brp_history.py` (5 more years of BRP, 2020-2024, read one at a time
off PDOK's nationwide GeoPackages via HTTP range requests -- ~3 min/year)
is even heavier and deliberately NOT run by default here: past years'
registrations don't change, so it's a one-time pull, not something that
needs to stay current on every pass. Run it yourself once
(`python src/fetch_brp_history.py`) before the first `crop_rotation.py`
run; after that this script's own `crop_rotation.py` step re-runs fine
since it only reads what's already on disk.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fetch_sentinel2 import fetch_date
from fetch_sentinel1 import fetch_date as fetch_sar_date, find_covering_date
from fetch_ahn import fetch_all as fetch_ahn_all
from fetch_brp import fetch_all as fetch_brp_all
from brp_zonal import compute as compute_brp_zonal_ndvi
from fetch_cbs import fetch as fetch_cbs
from fetch_air_quality import fetch_all as fetch_air_quality_all
from preprocess import process
from preprocess_sar import process as process_sar, cross_check_water, flood_extent
from landcover_ml import classify, ndvi_change
from crop_rotation import run as run_crop_rotation
from fetch_brp_history import HISTORY_YEARS as ROTATION_HISTORY_YEARS
from ndvi_trend import build_trend as build_ndvi_trend
from flood_event import run as run_flood_event
from fetch_weather import run as fetch_weather
from climate_correlation import run as run_climate_correlation
from visualize import build_map

DATES = [
    ("2025-08-01/2025-08-31", "summer_2025", 5),
    ("2024-08-01/2024-08-31", "summer_2024", 5),
]


def main():
    print("== 1a. fetch: Sentinel-2 L2A via Planetary Computer STAC ==")
    for date_range, label, max_cloud in DATES:
        fetch_date(date_range, label, max_cloud=max_cloud)

    print("\n== 1b. fetch: Sentinel-1 RTC (SAR) via Planetary Computer STAC ==")
    sar_date = find_covering_date("2025-08-11")
    fetch_sar_date(sar_date, "sar_2025")
    fetch_sar_date("2024-01-13", "sar_highwater_2024")  # documented Rhine/Waal high water, Jan 2024

    print("\n== 1c. fetch: AHN LiDAR (DTM/DSM) via PDOK WCS ==")
    fetch_ahn_all()

    print("\n== 1d. fetch: BRP field boundaries & crops via PDOK WFS ==")
    fetch_brp_all()

    print("\n== 1e. fetch: CBS municipality statistics via PDOK WFS ==")
    fetch_cbs()

    print("\n== 1f. fetch: RIVM air quality (NO2/PM10/PM2.5) via WCS ==")
    fetch_air_quality_all()

    print("\n== 2. preprocess: cloud mask + NDVI/NDWI, SAR dB + water mask ==")
    for _, label, _ in DATES:
        process(label)
    process_sar("sar_2025")
    process_sar("sar_highwater_2024")

    print("\n== 3. ml: unsupervised land cover (KMeans), NDVI change, SAR/optical cross-check, flood extent ==")
    classify("summer_2025")
    ndvi_change("summer_2025", "summer_2024")
    cross_check_water("sar_2025", "summer_2025")
    flood_extent("sar_highwater_2024", "sar_2025")
    compute_brp_zonal_ndvi()  # per-field NDVI for the dashboard's Field Explorer tab

    raw_dir = Path(__file__).resolve().parent / "data" / "raw"
    if all((raw_dir / f"brp_parcels_{y}.geojson").exists() for y in ROTATION_HISTORY_YEARS):
        print("\n== 3a. crop rotation: match 2025 fields back to 2020-2024 (historical BRP already on disk) ==")
        run_crop_rotation()
    else:
        print("\n== 3a. crop rotation: SKIPPED -- run `python src/fetch_brp_history.py` once first "
              "(one-time, ~15min pull of 5 historical BRP years) ==")

    print("\n== 3b. ndvi trend: 2005-present August NDVI+NDWI series (fetches missing years) ==")
    build_ndvi_trend()

    print("\n== 3c. flood event: multi-date SAR change-detection through the Jan 2024 high water ==")
    print("(fetches 4 reference + 5 event-window scenes -- adds several minutes)")
    run_flood_event()

    print("\n== 3d. weather: KNMI daily records 2005-present (Deelen Airport station) ==")
    fetch_weather()

    print("\n== 3e. climate correlation: NDVI/NDWI trend vs August weather ==")
    run_climate_correlation()

    print("\n== 4. visualize: PNG maps + interactive layer-toggle map ==")
    map_path = build_map("summer_2025", "summer_2024")

    print(f"\nDone. Open outputs/{map_path.name} in a browser, or run "
          f"`streamlit run dashboard.py` for the dashboard view.")


if __name__ == "__main__":
    main()
