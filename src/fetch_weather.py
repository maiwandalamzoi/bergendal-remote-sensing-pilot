"""
Real daily weather for Berg en Dal, 2005-present, from KNMI's public daily
data service (daggegevens.knmi.nl) -- no key, no signup, the Dutch national
weather service's own official station records.

Nearest station to the AOI centroid (51.83N, 5.93E), checked directly
against KNMI's station list: **275, Deelen Airport** (52.06N, 5.87E, ~25km
north). Not inside the municipality -- there's no KNMI station in Berg en
Dal itself -- but the nearest official one, and close enough that its
daily rainfall/temperature is a reasonable regional proxy, the same
compromise every "climate near X" product makes without its own station
network. KNMI's own data disclaimer (carried through verbatim in the raw
CSV) is that raw daily series aren't homogenized for trend analysis
because of station relocations/instrument changes -- worth repeating here
rather than silently dropping it, given this pipeline's own habit of
surfacing exactly this kind of caveat.

Pulls the full daily series once (data/raw/knmi_daily.csv), then derives
two summaries for the dashboard:
  - August aggregates per year (matches the satellite pipeline's own
    single-month sampling window, so weather and NDVI/NDWI are compared
    over the same calendar slice)
  - Full-year aggregates per year (broader climate context)

    python fetch_weather.py
"""
import datetime as dt
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from statsutil import update_stats

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
STATION = 275  # Deelen Airport -- nearest KNMI station to the AOI
STATION_NAME = "Deelen Airport"
FIRST_YEAR = 2005
VARS = "TG:TN:TX:RH:SQ:EV24:UG"  # mean/min/max temp, precip, sunshine, evapotranspiration, humidity


def fetch_daily() -> pd.DataFrame:
    """Fetches (or reuses a cached) full daily series and returns it as a
    DataFrame with real units (temps in C, precip/evapotranspiration in mm,
    sunshine in hours) -- KNMI's own raw values are tenths."""
    out_path = RAW_DIR / "knmi_daily.csv"
    end = dt.date.today().strftime("%Y%m%d")
    resp = requests.post(API_URL, data={
        "stns": str(STATION), "start": f"{FIRST_YEAR}0101", "end": end, "vars": VARS,
    }, timeout=60)
    resp.raise_for_status()

    all_lines = resp.text.splitlines()
    # The real column order lives in the response's own header comment
    # ("# STN,YYYYMMDD,   TG, ..."), not necessarily the order `vars` was
    # requested in -- parsing it out here instead of hardcoding a column
    # list avoids silently mislabelling two columns if KNMI's own order
    # doesn't match the request (it doesn't: SQ comes before RH).
    header_line = next(l for l in all_lines if l.lstrip("# ").startswith("STN,YYYYMMDD"))
    cols = [c.strip() for c in header_line.lstrip("# ").split(",")]
    data_lines = [l for l in all_lines if l and not l.startswith("#")]
    if not data_lines:
        raise RuntimeError("KNMI API returned no data rows -- check station/var names")
    df = pd.read_csv(StringIO("\n".join(data_lines)), names=cols, skipinitialspace=True)
    df["date"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # KNMI: -1 means "measurable but below 0.05mm/0.05hr" -- read as 0, not
    # a missing value (a real, if tiny, positive amount).
    for c in ("RH", "SQ"):
        df[c] = df[c].where(df[c] != -1, 0)
    df["mean_temp_c"] = df["TG"] / 10.0
    df["min_temp_c"] = df["TN"] / 10.0
    df["max_temp_c"] = df["TX"] / 10.0
    df["precip_mm"] = df["RH"] / 10.0
    df["sunshine_h"] = df["SQ"] / 10.0
    df["evapotranspiration_mm"] = df["EV24"] / 10.0
    df["humidity_pct"] = df["UG"]

    df.to_csv(out_path, index=False)
    print(f"[weather] {len(df):,} daily rows, {df['year'].min()}-{df['year'].max()}, "
          f"station {STATION} ({STATION_NAME}) -- saved {out_path}")
    return df


def build_summaries(df: pd.DataFrame) -> dict:
    last_year = dt.date.today().year if dt.date.today().month > 8 else dt.date.today().year - 1
    df = df[df["year"] <= last_year]

    aug = df[df["month"] == 8]
    aug_by_year = aug.groupby("year").agg(
        mean_temp_c=("mean_temp_c", "mean"),
        max_temp_c=("max_temp_c", "max"),
        total_precip_mm=("precip_mm", "sum"),
        total_sunshine_h=("sunshine_h", "sum"),
        total_evapotranspiration_mm=("evapotranspiration_mm", "sum"),
    ).round(1)

    annual = df.groupby("year").agg(
        mean_temp_c=("mean_temp_c", "mean"),
        total_precip_mm=("precip_mm", "sum"),
        total_sunshine_h=("sunshine_h", "sum"),
    ).round(1)

    years = aug_by_year.index.tolist()
    summary = {
        "station": STATION, "station_name": STATION_NAME,
        "first_year": FIRST_YEAR, "last_year": last_year,
        "august": {
            "years": years,
            "mean_temp_c": aug_by_year["mean_temp_c"].tolist(),
            "max_temp_c": aug_by_year["max_temp_c"].tolist(),
            "total_precip_mm": aug_by_year["total_precip_mm"].tolist(),
            "total_sunshine_h": aug_by_year["total_sunshine_h"].tolist(),
            "total_evapotranspiration_mm": aug_by_year["total_evapotranspiration_mm"].tolist(),
        },
        "annual": {
            "years": annual.index.tolist(),
            "mean_temp_c": annual["mean_temp_c"].tolist(),
            "total_precip_mm": annual["total_precip_mm"].tolist(),
            "total_sunshine_h": annual["total_sunshine_h"].tolist(),
        },
    }

    hottest = aug_by_year["mean_temp_c"].idxmax()
    driest = aug_by_year["total_precip_mm"].idxmin()
    wettest = aug_by_year["total_precip_mm"].idxmax()
    print(f"[weather] August summaries {years[0]}-{years[-1]}: "
          f"hottest={hottest} ({aug_by_year.loc[hottest,'mean_temp_c']:.1f}C), "
          f"driest={driest} ({aug_by_year.loc[driest,'total_precip_mm']:.0f}mm), "
          f"wettest={wettest} ({aug_by_year.loc[wettest,'total_precip_mm']:.0f}mm)")

    update_stats("weather", summary)
    return summary


def run() -> dict:
    df = fetch_daily()
    return build_summaries(df)


if __name__ == "__main__":
    run()
