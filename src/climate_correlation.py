"""
Ties the satellite vegetation trend to the real weather record: does a
drier or hotter August actually line up with lower NDVI that same year,
across the 22-year series? Reads the `ndvi_trend` and `weather` stats
sections (both already on disk once ndvi_trend.py and fetch_weather.py
have run) and writes one more -- no new fetching here, just analysis.

Honest limits, stated once here rather than re-argued at every number:
  - n=22 years is a small sample for a correlation coefficient -- treat
    |r| < ~0.4 as noise, not "no relationship."
  - Each year's NDVI is ONE least-cloudy day in August; the weather side
    is that whole month's total/mean. A single good or bad week can move
    the satellite sample without moving the monthly weather aggregate
    much, and vice versa -- this compares a snapshot to an integral.
  - Correlation, not causation: temperature, rainfall and sunshine all
    move together across a real summer, so isolating "the" driver from
    three correlated weather variables isn't something 22 points can do.

    python climate_correlation.py
"""
import numpy as np

from statsutil import load_stats, update_stats


def _pearson(x: list[float], y: list[float]) -> float:
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run() -> dict:
    stats = load_stats()
    trend = stats.get("ndvi_trend")
    weather = stats.get("weather")
    if not trend or not weather:
        raise RuntimeError("run ndvi_trend.py and fetch_weather.py first -- both stats sections are required")

    trend_years = trend["years"]
    aug = weather["august"]
    aug_by_year = dict(zip(aug["years"], zip(
        aug["mean_temp_c"], aug["total_precip_mm"], aug["total_sunshine_h"], aug["total_evapotranspiration_mm"],
    )))

    years, ndvi, ndwi, temp, precip, sun, evap = [], [], [], [], [], [], []
    for i, y in enumerate(trend_years):
        if y not in aug_by_year:
            continue
        t, p, s, e = aug_by_year[y]
        years.append(y)
        ndvi.append(trend["mean_ndvi"][i])
        ndwi.append(trend["mean_ndwi"][i])
        temp.append(t)
        precip.append(p)
        sun.append(s)
        evap.append(e)

    pairs = {
        "ndvi_vs_august_precip": _pearson(ndvi, precip),
        "ndvi_vs_august_temp": _pearson(ndvi, temp),
        "ndvi_vs_august_sunshine": _pearson(ndvi, sun),
        "ndvi_vs_august_evapotranspiration": _pearson(ndvi, evap),
        "ndwi_vs_august_precip": _pearson(ndwi, precip),
        "ndwi_vs_august_temp": _pearson(ndwi, temp),
    }

    print(f"climate correlation, {len(years)} years ({years[0]}-{years[-1]}):")
    for k, v in pairs.items():
        print(f"  r({k}) = {v:+.2f}")

    result = {
        "n_years": len(years),
        "years": years,
        "correlations": {k: (round(v, 3) if v == v else None) for k, v in pairs.items()},
        "series": {
            "ndvi": ndvi, "ndwi": ndwi, "august_temp_c": temp,
            "august_precip_mm": precip, "august_sunshine_h": sun, "august_evapotranspiration_mm": evap,
        },
    }
    update_stats("climate_correlation", result)
    return result


if __name__ == "__main__":
    run()
