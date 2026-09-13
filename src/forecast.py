"""
Simple, honest forecasts built entirely on data this pipeline has already
fetched -- no new network calls, and no black box. Every series here is
short (5-9 real annual data points) and genuinely noisy, so the method
stays deliberately simple (ordinary least squares) and every number ships
with its own uncertainty, not a bare projection dressed up as a
certainty:

- **Vegetation health (NDVI/NDWI):** linear trend + a real OLS 80%
  prediction interval (widens with distance from the data's own mean
  year via the textbook formula, not a fixed +/-), projected 5 years past
  the last observed year.
- **Land cover** (built-up / forest / agriculture / water): the same
  linear-trend treatment per category, on landcover_trend.py's already
  common-footprint-corrected series, clipped to [0, that series' own
  footprint] since a straight line doesn't know hectares can't go
  negative or exceed the area it's measured over.
- **Population & housing stock:** the same, on cbs_trend.py's registry
  series.
- **Crop rotation:** not a trend line at all -- a first-order Markov
  chain over crop *families* (not the 102 raw crop names; not enough
  real transitions per exact crop to say anything about most of them),
  built from every field's own real multi-year BRP history
  (crop_rotation.py's rotation_json). transition_matrix[A][B] is the
  empirical probability that a field grown as family A one year is
  family B the next, among the real transitions this pipeline's own
  matched history actually contains -- not a hand-picked agronomic rule.
  Powers Field Explorer's "Predicted next crop family" colour mode
  (src/visualize.py) and a municipality-wide projected-area-by-family
  number below.

**The honest limitation, stated once here rather than five times below:**
every trend forecast is 5-9 points of real but noisy annual data,
extrapolated with a straight line -- read this as "if the recent trend
continues," never a guarantee, and read the interval, not the point
estimate, as the actual answer. The Trends & Climate tab's own NDVI chart
already says this about the *historical* series ("read the shape, not
the slope"); a forecast built on top of that series inherits the same
caveat, more so. **A real bug found and fixed while building this:**
visualize.py's classify_crop() matched "mais" as an ASCII substring, so
it silently missed every occurrence of BRP's historical-archive spelling
"Maïs" (diaeresis) -- every archive-year maize parcel would have fallen
into "other" instead of "maize" in the transition counts below. Fixed by
accent-normalizing (NFKD-decompose, drop combining marks) before matching,
the same technique crop_rotation.py's own normalize_crop() already uses
for a different reason.

    python forecast.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from statsutil import load_stats, update_stats
from visualize import classify_crop

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
BRP_PATH = RAW_DIR / "brp_parcels.geojson"

FORECAST_YEARS_AHEAD = 5


def _ols_forecast(years: list, values: list, years_ahead: int = FORECAST_YEARS_AHEAD) -> dict | None:
    """Ordinary least squares on (year, value), with a real 80% prediction
    interval (the textbook OLS formula -- widens away from the data's own
    mean year, not a fixed +/- band) for `years_ahead` years past the
    last observed one. Returns None with fewer than 3 real points -- not
    enough to say anything honest about a trend, let alone a forecast."""
    pairs = [(y, v) for y, v in zip(years, values) if v is not None]
    if len(pairs) < 3:
        return None
    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    n = len(xs)
    dof = n - 2

    model = LinearRegression().fit(xs.reshape(-1, 1), ys)
    slope, intercept = float(model.coef_[0]), float(model.intercept_)
    fitted = model.predict(xs.reshape(-1, 1))
    resid = ys - fitted
    rss = float(np.sum(resid ** 2))
    r2 = float(model.score(xs.reshape(-1, 1), ys))
    s = np.sqrt(rss / dof) if dof > 0 else None
    x_mean = float(xs.mean())
    sxx = float(np.sum((xs - x_mean) ** 2))

    future_years = [int(xs.max()) + i for i in range(1, years_ahead + 1)]
    future_x = np.array(future_years, dtype=float)
    future_pred = model.predict(future_x.reshape(-1, 1))

    if s is not None and sxx > 0:
        t80 = float(stats.t.ppf(0.90, dof))  # two-sided 80% CI -> 90th percentile of |t|
        se_pred = s * np.sqrt(1 + 1 / n + (future_x - x_mean) ** 2 / sxx)
        lower = (future_pred - t80 * se_pred).tolist()
        upper = (future_pred + t80 * se_pred).tolist()
    else:
        lower = upper = [None] * len(future_years)

    return {
        "history_years": [int(x) for x in xs.tolist()],
        "history_values": ys.tolist(),
        "future_years": future_years,
        "future_values": future_pred.tolist(),
        "future_lower80": lower,
        "future_upper80": upper,
        "slope_per_year": round(slope, 5),
        "intercept": intercept,
        "r_squared": round(r2, 3),
        "n_points": n,
    }


def forecast_vegetation() -> dict | None:
    trend = load_stats().get("ndvi_trend", {})
    if not trend.get("years"):
        return None
    out = {}
    ndvi_fc = _ols_forecast(trend["years"], trend.get("mean_ndvi", []))
    if ndvi_fc:
        out["ndvi"] = ndvi_fc
    ndwi_vals = trend.get("mean_ndwi")
    if ndwi_vals:
        ndwi_fc = _ols_forecast(trend["years"], ndwi_vals)
        if ndwi_fc:
            out["ndwi"] = ndwi_fc
    return out or None


def forecast_landcover() -> dict | None:
    lc = load_stats().get("landcover_trend", {})
    if not lc.get("years"):
        return None
    cap = lc.get("common_footprint_ha")

    def _clip(v):
        if v is None:
            return None
        v = max(0.0, v)
        return min(v, cap) if cap else v

    out = {}
    for cat, vals in lc.get("series_ha", {}).items():
        fc = _ols_forecast(lc["years"], vals)
        if not fc:
            continue
        fc["future_values"] = [_clip(v) for v in fc["future_values"]]
        fc["future_lower80"] = [_clip(v) for v in fc["future_lower80"]]
        fc["future_upper80"] = [_clip(v) for v in fc["future_upper80"]]
        out[cat] = fc
    return out or None


def forecast_population() -> dict | None:
    cbs = load_stats().get("cbs_trend", {})
    if not cbs.get("years"):
        return None
    out = {}
    pop_fc = _ols_forecast(cbs["years"], cbs.get("population", []))
    if pop_fc:
        out["population"] = pop_fc
    stock_fc = _ols_forecast(cbs["years"], cbs.get("housing_stock", []))
    if stock_fc:
        out["housing_stock"] = stock_fc
    return out or None


def forecast_crop_rotation() -> dict | None:
    """A first-order Markov chain over crop families, built from every
    field's own real multi-year BRP history -- see the module docstring.
    Also returns a municipality-wide expected-value area projection for
    next year (each field's area spread across families by its own
    family's transition probabilities), distinct from the single
    most-likely-family prediction Field Explorer's map mode computes
    per field on the fly from this same matrix."""
    if not BRP_PATH.exists():
        return None
    gdf = gpd.read_file(BRP_PATH)
    if "rotation_json" not in gdf.columns:
        return None

    counts: dict[str, dict[str, int]] = {}
    for _, row in gdf.iterrows():
        raw = row.get("rotation_json")
        if not raw or (isinstance(raw, float) and np.isnan(raw)):
            continue
        by_year = json.loads(raw) if isinstance(raw, str) else raw
        years_sorted = sorted((y for y, c in by_year.items() if c), key=int)
        for y0, y1 in zip(years_sorted, years_sorted[1:]):
            fam0 = classify_crop(by_year[y0], row["category"])
            fam1 = classify_crop(by_year[y1], row["category"])
            counts.setdefault(fam0, {}).setdefault(fam1, 0)
            counts[fam0][fam1] += 1

    if not counts:
        return None

    matrix = {
        fam0: {fam1: round(c / sum(row_counts.values()), 4)
               for fam1, c in sorted(row_counts.items(), key=lambda kv: -kv[1])}
        for fam0, row_counts in counts.items()
    }

    projected_area: dict[str, float] = {}
    n_predicted_fields = 0
    for _, row in gdf.iterrows():
        fam = classify_crop(row["gewas"], row["category"])
        area = row.get("area_ha") or 0.0
        row_probs = matrix.get(fam)
        if not row_probs:
            projected_area[fam] = projected_area.get(fam, 0.0) + area  # no observed transitions: assume unchanged
            continue
        n_predicted_fields += 1
        for fam1, p in row_probs.items():
            projected_area[fam1] = projected_area.get(fam1, 0.0) + area * p
    projected_area = {k: round(v, 1) for k, v in sorted(projected_area.items(), key=lambda kv: -kv[1])}

    n_transitions = sum(sum(r.values()) for r in counts.values())
    return {
        "transition_matrix": matrix,
        "transition_counts": counts,
        "n_transitions_observed": n_transitions,
        "n_fields_with_prediction": n_predicted_fields,
        "projected_area_ha_next_year": projected_area,
    }


def build() -> dict:
    result = {
        "years_ahead": FORECAST_YEARS_AHEAD,
        "vegetation": forecast_vegetation(),
        "landcover": forecast_landcover(),
        "population": forecast_population(),
        "crop_rotation": forecast_crop_rotation(),
    }

    veg = result["vegetation"] or {}
    if veg.get("ndvi"):
        f = veg["ndvi"]
        print(f"[forecast] NDVI: {f['n_points']} real years, r²={f['r_squared']:.2f}, "
              f"slope {f['slope_per_year']:+.4f}/yr -> {f['future_years'][0]}: "
              f"{f['future_values'][0]:.3f} [{f['future_lower80'][0]:.3f}, {f['future_upper80'][0]:.3f}] "
              f"(80% interval)")
    lc = result["landcover"] or {}
    for cat, f in lc.items():
        print(f"[forecast] {cat}: r²={f['r_squared']:.2f} -> {f['future_years'][-1]}: "
              f"{f['future_values'][-1]:,.0f} ha [{f['future_lower80'][-1]:,.0f}, {f['future_upper80'][-1]:,.0f}]")
    pop = result["population"] or {}
    if pop.get("population"):
        f = pop["population"]
        print(f"[forecast] Population: r²={f['r_squared']:.2f} -> {f['future_years'][-1]}: "
              f"{f['future_values'][-1]:,.0f} [{f['future_lower80'][-1]:,.0f}, {f['future_upper80'][-1]:,.0f}]")
    cr = result["crop_rotation"] or {}
    if cr.get("transition_matrix"):
        print(f"[forecast] Crop rotation: {cr['n_transitions_observed']:,} real transitions observed across "
              f"{len(cr['transition_matrix'])} families, {cr['n_fields_with_prediction']:,} fields predictable")
        top_proj = list(cr["projected_area_ha_next_year"].items())[:3]
        print(f"[forecast]   projected next-year area (top 3): " +
              ", ".join(f"{fam} {ha:,.0f}ha" for fam, ha in top_proj))

    update_stats("forecast", result)
    return result


if __name__ == "__main__":
    build()
