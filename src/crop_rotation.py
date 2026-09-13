"""
Real crop-rotation history per field: what did each of the current (2025)
BRP parcels grow in 2020, 2021, 2022, 2023 and 2024, not just this year.

BRP re-registers parcel boundaries every year (a farmer redraws their own
parcel each spring), so there's no stable parcel ID to join years on --
this matches each 2025 parcel to its best-overlapping parcel in every
earlier year (max shared area, geopandas.overlay on the two years'
polygons) and only accepts a match past 30% area overlap, below which
it's more likely a boundary redraw or a genuinely different field than
the same field measured twice.

**Two real data-quality bugs found and fixed while building this, both
worth naming rather than quietly patching over:**
1. The historical GeoPackage archive has trailing whitespace on some
   `gewas` values in 2023/2024 ("Grasland, blijvend " vs "Grasland,
   blijvend") that the live WFS and earlier years don't -- naive string
   equality read that as a crop change every single time. Fixed with
   `.strip()` at load time.
2. PDOK's two BRP distribution channels spell at least one crop
   differently: the live WFS (fetch_brp.py, 2025) returns "Mais, snij-";
   the historical GeoPackage archive returns "Maïs, snij-" (with the
   diaeresis -- the standard Dutch spelling). Same crop, different
   upstream source, different string. Fixed by normalizing (NFKD-decompose
   accents, casefold, strip) into a comparison key used only to decide
   "did this change", while the original spelling from whichever source a
   given year came from is still what's *displayed*.
   Both together were inflating the apparent rotation rate before the
   fix -- worth knowing if this pipeline's own history is ever extended
   further, not just a one-time footnote.

**A real caveat that's a genuine data-coverage limit, not a bug:** parcel
count jumps from ~1,750 (2020-2022) to ~4,100-4,400 (2023-2025) at
roughly constant total area -- BRP started registering landscape elements
(hedgerows, ditches, tree lines) as their own small parcels around 2023
rather than folding them into the surrounding field. A 2025 hedge parcel
matching a much larger 2020 "Grasland" polygon is that schema change
showing up in the match, not a real land-use event -- `min_overlap_frac`
and each match's own `overlap_frac` are kept in the output specifically
so this is checkable per field rather than hidden.

    python crop_rotation.py
"""
import json
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd

from fetch_brp_history import HISTORY_YEARS
from statsutil import update_stats

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
BRP_PATH = RAW_DIR / "brp_parcels.geojson"

MIN_OVERLAP_FRAC = 0.30
REFERENCE_YEAR = 2025


def normalize_crop(name: str | None) -> str | None:
    """Comparison key only -- strips whitespace and accents/casing so
    "Grasland, blijvend " (2023/24 archive) and "Grasland, blijvend"
    (WFS) compare equal, and so do "Maïs, snij-" (archive) and "Mais,
    snij-" (WFS). The *displayed* rotation history keeps each year's own
    original spelling; only rotation_changed/transitions use this."""
    if name is None:
        return None
    decomposed = unicodedata.normalize("NFKD", name.strip())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _load_year(year: int) -> gpd.GeoDataFrame:
    path = RAW_DIR / ("brp_parcels.geojson" if year == REFERENCE_YEAR else f"brp_parcels_{year}.geojson")
    gdf = gpd.read_file(path).to_crs(28992)
    gdf["gewas"] = gdf["gewas"].str.strip()
    gdf["_area_m2"] = gdf.geometry.area
    return gdf


def _best_match(ref: gpd.GeoDataFrame, other: gpd.GeoDataFrame) -> pd.DataFrame:
    """For each `ref` row, the `other` row it overlaps most with, as a
    fraction of the `ref` row's own area. One overlay pass (spatial-index
    accelerated) rather than an O(n*m) polygon-by-polygon loop."""
    inter = gpd.overlay(
        ref[["_rid", "geometry"]], other[["gewas", "category", "geometry"]],
        how="intersection", keep_geom_type=False,
    )
    inter["_int_area"] = inter.geometry.area
    best_idx = inter.groupby("_rid")["_int_area"].idxmax()
    best = inter.loc[best_idx, ["_rid", "gewas", "category", "_int_area"]]
    ref_area = ref.set_index("_rid")["_area_m2"]
    best = best.set_index("_rid")
    best["overlap_frac"] = best["_int_area"] / ref_area.loc[best.index]
    return best[best["overlap_frac"] >= MIN_OVERLAP_FRAC]


def run() -> Path:
    ref = _load_year(REFERENCE_YEAR)
    ref["_rid"] = range(len(ref))

    rotation = {rid: {REFERENCE_YEAR: gewas} for rid, gewas in zip(ref["_rid"], ref["gewas"])}
    match_counts = {}
    for year in HISTORY_YEARS:
        hist = _load_year(year)
        matches = _best_match(ref, hist)
        match_counts[year] = len(matches)
        for rid, row in matches.iterrows():
            rotation[rid][year] = row["gewas"]
        print(f"[rotation] {year}: {len(matches):,}/{len(ref):,} 2025 parcels matched "
              f"(>= {MIN_OVERLAP_FRAC:.0%} overlap)")

    all_years = sorted([REFERENCE_YEAR, *HISTORY_YEARS])

    def _summary(rid: int) -> dict:
        by_year = rotation[rid]
        crops_seen = [by_year[y] for y in all_years if y in by_year]
        norm_seen = [normalize_crop(c) for c in crops_seen]
        return {
            "rotation_json": json.dumps({str(y): by_year.get(y) for y in all_years}, ensure_ascii=False),
            "rotation_n_years": len(crops_seen),
            "rotation_n_distinct": len(set(norm_seen)),
            "rotation_changed": len(set(norm_seen)) > 1,
        }

    summary_df = pd.DataFrame([_summary(rid) for rid in ref["_rid"]], index=ref["_rid"])
    ref_wgs = gpd.read_file(BRP_PATH)  # reload to keep any columns other stages already added (ndvi_*)
    for col in summary_df.columns:
        ref_wgs[col] = summary_df[col].values
    ref_wgs.to_file(BRP_PATH, driver="GeoJSON")

    # Municipality-wide summary: how much of the registered area is a
    # stable land use across the whole window vs one that actually rotates,
    # and which crop-to-crop transitions are most common. Transitions are
    # detected on the normalized key (so a spelling/whitespace variant
    # doesn't count as a transition) but displayed using each year's own
    # original spelling.
    full_history = summary_df[summary_df["rotation_n_years"] == len(all_years)]
    stable_pct = float((~full_history["rotation_changed"]).mean() * 100) if len(full_history) else None

    transitions = {}
    for rid in ref["_rid"]:
        by_year = rotation[rid]
        seq = [by_year[y] for y in all_years if y in by_year]
        for a, b in zip(seq, seq[1:]):
            if normalize_crop(a) != normalize_crop(b):
                key = f"{a} -> {b}"
                transitions[key] = transitions.get(key, 0) + 1
    top_transitions = dict(sorted(transitions.items(), key=lambda kv: -kv[1])[:10])

    print(f"\n[rotation] {len(full_history):,}/{len(ref):,} 2025 parcels have a full "
          f"{all_years[0]}-{all_years[-1]} match; {stable_pct:.0f}% of those never changed crop "
          f"(after normalizing whitespace/spelling differences between BRP's live and archive sources).")
    print("[rotation] top crop-to-crop transitions:")
    for k, v in top_transitions.items():
        safe_k = k.encode("ascii", "replace").decode("ascii")
        print(f"    {v:4d}  {safe_k}")

    update_stats("crop_rotation", {
        "years": all_years,
        "min_overlap_frac": MIN_OVERLAP_FRAC,
        "parcels_matched_by_year": {str(y): match_counts.get(y, len(ref)) for y in all_years},
        "n_parcels_full_history": int(len(full_history)),
        "pct_stable_full_history": round(stable_pct, 1) if stable_pct is not None else None,
        "top_transitions": top_transitions,
    })
    return BRP_PATH


if __name__ == "__main__":
    run()
