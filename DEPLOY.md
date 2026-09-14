# Deploying to Streamlit Community Cloud

This repo is prepared to deploy, but the last two steps need your own
GitHub and Streamlit accounts — those can't be done from here. Everything
up to that point is already in place.

## What works on a fresh cloud deploy

The *full* multi-year archive in `data/raw/` and `data/processed/` is
**344 MB** (22 years of imagery, kept for the historical trend
computation) and stays out of git. But the map only ever reads the
*current* pipeline year at render time -- checked against every
`RAW_DIR`/`PROC_DIR` reference in `src/visualize.py`, not guessed -- and
that subset is a real **~57 MB across 14 files**, small enough to
version directly. Those 14 files (the current year's true-colour/NDVI/
land-cover/SAR/elevation rasters, the flood-event raster, the BRP parcel
geometry, `villages.geojson`, `no2.tif`) plus `stats.json` are carved out
of `.gitignore` (see its own comment for the exact list and the
`data/raw/*` + `!filename` pattern that makes the carve-out actually
work). Verified by isolating just those 14 files from the rest of
`data/raw`/`data/processed` locally and confirming `make_map()` and
`field_explorer_map()` still build correctly in both languages before
this was ever pushed -- not assumed from reading the code.

Net effect: **the whole app, including the interactive map, works on a
fresh Streamlit Cloud deploy** with no pipeline run required. The one
thing that still needs the full local archive is the **PDF report**
builder's *other* layers -- NDVI-change/SAR/elevation pages render fine
(they're in the committed 14), but anything reaching for an
uncommitted year (e.g. picking a different NDVI-change pair than
2024→2025) will show the graceful "run the pipeline locally" message
instead of crashing, per the fallback already built into `dashboard.py`.

**Known limitation:** the 14 filenames above are pinned to *this*
pipeline run's current year (`summer_2025`, `sar_2025`, the 2018–2026
change map, the 2024-01-16 flood peak). The next time `run_pipeline.py`
is run for a new current year, the `.gitignore` carve-out list and this
paragraph both need updating to the new filenames, or the map will fall
back to the old year's data until that's done.

## Steps (yours to run)

1. **Create a GitHub repository** (github.com → New repository). Public
   or private both work with Streamlit Community Cloud, but a private
   repo needs you to grant Streamlit access to it during step 3.

2. **Push this repo to it**, from this project's root:
   ```bash
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - "New app" → pick the repository and branch you just pushed.
   - Main file path: `dashboard.py`.
   - Click Deploy. First build takes a few minutes (it's installing
     geopandas/rasterio's system-level GDAL dependency via
     `packages.txt`, already in this repo).

4. **Share the URL** it gives you (`<something>.streamlit.app`) — that's
   the link people click to open the app, no account needed on their end.

## Already prepared in this repo

- `packages.txt` — `gdal-bin`, `libgdal-dev`, `libspatialindex-dev`:
  the OS-level libraries geopandas/rasterio need that Streamlit Cloud's
  base image doesn't include by default.
- `requirements.txt` — pinned, includes `pymupdf` (used by the PDF
  report's clickable navigation).
- `.gitignore` — carved out to keep `data/processed/stats.json`
  versioned while the actual raster archive stays local-only.
- Graceful fallback in `dashboard.py` for the map and PDF-report
  features when the raster archive isn't present (see table above).
