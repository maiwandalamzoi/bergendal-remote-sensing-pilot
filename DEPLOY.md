# Deploying to Streamlit Community Cloud

This repo is prepared to deploy, but the last two steps need your own
GitHub and Streamlit accounts — those can't be done from here. Everything
up to that point is already in place.

## What works on a fresh cloud deploy, and what doesn't

`data/raw/` and `data/processed/` (the actual satellite/LiDAR/SAR rasters)
are **344 MB** and deliberately not committed — see the comment in
`.gitignore`. The one exception is `data/processed/stats.json` (a few
tens of KB, pure JSON), which **is** committed, because it's what almost
everything on the dashboard actually reads:

| Works immediately (reads `stats.json` only) | Needs the full local pipeline |
|---|---|
| All Overview KPI cards | The interactive map (Overview + Field Explorer tabs) |
| Every Altair chart (trends, forecast, land cover breakdown, air quality) | The "Build a PDF report" feature |
| Methodology tab | |
| Business case tab | |
| Villages/Kernen tab | |

If someone opens the deployed map or tries to build a PDF report without
the full data, they'll see a clear explanation (`run python
run_pipeline.py locally`), not a crash — that graceful fallback is
already built into `dashboard.py`.

**Want the map working on the public deploy too?** That means either:
1. Committing a *reduced* copy of `data/raw`/`data/processed` (re-encode
   the GeoTIFFs at lower resolution/compression to get well under the
   344 MB this repo currently produces) — a real follow-up task, not
   done here since it would change what the map actually shows.
2. Running `run_pipeline.py` (plus `fetch_cbs.py`, `fetch_brp.py`,
   `fetch_air_quality.py`) as a one-time step against a persistent disk
   Streamlit Cloud doesn't offer on its free tier — needs a different
   host (Render, Fly.io, a VM) with a real volume.

Neither is required to get the numbers/charts/methodology/business-case
majority of the app live today.

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
