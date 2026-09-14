# Berg en Dal — remote sensing pilot

Working code for the concept note ("Berg en Dal from Orbit"): a real satellite
+ LiDAR pipeline over the Berg en Dal municipality (Gelderland, NL), from
fetch through preprocessing to unsupervised ML and two ways to look at the
result. Everything here runs on public data — no API keys, no credentials,
nothing to configure.

## Run it

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python run_pipeline.py
```

Takes a while the first time — it now also pulls 6 extra years of Sentinel-2
(`ndvi_trend.py`) and 9 extra Sentinel-1 scenes (`flood_event.py`) beyond the
core fetch. Every fetch step skips a date/label already on disk, so re-runs
are fast except for whatever's genuinely new; the two heavy steps can also be
run alone (`python src/ndvi_trend.py`, `python src/flood_event.py`) without
repeating the rest. Then either:

```bash
streamlit run dashboard.py     # headline numbers + embedded map
```

or open `outputs/bergendal_map.html` directly in a browser for just the map.

## What it does

1. **AOI** (`src/aoi.py`) — the official municipal boundary (Kadaster/CBS
   "bestuurlijkegebieden", gemeentecode `GM1945`), pulled live from PDOK's
   public WFS. Not a hand-drawn bounding box — the real polygon.
2. **Fetch**
   - `src/fetch_sentinel2.py` — least-cloudy Sentinel-2 L2A scene per date
     window from Microsoft's Planetary Computer STAC catalog, mosaicking
     the (up to two) MGRS tiles that cover the AOI and clipping to the exact
     municipal polygon. Only the AOI window is read off each remote COG.
   - `src/fetch_sentinel1.py` — Sentinel-1 RTC (radiometrically
     terrain-corrected SAR, VV+VH). Picks the nearest date whose scene
     *fully* covers the AOI polygon, not just whose bounding box intersects
     it — Berg en Dal sits close to a swath edge on some passes, where a
     scene can clip only one corner of the municipality while still
     technically "intersecting" it.
   - `src/fetch_ahn.py` — DTM + DSM from AHN (the Dutch national LiDAR
     archive) via PDOK's WCS, requested pre-downsampled to 5 m directly by
     the service rather than pulled at native 0.5 m and resampled locally.
     DSM − DTM = a canopy/building-height model.
3. **Preprocess**
   - `src/preprocess.py` — reverses the Sentinel-2 L2A baseline-04.00
     reflectance offset, cloud/shadow-masks using the Scene Classification
     Layer, computes NDVI and NDWI. The offset is applied conditionally on
     each scene's own processing baseline (stamped onto the raster by
     `fetch_sentinel2.py`), not unconditionally — see the bug note below.
   - `src/preprocess_sar.py` — linear gamma0 → dB, a simple open-water mask
     (VV below a data-driven fixed threshold), a cross-check against the
     optical land cover's own water class, and a before/after flood-extent
     compare between the ordinary-flow baseline and a documented Jan 2024
     high-water scene. Kept as-is for comparison; `src/flood_event.py`
     (below) is the improved version of this specific analysis.
   - `src/flood_event.py` — the flood-mapping fix the section below used to
     flag as needed: change detection instead of one fixed absolute
     threshold. Builds a per-pixel reference composite (median VV dB) from
     four ordinary-condition autumn-2023 Sentinel-1 passes, then flags a
     pixel as flooded where a scene from the documented Jan 2024 Rhine/Waal
     high-water event drops more than 3 dB below *its own* normal level —
     for five dates bracketing the Lobith gauge's peak (rise, and three
     stages of recession; no scene fully covers the AOI on the peak day
     itself, checked directly against the STAC catalogue).
4. **ML** (`src/landcover_ml.py`) —
   - unsupervised land cover: KMeans over NDVI/NDWI/reflectance/brightness,
     with clusters named from their own centroid signature (water, built-up,
     farmland, and three vegetation-density tiers) rather than a fixed
     lookup table, since a hard NDVI threshold turned out not to separate
     Berg en Dal's uniformly-green August land cover well;
   - year-over-year NDVI change between the two most recent cloud-free
     August scenes (2024 → 2025).
   - `src/ndvi_trend.py` — the actual multi-year series that one year-pair
     delta above can't be: mean clear-ground NDVI *and* NDWI for the
     least-cloudy August date in every year 2005–present, computed to
     whichever August has most recently finished. Two sensors, one series:
     `src/fetch_landsat.py` + `src/preprocess_landsat.py` (Landsat 5/7/8,
     30m, red/green/nir08/qa_pixel bands) for 2005–2017, where Sentinel-2
     L2A coverage over this AOI isn't reliable yet, handing off to
     `fetch_sentinel2.fetch_date` + `preprocess.process` (10m) from 2018 on
     — both write the same `optical_summer_{year}` stats key regardless of
     source, so the dashboard reads one continuous series.
5. **Weather & correlation**
   - `src/fetch_weather.py` — real daily weather 2005–present from KNMI's
     public daggegevens API (no key), station 275 Deelen Airport (the
     nearest official station to the AOI — none sits inside the
     municipality itself). Saves the full daily series
     (`data/raw/knmi_daily.csv`) and derives August + full-year summaries
     (rainfall, temperature, sunshine, evapotranspiration, humidity).
   - `src/climate_correlation.py` — Pearson correlation between the
     NDVI/NDWI trend and August weather, across every year both exist for.
     Finding: August temperature is the strongest single correlate of that
     year's NDVI (r ≈ -0.6, hotter → lower vigour); rainfall alone
     correlates weakly. Documented caveats ship with the numbers: ~20
     years is a small sample for a correlation coefficient, a snapshot day
     is being compared against a month's integral, and temperature/
     rainfall/sunshine move together across a real summer so no single
     variable can be cleanly isolated as "the" driver.
6. **Visualize** (`src/visualize.py`) — PNG maps for every layer (true
   colour, NDVI, land cover, NDVI change, SAR backscatter, SAR water mask,
   both flood-extent methods, elevation, canopy/building height), assembled
   into one Folium/Leaflet map (`outputs/bergendal_map.html`) with all of
   them as toggleable overlays. The map's own layer names and legends
   switch language too (`make_map(lang=...)` / `field_explorer_map(lang=...)`)
   — the dashboard's language switch isn't just the Streamlit chrome around it.
7. **Dashboard** (`dashboard.py`) — a Streamlit app with a sidebar (real
   English/Dutch language switch — every UI string routes through one
   `t(en, nl)` call keyed off `st.session_state.lang`, not both languages
   shown inline at once — plus a village/place selector, see **Villages**
   below) and nine tabs: Overview, **Field Explorer**, **Villages**, Land
   & Crops, **Trends & Climate**, **🔮 Forecast** (simple OLS trend
   projections plus a crop-rotation Markov model — see **Forecast**
   below), Environment & Energy, Water, and a Business case tab spelling
   out who'd actually pay for this and what the real remaining data gaps
   are. Genuinely Dutch data values (BRP crop and
   category names) stay Dutch regardless of the switch — those are real
   registry records, not UI chrome to translate. Everything reads from
   `data/processed/stats.json` (each stage writes its own summary there via
   `src/statsutil.py` — nothing is recomputed for display).

   **Villages** lists all 13 real places inside the municipality (real CBS
   district boundaries and population/area, not estimated), with a
   population and an area chart, a full detail table, and the
   municipality-wide population/housing-stock growth 2018-2024
   (`cbs_trend.py`) for context. The sidebar's village selector filters
   the Field Explorer map to just that place's real administrative
   boundary *and* re-cuts the Overview map to it too — picking a village
   doesn't just zoom the map in on the whole municipality's imagery, it
   clips every raster layer to the village's real boundary
   (`_clip_to_village()` in `visualize.py`: the alpha channel of each
   already-rendered true colour / NDVI / land cover / SAR / flood /
   elevation / air-quality / BRP-category PNG is zeroed outside the
   village polygon, rasterized against the same WGS84 rectangle Folium
   draws the PNG onto), leaving the plain OpenStreetMap basemap visible
   everywhere else — a real "snap and cut," not an outline drawn on top
   of unchanged imagery (other tabs stay municipality-wide, stated
   explicitly rather than left ambiguous).

   **Map controls & print report** — both Folium maps (Overview and Field
   Explorer) now read like an actual map report, not just a widget: a
   real Leaflet fullscreen control (`folium.plugins.Fullscreen`,
   top-left), a scale bar (`control_scale=True`), a north arrow (both
   basemaps are plain north-up OpenStreetMap tiles, so a static badge is
   cartographically correct), a title/scope/date/data-source strip baked
   into the map's own HTML (`_report_header_element()` — a small pill on
   screen, a full title block under `@media print`, since it's inside the
   map's `st.components.v1` iframe and needs to be part of *that*
   document to survive printing), and a real legend for every colour mode
   (category, crop family, and now the two NDVI modes too, each a
   gradient swatch built from the same vmin/vmax/colormap the fill itself
   uses, so legend and fill can't disagree). The sidebar's "Print this
   tab" button triggers the browser's own print dialog
   (`window.parent.print()`) against dedicated print CSS that hides the
   sidebar, Streamlit's header/toolbar and the tab bar — a plain "save
   the page as PDF" rather than a separate export pipeline, but with
   the title/legend/scale/north-arrow already drawn onto the map so what
   prints looks like an actual cartographic sheet, not a screenshot of an
   interactive widget.

   **Trends & Climate** is the 22-year analysis tab: a **Year Explorer**
   widget up top (`st.select_slider`) that pulls every number this
   pipeline has for one chosen year — NDVI, NDWI, sensor/platform, that
   August's rainfall/temperature/sunshine, crop-rotation coverage, and
   flagged notes (hottest/driest/wettest August on record, the 2018
   drought, the Jan 2024 flood) — into one place, and rings that year on
   the trend charts below it so the selection stays visible in context
   rather than just as a number; then the NDVI and NDWI trend charts
   themselves (colour-coded by sensor, validated CVD-safe palette), the
   three August weather charts as small multiples (never one dual-axis
   panel combining unrelated scales — the dataviz method's #1 flagged
   anti-pattern), and a diverging correlation-coefficient chart with the
   full honest write-up of what the correlation does and doesn't show.

   **Field Explorer** is the interactive centrepiece: every one of the
   3,787 BRP parcels is individually clickable, recoloured live by whatever
   you pick from a dropdown (category, crop family, NDVI 2025, or NDVI
   change 2024→2025), and clicking one shows its real crop name, area, a
   two-year NDVI bar chart, and — where `crop_rotation.py` has run — its
   real 2020–2025 crop-rotation history with change markers, not just this
   year's snapshot (`src/brp_zonal.py` computes real per-parcel NDVI via a
   vectorised zonal-stats pass, not a
   per-polygon loop). The click-to-inspect mechanism does a point-in-polygon
   lookup against the precise geometry rather than trying to read properties
   back off the rendered map layer — streamlit-folium's click return for a
   plain (non-Draw) GeoJson layer turned out to expose only geometry and
   click coordinates, not feature properties, so the reliable path is
   `last_object_clicked` (lat/lng) → `geopandas` `.contains()` against the
   source data.

   **Crop family colouring** (`classify_crop()` in `src/visualize.py`) sorts
   every one of the AOI's 102 distinct real BRP crop names into one of eight
   families by keyword — grassland, maize, cereals, root/bulb/tuber,
   vegetables, fruit/orchard, cover/fodder/oilseed, nature/landscape/water —
   plus an "other" catch-all that only 3% of parcels land in. 102 crops is
   past what any palette can give distinguishable individual hues (a
   choropleth's own hard cap is 3-4 fully-safe categories), so colour here
   carries the *family* and an emoji icon + the exact crop name (tooltip,
   click panel, and the map's own on-canvas legend) carries the specific
   crop — composite encoding instead of a top-9-and-grey scheme. In crop-
   family mode the icon also renders directly on the map at each field
   >=0.3ha (not just on hover), clustered below zoom 15 (`folium.plugins.
   MarkerCluster`) so 1,500+ markers don't turn into an unreadable pile at
   municipality-wide zoom, and expanding to individual icons above it.

   Clicking also samples the AHN elevation/canopy-height and SAR
   backscatter rasters at that exact point (reprojecting the click
   coordinate into whatever CRS each raster actually uses), so the panel
   answers "what else do we know about this spot" beyond just the BRP record.

   The dashboard has its own visual identity (`.streamlit/config.toml` +
   injected CSS in `dashboard.py`) — Fraunces/Source Sans 3 type, a forest-
   green header banner with a data-freshness timestamp, and card-styled
   metrics — rather than default Streamlit chrome, since this is meant to
   go in front of the gemeente, not just a developer's own screen.

### Beyond satellite: registry data joined in

Three more real, free, no-key data sources close gaps the satellite/LiDAR
pipeline alone can't fill:

- **`src/fetch_brp.py`** — BRP (Basisregistratie Gewaspercelen), the Dutch
  government's registry of every subsidised farm parcel: exact polygon +
  actual crop name, filed by the farmer each year. 3,787 parcels, 4,499 ha,
  broken into grassland/arable/nature and named crops (permanent grassland,
  silage maize, winter wheat, sugar beet, potatoes...). This is the ground
  truth the concept note's Phase 2 asked for, and it complements rather than
  replaces the KMeans land cover — BRP only covers registered agricultural
  land, not unmanaged forest.
- **`src/fetch_cbs.py`** — official CBS municipality statistics: population
  (35,474), households (16,385), housing stock, and energy-transition
  indicators (46% of homes already have rooftop solar; 88% are still
  gas-heated). Not remote sensing — nobody derives household counts from a
  satellite image — but the registry half of the same picture.
- **`src/fetch_air_quality.py`** — RIVM's official 1×1 km national air
  quality grids: NO2 (9.5 µg/m³ mean, 19.7% of the area over the WHO
  guideline), PM10, PM2.5. Confirmed absent from the same RIVM service:
  NH3/nitrogen deposition, the figure Dutch farm nitrogen permitting
  actually runs on — that lives in RIVM's separate AERIUS/GDN product as an
  annual grid download, not a queryable API. Flagged, not faked.

### Crop rotation: five more years of BRP, matched field-to-field

`fetch_brp.py` only ever gave one year (2025). Real rotation needs
history, and the BRP WFS's own `GetCapabilities` confirms it only ever
serves the current year — checked directly, not assumed. The only
historical source is PDOK's ATOM feed of whole-Netherlands GeoPackages
(2.5–3GB *each*, one per year), so **`src/fetch_brp_history.py`** reads
each one over HTTP via GDAL's `/vsicurl/` virtual filesystem with a
bounding-box filter instead of downloading it — the GeoPackage's own
spatial index answers a bbox query in a few minutes and a few MB of real
transfer, confirmed directly (2,126 features in ~3 minutes for one test
year), not a multi-gigabyte pull. One-time, standalone step (not part of
the default `run_pipeline.py` pass — see its docstring), fetching
2020–2024 to sit alongside 2025.

**`src/crop_rotation.py`** then matches every 2025 field to its
best-overlapping parcel in each earlier year (max shared area via
`geopandas.overlay`, accepted only past 30% overlap — BRP re-registers
parcel boundaries every year, so there's no stable ID to join on) and
writes each field's real crop sequence back onto `brp_parcels.geojson`
(`rotation_json`, read by the dashboard's Field Explorer) plus a
municipality-wide summary (top transitions, % stable vs. rotating).

**Two real data-quality bugs found and fixed while building it:** the
historical archive has trailing whitespace on some crop names in
2023/2024 that the live registry doesn't, and spells maize with a
diaeresis ("Maïs") where the live registry drops it ("Mais") — both read
as a false crop change on every matching field before being normalized
away (`normalize_crop()`: strip, NFKD-decompose accents, casefold — used
only for *comparison*; the display still shows each year's own original
spelling). Fixing both moved the "never changed crop" rate from an
implausible 4% to a believable 38%. **A real data-coverage limit, left
visible rather than hidden:** parcel counts jump from ~1,750 (2020–2022)
to ~4,100+ (2023–2025) at roughly constant total area — BRP started
registering landscape elements (hedgerows, ditches) as their own small
parcels around 2023, so a small 2025 hedge parcel matching a much larger
2020 field is that schema change surfacing in the match, not a real
event; each match's own `overlap_frac` is kept in the data so this is
checkable per field.

**What it found:** genuine, textbook Dutch arable rotation — winter
wheat ↔ sugar beet and potatoes ↔ wheat are the two most common
transitions, real crop-management signal rather than noise. One
individually-checked field: wheat (2020) → maize (2021) → sugar beet
(2022) → wheat (2023) → maize (2024) → potatoes (2025), a full six-year
intensive arable cycle, visible for that one specific parcel in Field
Explorer. `"Grasland, tijdelijk" → "Grasland, blijvend"` (temporary → permanent
pasture status) is the single most common "transition" — a registry
reclassification, not a physical land-use change, called out explicitly
in the dashboard rather than left to read as farmers converting cropland
to pasture at that scale.

### Land cover, air quality & population change over time; soil; villages

- **`src/landcover_trend.py`** — built-up, forest/dense vegetation,
  agriculture and water as real hectares, 2018-present, re-running
  `landcover_ml.classify()` for every Sentinel-2 year already on disk (no
  new fetching). A real bug found and fixed while building it: comparing
  each year's *raw* classified hectares directly produced an implausible
  swing (built-up area apparently halving then partly regrowing) because
  each year's cloud-free footprint differs in size — fixed by restricting
  every year's count to the pixels valid in *all* years (a fixed,
  common-footprint denominator), documented in the module's own
  docstring. Even after the fix, "Built-up / bare" specifically stays
  noisier than "Forest" or "Water" (an unsupervised 6-cluster classifier
  can relabel bare/just-harvested cropland as built-up between one year's
  centroid fit and the next) — named as a real limitation, not smoothed
  over, with BAG (the building registry) flagged as the real cross-check
  this hasn't been validated against yet.
- **`src/fetch_air_quality_trend.py`** — NO2/PM10/PM2.5/EC, 2013-2024,
  from the same RIVM WCS `fetch_air_quality.py` already uses, checked
  directly against its own capabilities list for which years it actually
  publishes (all four, every year). **A real bug found and fixed:** some
  older coverages (checked directly, e.g. 2014's NO2) declare
  `nodata=0.0` in their own GDAL profile but actually fill nodata pixels
  with float32's most-negative value instead, corrupting a mean computed
  by excluding only the declared nodata to `-inf`. Fixed with a physical
  sanity bound (no real concentration is negative or above 1000 µg/m³),
  applied to both the trend fetcher and the original single-year one.
  **Finding:** all four pollutants fell over this period (real, documented
  Dutch/EU air-quality improvement), with a clear, real 2020 dip and 2021
  recovery — the documented COVID-19 lockdown traffic drop. EC (elemental
  carbon / soot) is included as the closest real combustion proxy this
  free service has, since neither CO2/GHG nor NH3 turned out to be
  reachable — see below.
- **`src/fetch_villages.py`** — the 13 real villages/hamlets inside the
  municipality (Berg en Dal is a 2015 merger of the former municipalities
  Groesbeek, Millingen aan de Rijn and Ubbergen), from CBS's own "wijken"
  (district) boundaries and population/area — the same PDOK WFS
  `fetch_cbs.py` already uses, one geographic level down. Powers the
  dashboard's village selector (both maps zoom/filter to whichever village
  is picked, see **Map controls & print report** below). A real precision
  bug fixed while wiring the Field Explorer's parcel filter: each field's
  centroid was tested against the village polygon in geographic
  (WGS84/EPSG:4326) coordinates, where degrees of longitude aren't
  constant-length — fixed by reprojecting both to RD New (EPSG:28992,
  the Dutch planar CRS already used elsewhere in this pipeline) before
  the `.within()` test.
- **`src/fetch_soil.py`** — organic carbon, pH, nitrogen, texture
  (clay/sand/silt), bulk density and cation exchange capacity, sampled at
  real BRP farmland-parcel centroids via ISRIC SoilGrids (a free, no-key,
  global 250m soil model). Not the first choice: PDOK/BRO's own Dutch
  soil map (Bodemkaart) has no discoverable public WFS/WMS under any of
  the endpoint patterns this pipeline's other PDOK services use (checked
  directly, several naming variants tried), and its province-level
  alternatives found via data.overheid.nl (Utrecht, Zeeland, Zuid-Holland)
  don't cover Gelderland — a real methodology substitution, documented
  rather than silent. SoilGrids itself needed two fixes: a point can land
  on open water in this river-adjacent AOI and return null (worked around
  by sampling many parcel centroids and averaging the real ones), and its
  free-tier rate limit needed real pacing (5s between requests) plus
  retry-on-429 handling.
- **`src/cbs_trend.py`** — population and housing stock, 2018-2024, from
  CBS's own per-year WFS republications. A real fix: 2018/2019 use a
  year-suffixed type name (`gemeenten2018`) that later years dropped for
  a plain `gemeenten` — checked directly rather than silently skipping
  those two years. `src/fetch_cbs.py` itself was also extended from ~18
  fields to most of the non-suppressed fields the same WFS record already
  carries (demographics, housing value/tenure/type, a full 7-sector
  business breakdown, benefit-recipient counts) — real municipal detail
  that was sitting in a response this pipeline was already making.

**What CO2/GHG and NH3 turned out to actually require, checked directly
rather than assumed:** municipal-level CO2/GHG (Klimaatmonitor, the
standard Dutch source) sits behind an authenticated OData API — confirmed
by an HTTP 401 "Guest user group not found, No access!" response, not
merely unfetched. NH3/nitrogen deposition (RIVM's GDN/AERIUS product) has
no public WCS/WFS at all, under any of the endpoint patterns RIVM's own
other services (used successfully elsewhere in this pipeline) follow.
Both are named gaps with a specific, checked reason, not silent
omissions — see the Environment & Energy tab.

### Forecast: simple, honest ML on everything above, not a black box

**`src/forecast.py`** — one module, four forecasts, all built on data this
pipeline had already fetched (zero new network calls, `python
src/forecast.py` runs in seconds):

- **Vegetation health (NDVI/NDWI):** ordinary least squares on the
  22-year trend, projected 5 years out with a real OLS 80% prediction
  interval (`scipy.stats.t`, widening with distance from the data's own
  mean year — the textbook formula, not a fixed ±band). R²=0.31 on the
  22-year series, stated on the tab itself rather than hidden behind the
  point estimate — this pipeline's own Trends & Climate tab already says
  "read the shape, not the slope" about the *historical* NDVI series; a
  forecast built on top of it inherits that caveat, more so.
- **Land cover** (built-up/forest/agriculture/water): the same
  OLS-plus-interval treatment per category on `landcover_trend.py`'s
  already common-footprint-corrected series, clipped to `[0, that
  series' own footprint]` — a straight line doesn't know hectares can't
  go negative or exceed the area it's measured over.
- **Population & housing stock:** the same, on `cbs_trend.py`'s CBS
  registry series — R²=0.85 (population) and 0.98 (housing stock), by
  far the most defensible forecasts here, a near-straight administrative
  series rather than a noisy environmental one.
- **Crop rotation — not a trend line at all:** a first-order Markov chain
  over crop *families* (not the 102 raw crop names — not enough real
  transitions per exact crop to say anything about most of them), built
  from **14,424 real transitions** in every field's own multi-year BRP
  history (`crop_rotation.py`'s `rotation_json`). `transition_matrix[A][B]`
  is the empirical probability that a field grown as family A one year is
  family B the next, among the transitions this pipeline's own matched
  history actually contains — not a hand-picked agronomic rule. Sample
  result: maize self-persists 49% of the time, rotates to cereals or root
  crops ~28% combined — textbook Dutch arable rotation, not noise.

**Powers two dashboard surfaces:** a **🔮 Forecast** tab (charts: solid
history, dashed projection, shaded 80% band — "read the band, not the
dashed line" stated once up top rather than five times below) and Field
Explorer's **🔮 Predicted next crop (ML)** colour mode, which colours
every field by its own most-likely next-year family (looked up from the
same transition matrix, not refit per request) with the predicted
probability in its tooltip.

**A real bug found and fixed while building this:**
`visualize.py`'s `classify_crop()` matched crop names by an ASCII
substring (`"mais" in name.lower()`), so it silently missed every
occurrence of BRP's historical-archive spelling "Maïs" (diaeresis) —
every archive-year maize parcel would have fallen into "other" instead
of "maize" in the transition counts, undercounting the single most
common rotation crop in the whole matrix. Fixed by accent-normalizing
(NFKD-decompose, drop combining marks) before matching — the same
technique `crop_rotation.py`'s own `normalize_crop()` already uses for a
different reason (detecting whitespace/spelling-variant false-positive
"crop changes"). Caught specifically *because* this forecast needed to
classify crop names from every historical year, not just 2025's — a case
the dashboard's own existing `classify_crop()` calls never exercised.

### Forest & land cover change: a real spatial map, not just a hectare number

The Trends & Climate tab's own land-cover-trend chart (2018-present) only
ever showed *how much* built-up/forest/agriculture/water changed, as an
aggregate hectare series — not *where*. **`landcover_trend.build_change_map()`**
answers that directly: every pixel that changed broad category between the
first and last classified Sentinel-2 year is compared pixel-for-pixel
(checked directly that both years share an identical transform/shape/CRS,
not assumed) and labelled — forest loss, forest gain, built-up growth, or
other change, priority-ordered so a pixel that's both "left forest" and
"became built-up" reads as forest loss, the category this map exists to
answer, rather than being split across two labels. Unchanged ground (the
large majority) is left fully transparent so only real change draws the
eye. Real result, 2018→2026: **714 ha forest lost, 789 ha gained** (net
+75 ha, matching the independently-computed aggregate trend to within
0.1 ha — a genuine cross-check between two different views of the same
underlying classification, not a coincidence). Rendered as its own
toggleable layer on the Overview map (`src/visualize.py`'s
`landcover_change_png()`) and summarized with metrics + a "how this map
is calculated" expander on the Land & Crops tab.

**A real, stated limit, not an oversight:** this map's spatial window is
the Sentinel-2 era (2018-present) only, even though the *trend line*
elsewhere in this dashboard goes back to 2005. The pre-2018 years use 30m
Landsat imagery — a different pixel grid entirely — so comparing it
pixel-for-pixel against this 10m grid without reprojecting one onto the
other first would silently misalign ground, not just lose precision.
Extending this spatial map back to 2005 is a real, identified next step
(resample the Landsat-derived classifications onto the Sentinel-2 grid),
not attempted here because doing it without that step would trade
accuracy for a bigger number, exactly the kind of thing this project's
own methodology consistently refuses to do.

### A deliberate icon pass: fewer, not more

After a design review, every tab-bar icon and one-off decorative emoji
prefixed onto section headings/inline notes was removed in favour of
plain text — Streamlit's `st.tabs()` can only render plain text or emoji
per tab (no custom icon graphics), so plain text was the more
professional, human-designed-reading option available, not a
compromise. What's *kept*: the crop-family pictograms on the map and its
legends (🌱🌽🌾🥔🥬🍎🍀🌳❔ — real, distinguishing symbols a user reads
in seconds, not decoration), the location-pin/print-button icons (📍🖨️,
conventional functional symbols), the single 🛰️ used consistently as
this project's own wordmark (header, sidebar, browser favicon), and
`st.info`/`st.warning`/`st.error`'s own default chrome icon (Streamlit's
UI, not something this project added). Custom `icon=` overrides on those
calls were removed for the same reason.

### Grouped map layers: a flat 14-checkbox list, categorized

The Overview map's own layer list grew to 13-15 items (true colour, NDVI,
land cover, NDVI change, SAR backscatter/water mask, two flood-extent
methods, elevation, canopy height, air quality, BRP fields, the forest
change map, municipal/village boundaries) as more real analyses landed —
all in one flat Leaflet checkbox list, genuinely hard to scan. Replaced
with `folium.plugins.GroupedLayerControl` (the `leaflet-groupedlayercontrol`
plugin), sorting every layer into six labelled groups — Optical &
vegetation, Land cover & change, Radar (SAR), Elevation, Environment,
Registry & boundaries — matching how a person actually thinks about "what
kind of layer is this," not the order it happened to get built in.
`exclusive_groups=False` keeps every group a checkbox list (several
layers, within or across groups, can still be shown together at once,
exactly like before) — only the layout changed, not the interaction.
Field Explorer's own layer list (3-4 items: fields, village outline,
municipal boundary) stays a plain `LayerControl` — short enough that
grouping it would be overhead, not help.

### Dutch by default; a design pass for "make it nicer, not more emoji"

**Dutch is now the default language** (`st.session_state["lang"] = "nl"`),
English one click away in the sidebar — the primary audience (Berg en Dal
municipal staff, local farmers) is Dutch; the app used to default to
English.

After the icon-reduction pass left the page feeling flat, a second design
pass added real visual structure back without reintroducing emoji:

- A **colored accent bar** on every section heading (`h2`/`h3` get a
  4px `border-left` in the forest-green brand colour) — one small,
  consistent "designed" touch repeated everywhere, cheaper and more
  restrained than decorating each heading individually.
- A **hand-drawn line-icon set** (`icon_svg()` in `dashboard.py`, plain
  inline SVG — calendar/grid/satellite/wave/person/house/leaf/pin/sun,
  Feather/Lucide-style single-stroke shapes, no icon-font or CDN
  dependency) replacing emoji on the header stat pills — the deliberate
  alternative to "colourful emoji vs. no icon at all."
- A **subtle diagonal texture** layered into the hero header's existing
  green gradient (a repeating linear-gradient at low opacity) — texture,
  not another colour.
- **Every headline metric is now real, native `st.metric` with
  `border=True`** (a hover-lift shadow instead of a hand-rolled card),
  and the two with real multi-year data (population, households) get an
  inline sparkline (`chart_data=cbs_trend[...]`) for free.
- **Every headline metric is genuinely clickable now** — a `st.popover`
  under each one surfaces real detail this pipeline already had but
  wasn't showing (population density/age split/births vs. deaths,
  household size/composition, top crops, water-vs-land area split,
  gas-free % and electricity use) rather than a fake link to nowhere.

**The map's own legends got the same treatment.** Four legend boxes
(land cover, BRP categories, forest-change, flood) used to float at
independently hand-picked pixel offsets, all always fully open — visual
clutter that got worse every time a new layer was added. Replaced with
plain HTML `<details>`/`<summary>` elements (no JS, works everywhere)
stacked into one collapsible panel bottom-left (land cover open by
default, the rest one click away) plus the flood legend collapsible on
its own bottom-right — the map now shows exactly as much as is asked
for, not everything at once.

### Chart polish, and a professional icon set for the map's own crop markers

Two more direct-feedback passes:

**Charts (`dashboard.py`):** every line chart (NDVI/NDWI trend, air
quality trend, the forecast charts) now uses `interpolate="monotone"` --
a real curve through the same real points, never overshooting past a
value, so the shape reads better without implying data that isn't
there. `CHART_AXIS_KW` split into `CHART_AXIS_X_KW`/`CHART_AXIS_Y_KW`: the
y-axis (a real quantity) now carries a light, recessive horizontal grid
genuinely useful for reading a value off, while the x-axis (almost
always an ordinal year -- 20+ categories) stays grid-free rather than
turning into a busy vertical comb. Both axes now render in the app's own
body font (Source Sans 3) instead of Vega-Lite's default, so a chart
never reads as a different, less-designed surface bolted onto the page.

**Crop icons on the map (`src/visualize.py`):** the CROP_FAMILIES emoji
(🌱🌽🌾🥔🥬🍎🍀🌳❔) stay as the safe fallback for plain-text contexts
(Altair chart labels, Leaflet `GeoJsonTooltip` field values -- a custom
icon literally cannot render there, only Unicode text does), but every
place that's real HTML -- the crop-family marker pins on Field
Explorer's map and both crop-family legends -- now uses `crop_icon_svg()`:
a hand-drawn Feather/Lucide-style line icon (grass blades, a corn cob,
a wheat ear, potato-with-eyes, a leafy vegetable, an apple, a
three-leaf clover, a tree, a dashed "unclassified" circle) in a white
circle badge coloured to match that family's own map fill -- so the pin
and the field it sits on always visually agree, and the icon itself
reads as a real pictogram rather than an emoji.

*(Superseded in part by the v2 redesign below -- three of these nine
shapes didn't hold up on the real map and were redrawn.)*

### Crop icons v2: fixing the three that read wrong on the actual map

The icon set above was designed and checked in isolation, but three of
the nine shapes read badly once actually seen in context on the live
map and legend, at the real 24px badge / 15px icon size:

- **"Nature, landscape & water"** was a plain circle-on-a-stick. On
  this map that's a direct collision -- pin markers already mark
  villages, so the icon for a crop family looked like a place marker,
  not a tree. Redrawn as a tiered conifer silhouette (three stacked
  triangular tiers + a trunk line) -- a shape a location pin can't be
  mistaken for.
- **"Vegetables"** (a concentric leaf/cabbage path) rendered as a
  striped melon at small size. Redrawn as a single classic leaf
  outline with a centre vein -- reads clearly even at 15px.
- **"Cover crops"** (three overlapping teardrop "petals") rendered as
  a wine glass or slingshot. Redrawn as three overlapping circles in a
  trefoil (a clover) with a short stem -- distinct from the other
  eight shapes and unambiguous at a glance.

Grassland, maize, cereals, root crops, fruit, and "other" were kept as
designed in the first pass -- they already read correctly.

Both redesigns went through the same check before shipping: render
every candidate at a large legibility-check size *and* the real
on-map badge size in a disposable local preview page, screenshot it,
and only replace the shipped icon once it read unambiguously at the
size it's actually seen at. The on-map marker badge itself was also
bumped from 22px/13px-icon to 24px/15px-icon (and the maize
kernel-dot radius from 0.5 to 0.8) -- the size actually tested, not
the original placeholder size.

### Overview KPI cards: one consistent structure, real micro-visuals, real units

The five Overview headline cards (`kpi_card()` in `dashboard.py`) were
inconsistent -- two had a native `st.metric` sparkline, three didn't, so
the row of "More" buttons beneath them sat at different heights, and
"Land area 8,639" was missing its unit. Rebuilt as one shared structure
(label, real period/year, big number **with unit**, a one-line real
context fact, then a fixed-height visual slot) so every card renders the
same height regardless of what it contains:

- **Population, Households** get a real inline sparkline (`kpi_sparkline_svg()`,
  plain hand-drawn SVG, no charting library) through their real
  `cbs_trend.py` multi-year series, first and last value labelled
  directly on the line -- the actual axis at this size, not decoration.
- **Homes with solar** gets a proportion bar (`kpi_proportion_bar_svg()`) --
  the honest micro-visual for one percentage with no time series behind it.
- **Registered farmland** and **Land area** originally showed *no*
  visual -- neither has a real multi-year series in this pipeline (BRP
  only has the current registry year; land area is structurally
  constant) and a decorative line with nothing behind it would have
  been the thing this whole redesign was fixing. Superseded below by a
  composition visual once a real, non-fake share-of-a-real-total
  existed to show instead.
- Every card's data-ink colour is the header's own dark green
  (`KPI_INK = "#2E5943"`, the same `--forest` used throughout the rest
  of the page), so the micro-visuals read as part of this design, not a
  generic grey afterthought.
- Each card also gained a real **ⓘ info popover** (see the Methodology
  section below) next to its "More" popover, both native `st.popover`
  widgets so they align at the same row across all five cards.

### Methodology: one real, code-verified entry per indicator

**`src/methodology.py`** is now the single source of truth for how every
number on this dashboard is actually computed -- a `MethodEntry` per
indicator (what it measures, exact source/provider/product, date range,
resolution, ordered processing steps, known limitations, and open
questions where the code itself doesn't pin something down), read
directly from the fetch/processing code while writing it, not from
memory. It powers two things that literally cannot drift apart, because
one generates the other:

- **`METHODOLOGY.md`** (repo root) -- `python src/methodology.py` regenerates it.
- **The dashboard's own Methodology tab** -- renders the same `ENTRIES` list directly.

Covers, at minimum, exactly the indicators asked for: population,
households, registered farmland, land area, homes with solar, NDVI,
NDVI change, KMeans land cover, SAR backscatter, SAR water mask, both
flood-extent methods, AHN DTM, AHN nDSM, and NO2 -- 15 entries in total.
Two real gaps were surfaced (not guessed around) while writing it:
Sentinel-1 RTC's native pixel resolution is never asserted in this
pipeline's own fetch code, and the "AHN4" label used elsewhere in this
app's own captions is never checked against the WCS response's own
version metadata -- both listed as open questions on their entries
rather than stated as fact.

### Every measurement placed on its own real scale, not shown bare

Direct feedback: several numbers looked "naked" -- a percentage or a
bounded scientific index with nothing showing *where in its real range*
the value sits, and two Overview cards (Registered farmland, Land area)
had no visual at all. Three small, reused helpers in `dashboard.py`
close this without inventing a different design language per metric:

- **`kpi_stacked_bar_svg()`** -- a composition, not a single
  proportion: two or more real quantities as segments of one bar, each
  segment's width its own real share of the total (a 2px gap between
  segments, per this project's own dataviz convention, rather than one
  flat fill implying a single uniform quantity). Used for **Registered
  farmland** (farmland ha vs. the rest of the municipality's land) and
  **Land area** (land ha vs. water ha) -- both cards now show a real
  composition instead of nothing, and both gained a "X% of ..." line
  in their context text.
- **`index_range_svg()`** -- a bounded scientific index (NDVI/NDWI's
  real -1..+1 range) placed on a diverging track from a low-value
  colour through a neutral grey midpoint to a high-value colour, both
  endpoints labelled with the scale itself, with a marker at the
  actual value -- so the number's *position* in its own real range is
  visible, not just its digits. NDVI uses brown (low vigour) -> green
  (high vigour); NDWI uses brown (dry) -> blue (wet), the same
  brown/green convention the map's own NDVI-change legend already
  used, so a number and its colour never disagree. Applied to the
  Trends & Climate Year Explorer's NDVI/NDWI metrics and to the
  Field Explorer field-detail panel's NDVI 2025 / NDVI-change numbers.
- **`metric_with_bar()`** -- a thin wrapper that renders a normal
  `st.metric` and drops one of the visuals above directly underneath
  it, so every percentage metric elsewhere in the app (cloud-free
  coverage, homes with solar, gas-free homes, SAR/optical water
  agreement IoU, soil clay/sand/silt %) gets the same proportion-bar
  treatment as the Overview cards without a bespoke HTML block at each
  call site.

Every one of these bars is real language-independent SVG -- built once
per value, not per language -- so the fix carries through both English
and Nederlands without a second implementation to keep in sync; the
`ⓘ` methodology popovers next to the NDVI/NDWI metrics use the same
`method_popover()` as the Overview cards, so the "what does this
number actually mean" link exists outside the Overview tab too.

**A small ⓘ info popover, not a broken cross-tab link.** Every KPI card
and four of the map's own legend boxes (land cover, BRP, forest-change,
flood) carry a `method_popover()`/`ℹ️ Method:` reference to the matching
entry. This is deliberately a real `st.popover` (or, on the map, a plain
text pointer) rather than a link that jumps to the Methodology tab:
Streamlit's `st.tabs()` has no supported API to switch tabs from a
click, and an anchor pointing into a different tab's panel lands on an
element hidden by Streamlit's own `display:none`, which a browser can't
usefully scroll to. Delivering the real content on the spot is the
version of "an info icon that links to its entry" that this Streamlit
version can actually do reliably.

## What it found (first pass)

- Land cover, Aug 2025: ~77% dense vegetation (three brightness tiers —
  darkest is very likely conifer/shaded canopy on the ridge), ~12%
  grass/farmland, ~6% water, ~6% built-up/bare. Unlabelled by anything
  except its own spectral signature — real BRP crop names now sit alongside
  it (Land & Crops tab) for the ~half of the AOI that's registered farmland,
  but the unmanaged-forest majority of "dense vegetation" is still a
  spectral best guess, not a validated classification.
- **NDVI, 2005–2026 (`ndvi_trend.py`):** net -0.157 over 21 years, but read
  the shape, not that slope — this series is noisy by construction (one
  least-cloudy August date per year, phenology and weather both riding
  along with the day it happened to land on) and mixes two sensors: Landsat
  5/7/8 (30m) through 2017, Sentinel-2 (10m) from 2018. **2018 is this
  series' single lowest point (0.532)** — which lines up with the
  documented, severe 2018 Northwestern-Europe drought, corroborating
  evidence the pipeline is tracking something real rather than an artifact.
  2024→2025 alone still browned slightly (0.676→0.655, 28.6% of the AOI
  browning >0.05 NDVI against 17.8% greening) — worth checking against KNMI
  rainfall before reading anything into one year-pair.
  **Two real bugs surfaced building this, both fixed, both worth naming:**
  (a) `preprocess.py` was applying Sentinel-2's baseline-04.00 (+1000 DN)
  reflectance offset unconditionally; scenes from before that 25 Jan 2022
  processing-baseline change never had the offset in the first place, so
  2018-2021 initially came back inflated to 0.81-0.90 NDVI — a jump landing
  exactly on the baseline-change date, not anywhere a growing season would
  produce it. Fixed by stamping each scene's real baseline onto the raster
  at fetch time and applying the offset only when it's ≥ 04.00.
  (b) The first Landsat pass used a wider fetch window (mid-July to
  mid-September) than Sentinel-2's August-only one, so about half the
  Landsat years landed in September — a second, smaller step right at the
  2017/2018 sensor handover that looked like a calibration issue. A direct
  same-month cross-check (Landsat 2017-08-25 vs Sentinel-2 2017-08-29, 4
  days apart: 0.650 vs 0.640) confirmed the two sensors actually agree
  closely; the fix was narrowing Landsat to August-only, same as
  Sentinel-2, for a genuine apples-to-apples series.
- SAR and optical independently estimate almost the same total water area
  (5.0% vs 5.6%) with 59.2% spatial agreement (IoU) — two different sensors,
  two different methods, roughly the same answer. The gap is plausibly
  narrow-channel speckle noise and a few days' date offset, not real
  disagreement, but that's a guess, not something this pass verified.
- **Flood extent, documented Jan 2024 high water (`flood_event.py`):**
  Lobith (the reference Rhine gauge just upstream) peaked at 14.5–14.7 m NAP
  on Jan 8–9, 2024 — a real, reported ~1-in-5-year event
  ([Rijkswaterstaat](https://www.rijkswaterstaat.nl/nieuws/archief/2024/01/waterstanden-rijn-en-maas-stijgen-nog-steeds-hoge-waterstanden-op-ijsselmeer-en-markermeer),
  [Omroep Gelderland](https://www.gld.nl/nieuws/2296171/waterstand-rijn-bij-lobith-hoger-dan-verwacht-zand-sijpelt-weg-aan-waalkade-in-nijmegen)).
  The first pass (one fixed -17dB threshold, one post-peak date) read this as
  essentially no change: 2.1% of the AOI dry→wet vs 2.3% wet→dry. Replacing
  that with change detection against a per-pixel reference composite (median
  backscatter over four ordinary-condition autumn-2023 passes) and a
  five-date series through the event (no scene fully covers the AOI on the
  peak day itself, confirmed against the STAC catalogue directly) finds a
  real signal: 9.1% flagged as flooded pre-event → 21.2% at peak (Jan 16,
  four days into the river's own recession), net **+12.1 points**. The
  pre-event reading isn't 0%, so there's real background noise in this method
  too (speckle, seasonal backscatter drift between the reference dates and
  the event window) — read the *change* as the signal, not the absolute
  level. The floodplain's own peak landing after the river's suggests the
  Ooijpolder/Millingerwaard washland filling and draining on its own delay
  rather than tracking river stage instantly, which is a plausible, real
  hydrological explanation this series can show — not something either
  method could have distinguished from "no real flooding" on a single date.
- AHN confirms the terrain profile the concept note sketched by hand:
  elevation actually runs 8.0–95.1 m NAP across the municipality (river
  floodplain to ridge crest), and canopy/roofline over 15 m covers only 1.1%
  of the ground — consistent with a landscape that's mostly open polder and
  low-rise villages plus one wooded ridge.

## Not done yet (see the concept note's phasing)

- No BRP/BGT ground-truth join for the *unmanaged* land — BRP only covers
  registered agricultural parcels (see Land & Crops), so unmanaged forest on
  the ridge is still an unvalidated KMeans guess, not ground-truthed.
- The change-detection flood series is a real improvement over the old
  fixed-threshold read, but still isn't validated against anyone's
  independent measurement — Waterschap Rivierenland's own gauge/extent data
  is the concept note's recommended next step before calling it trustworthy.
- The NDVI trend is one date per year, not a dense time series — each point
  still carries whatever that single scene's conditions were, and the
  within-August day-of-month drifts year to year.
- Nothing is validated against ARK Nature's own ground data — the concept
  note's other recommended starting point, alongside Waterschap
  Rivierenland above.

### A real NH3 number, and land cover colours/icons that mean something

Two more direct-feedback fixes:

**NH3 concentration is now real data (`src/fetch_air_quality.py`).**
The dashboard used to only explain that RIVM's Atlas Leefomgeving WCS has
no NH3 layer — true, checked again live against its capabilities list —
but that same investigation found RIVM separately publishes NH3
*concentration* as an open, no-key-needed grid download: **GCN**
(Grootschalige Concentratiekaarten Nederland, `data.rivm.nl/data/gcn/`),
an Esri ASCII grid in RD-New, same 1x1km resolution and OPS-pro modelling
family as the NO2/PM10/PM2.5 grids already used here. `_fetch_nh3()`
downloads it, assigns the CRS from the product's own metadata PDF (the
.asc format carries none itself), clips to the municipal polygon the same
way the WCS-based pollutants are, and reads both real published years
(2024, 2025 — GCN's 2030/2035/2040 are policy-scenario projections, not
history, and are deliberately not fetched as if they were real). Current
result: mean 5.28 ug/m3 (2025), +0.50 vs 2024. This is **concentration**,
not **deposition** (mol N/ha/yr, the AERIUS/GDN figure Dutch farm-nitrogen
permitting actually runs on) — that distinction is carried through the
card, its ⓘ methodology entry, and the Environment & Energy tab's own
explanatory box, not glossed over now that a real NH3 number exists.

**Land-cover colours and icons now mean something, everywhere they're
shown (`src/visualize.py`, `dashboard.py`).** `LANDCOVER_COLORS` gave
"Built-up / bare" and "Grass / farmland" almost the same tan/brown hue —
hard to tell apart on the map or in a bar chart. Redrawn so each class
comes from its own colour family (water=blue, built-up=neutral grey,
farmland=gold, the three canopy-brightness tiers=this app's own forest
green, light to dark). Each class also gets a small hand-drawn line icon
(`landcover_icon_svg()` — a building, grass blades, a tree, a wave, a
leaf; the same Feather/Lucide-style single-stroke convention as the crop
icons, not a photo or a generic AI-generated image) shown next to its
colour swatch on both the interactive map's own legend and a new
`landcover_class_chart()` Altair chart that replaced a plain
default-blue `st.bar_chart` on the Overview tab. A `LANDCOVER_LABELS_NL`
lookup translates the fixed English cluster names `landcover_ml.py`
produces (that stage labels clusters from their own spectral signature,
not UI text, so it has no bilingual convention of its own) everywhere
these names are displayed, so the legend and the chart read in Dutch
when the page does. `LANDCOVER_TREND_COLORS` (the separate, coarser
4-category year-over-year trend) was also re-toned from bright
"default chart" hues to the same muted palette family.

### Business case: NH3 reframed, and the roadmap now includes ML and drone data

The Business case tab's nitrogen-permitting pitch referenced NH3/AERIUS
as entirely unavailable; updated to reflect what's now true — real NH3
*concentration* is delivered today, *deposition* remains the one
open item, still requiring a data-sharing conversation with RIVM/AERIUS.
The roadmap ("gaps that are also the roadmap") gained two genuinely new,
honestly-scoped items: higher-resolution height/canopy data via a
commissioned drone (UAV LiDAR/photogrammetry) survey — AHN is real
LiDAR but nationally flown on a multi-year cycle, and no such
commissioned dataset exists for this municipality today, stated as a
future option rather than data this pipeline already has; and ML beyond
the current (real, non-black-box) OLS trend and Markov rotation models —
a per-field NDVI-anomaly early-stress detector and a land-cover
classifier trained on this pipeline's own multi-year KMeans output plus
BRP labels, neither built yet.

### A real print/export report (`src/report.py`)

The sidebar's "Print this tab" button only ever triggered the browser's
own print dialog on whatever tab happened to be open -- no choice of
what's on the page, and no real legend/scale bar baked in reliably
across browsers. `src/report.py` builds an actual multi-page PDF
instead, in the same spirit as a QGIS/ArcGIS print composer: a sidebar
checklist (`REPORT_LAYER_CHOICES`) lets a person pick which real layers
to include (true colour, NDVI, NDVI change, land cover, SAR, SAR water
mask, flood extent, elevation, canopy height, NO2, BRP fields), and
`build_report_pdf()` renders one page per layer -- the real raster
(reusing the exact PNG-generation functions `make_map()` already uses
for the live map, not a fresh render path that could drift from what
the map actually shows), a title/subtitle block, a legend that matches
what's actually on the page (coloured swatches for categorical layers
like land cover/crops, a real colour-ramp bar with the *actual*
min/max the PNG was stretched to for continuous ones like NDVI/SAR),
a north arrow, and a scale bar computed from that page's own real
geographic extent (1° longitude ≈ 111.32·cos(latitude) km at this
municipality's latitude, not a fixed bar reused regardless of zoom).
Returned as bytes straight into `st.download_button` -- no temp file
left on disk. Two real bugs caught by actually rendering the PDF and
looking at it, not just running the code: the categorical legend's
"square" swatches rendered as tall thin bars (an axes-fraction
Rectangle sized without accounting for that column's own aspect
ratio -- fixed by switching to a fixed-size marker instead), and the
cover page's table of contents could overlap its own footer when many
layers were selected (a fixed per-line step regardless of item count --
fixed by computing the step from how many lines actually need to fit).
