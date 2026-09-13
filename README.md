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
   below) and eight tabs: Overview, **Field Explorer**, **Villages**, Land
   & Crops, **Trends & Climate**, Environment & Energy, Water, and a
   Business case tab spelling out who'd actually pay for this and what the
   real remaining data gaps are. Genuinely Dutch data values (BRP crop and
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
   boundary *and* zooms/outlines it on the Overview map too
   (`make_map(village=...)` fits the map's bounds to the village polygon
   and draws it as its own toggleable layer) — picking a village in the
   sidebar now visibly moves both maps, not just one (other tabs stay
   municipality-wide, stated explicitly rather than left ambiguous).

   **Map controls & print report** — both Folium maps (Overview and Field
   Explorer) carry a real Leaflet fullscreen control
   (`folium.plugins.Fullscreen`, top-left), and the sidebar has a
   "Print this tab" button that triggers the browser's own print dialog
   (`window.parent.print()`, since the map lives in an `st.components.v1`
   iframe) against dedicated print CSS that hides the sidebar, Streamlit's
   header/toolbar and the tab bar, and stamps the printed page with the
   date and the current village/whole-municipality scope — a plain
   "save the page as PDF" rather than a separate export pipeline.

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
