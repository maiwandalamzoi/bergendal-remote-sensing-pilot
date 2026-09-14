# Methodology

One entry per indicator on the Berg en Dal dashboard: what it measures, the exact source and processing, and its known limitations — read directly from the code that computes it (`src/methodology.py` is the single source of truth this file is generated from; run `python src/methodology.py` after editing that file to regenerate this one).

Every number below reflects this repository's state when generated; see each entry's own "Code" line for where to check the live logic, and `data/processed/stats.json` for the live current value.

## Population

**Measures:** Total registered residents of the municipality.

**Source:** CBS (Statistics Netherlands) via PDOK WFS 'wijkenbuurten', feature type wijkenbuurten:gemeenten, property aantalInwoners, gemeentecode GM1945.

**Date/period:** 2024 (the WFS endpoint itself is published as .../2024/wfs/v1_0; CBS's own reference date for this specific figure).

**Resolution:** Municipality-wide aggregate -- one number for the whole gemeente, no spatial grid.

**Processing:**
- Live GET to https://service.pdok.nl/cbs/wijkenbuurten/2024/wfs/v1_0 (service=WFS, version=2.0.0, request=GetFeature, typeName=wijkenbuurten:gemeenten, outputFormat=application/json).
- Filter the returned FeatureCollection to the one feature with gemeentecode == 'GM1945'.
- Read the aantalInwoners property directly -- no computation or aggregation performed here.
- CBS's own suppression sentinel (-99997) is converted to null (not triggered at municipality scale; it exists for small-area breakdowns).

**Limitations:**
- A registry snapshot at CBS's own 2024 publication date, not a live count -- the real population has grown since (the Villages tab's own growth figure and the Forecast tab's population projection both use this same series and will show later/derived numbers that legitimately disagree with this exact one).
- No margin of error is published by CBS for this field; treated here as exact.

**Code:** `src/fetch_cbs.py`

---

## Households

**Measures:** Total number of households registered in the municipality.

**Source:** CBS via the same PDOK WFS as Population, property aantalHuishoudens.

**Date/period:** 2024 (same WFS publication as Population).

**Resolution:** Municipality-wide aggregate.

**Processing:**
- Identical fetch/filter as Population (one shared WFS response supplies every CBS field this pipeline reads); aantalHuishoudens is read from the same feature.

**Limitations:**
- Same 2024 registry-vintage caveat as Population.

**Code:** `src/fetch_cbs.py`

---

## Registered farmland (BRP)

**Measures:** Total area of parcels farmers registered for subsidy this year.

**Source:** RVO/PDOK BRP (Basisregistratie Gewaspercelen), WFS feature type brpgewaspercelen:BrpGewas, live current-year registry (no historical WFS -- see Crop rotation's own 5-year GeoPackage-archive method for years before this one).

**Date/period:** 2025 (the registry's own 'jaar' field on each returned feature).

**Resolution:** Per-parcel vector polygons (exact registered boundaries, not a raster).

**Processing:**
- Page through the WFS in 1,000-feature pages (service=WFS, version=2.0.0, request=GetFeature, bbox filter in EPSG:28992) over the municipal bounding box.
- Clip the result to the exact municipal polygon with geopandas.clip (the bbox fetch over-fetches a rectangle; parcels straddling the boundary are cut to the real line).
- area_ha = geometry.area / 10,000, computed in RD New (EPSG:28992, a projected metric CRS) -- a true planar area, not a geographic-degree approximation.
- total_area_ha = sum of every parcel's area_ha; by_category_ha and top_crops_ha are the same sum grouped by the registry's own 'category'/'gewas' fields.

**Limitations:**
- Only parcels a farmer actually registered for subsidy -- unmanaged forest, most private gardens, and non-agricultural land are not in BRP at all, so this is not "all vegetated land," it's specifically the subsidy-registered farmland footprint (compare against the KMeans land cover's own, broader vegetation estimate on Land & Crops, which measures a different thing on purpose).

**Code:** `src/fetch_brp.py`

---

## Land area

**Measures:** Land area of the municipality, excluding inland water.

**Source:** CBS via the same PDOK WFS as Population, property oppervlakteLandInHa.

**Date/period:** 2024.

**Resolution:** Municipality-wide aggregate.

**Processing:**
- Identical fetch/filter as Population; oppervlakteLandInHa read directly from the same feature. The companion field oppervlakteWaterInHa (689 ha -- the Rhine/Waal floodplain inside the municipal boundary) is fetched in the same response but shown separately (the Overview card's own 'More' popover), not added in.

**Limitations:**
- A CBS registry figure, not independently re-measured from the municipal boundary polygon this pipeline also holds (src/aoi.py) -- the two should agree closely since both ultimately derive from Kadaster/CBS boundary data, but that agreement hasn't been explicitly cross-checked in code.

**Code:** `src/fetch_cbs.py`

---

## Homes with solar

**Measures:** Share of homes with rooftop solar generation.

**Source:** CBS via the same PDOK WFS as Population, property percentageWoningenMetZonnestroom.

**Date/period:** 2024.

**Resolution:** Municipality-wide aggregate.

**Processing:**
- Identical fetch/filter as Population; read directly, no computation.

**Limitations:**
- CBS's own definition of "has solar" (metered feed-in registered) is not restated in this pipeline's code -- taken as CBS defines it.

**Code:** `src/fetch_cbs.py`

---

## NDVI (vegetation index)

**Measures:** Vegetation greenness/vigour per pixel, from red and near-infrared reflectance.

**Source:** Sentinel-2 L2A, via Microsoft Planetary Computer's STAC catalog (collection sentinel-2-l2a), bands B04 (red) and B08 (NIR); Landsat 5/7/8 Collection 2 (30m) fills 2005-2017 where Sentinel-2 coverage isn't reliable yet.

**Date/period:** 2005-present (one value per year, the least-cloudy date found in a given window). Current headline figure (Overview/Land & Crops): August 2025, mean NDVI 0.655 (processing baseline 05.11).

**Resolution:** 10m (Sentinel-2 native); 30m (Landsat, 2005-2017).

**Processing:**
- NDVI = (NIR - RED) / (NIR + RED), both bands first converted from digital numbers to surface reflectance: refl = (DN + offset) / 10000, clipped to [0, 1].
- offset = -1000 applied only when the scene's own processing-baseline tag is >= 04.00 (25 Jan 2022) -- baseline >= 04.00 scenes store DNs with a +1000 additive offset so reflectance never goes negative; earlier scenes never had it applied and would be wrongly deflated if the offset were subtracted anyway. A real bug found here: an earlier version applied the offset unconditionally, producing a false ~0.2 NDVI step exactly at the baseline-04.00 boundary (2021->2022) that had nothing to do with a real growing season.
- Cloud/shadow masking via the Scene Classification Layer (SCL): only codes {2,4,5,6,7,11} count as clear ground; 0 (no-data), 1 (saturated), 3 (cloud shadow), 8/9 (cloud), 10 (cirrus) are excluded from every mean.
- One date per year: the least-cloudy Sentinel-2 scene in a search window, filtered by whole-scene eo:cloud_cover. The two current headline dates (Aug 2025, Aug 2024, used on the Overview map and for NDVI change) were fetched at max_cloud=5%; other trend years (2018-2023, fetched only if not already on disk) use max_cloud=30% for Sentinel-2, or max_cloud=50% (Landsat, 2005-2017, a whole-scene estimate only -- the per-pixel QA_PIXEL clear-bit mask, not this ceiling, is what actually excludes cloud from the mean).
- Both Sentinel-2 and Landsat years are searched within an August-only date window -- an earlier version used a wider mid-July-to-September window for Landsat and produced a second real bug: about half the Landsat years landed in September while every Sentinel-2 year was August, creating a false trend step at the sensor handover that a same-month cross-check (two scenes 4 days apart, one per sensor) showed wasn't a real signal.

**Limitations:**
- One snapshot day per year, not a seasonal integral -- read the shape of the 22-year series, not any single year-to-year jump, as this dashboard's own Trends & Climate tab already says.
- Two different sensors at two different resolutions (10m/30m) share one series; the handover year (2017->2018) is the single most cross-checkable point and was validated once (see the window-mismatch bug above), not on an ongoing basis.
- Cloud-cover ceiling differs by year (5%/30%/50%) depending on when that year happened to be fetched -- the per-pixel clear mask is the real quality control, but a year fetched at the 30%/50% ceiling had fewer candidate scenes to pick the least-cloudy one from.

**Open questions:**
- The exact day-of-month chosen for each year's scene is stamped on that raster's own file (data/raw/summer_{year}.tif) but not persisted into stats.json -- the Methodology tab can state the search window and cloud ceiling, not the literal calendar date, without re-reading every raster's tags live.

**Code:** `src/preprocess.py, src/ndvi_trend.py, src/fetch_sentinel2.py, src/fetch_landsat.py, src/preprocess_landsat.py`

---

## NDVI change

**Measures:** Pixel-level vegetation change between two dates.

**Source:** Derived from the same two Sentinel-2 NDVI rasters as the NDVI entry above (summer_2024 -> summer_2025), no new fetch.

**Date/period:** August 2024 -> August 2025.

**Resolution:** 10m.

**Processing:**
- delta = NDVI(2025) - NDVI(2024), computed only where both dates' clear-ground mask is valid; pixels invalid in either date are set to NaN, not zero (so a cloud in one year doesn't register as "vegetation loss").
- greening = share of valid pixels where delta > +0.05; browning = share where delta < -0.05 (a fixed absolute threshold on the index itself, not adaptive to the AOI's own noise level).
- Current result: mean delta -0.022, 17.8% of clear ground greening, 28.6% browning.

**Limitations:**
- Inherits every NDVI limitation above (one snapshot day per date, not a seasonal integral) -- a two-year delta is even more exposed to single-day weather than the 22-year trend is.
- The +-0.05 greening/browning threshold is a round number, not derived from this AOI's own pixel-to-pixel noise floor.

**Code:** `src/landcover_ml.py (function ndvi_change)`

---

## Land cover (KMeans)

**Measures:** Unsupervised classification of ground into water/built-up/farmland/vegetation-density classes.

**Source:** Derived from the same Sentinel-2 reflectance/index bands as NDVI, no new fetch; clustering via scikit-learn's KMeans.

**Date/period:** One classification per year, 2018-present (Sentinel-2 era); see the Forest & land cover change entry for the 2018-2026 spatial comparison this also feeds.

**Resolution:** 10m.

**Processing:**
- Seven features per pixel: NDVI, NDWI, red/NIR/green/blue reflectance, and brightness (mean of red+green+blue) -- standardized (zero mean, unit variance) before clustering so no one feature's raw scale dominates the distance metric.
- KMeans(n_clusters=6, random_state=42, n_init=10) -- a fixed cluster count and a fixed random seed (reproducible, not re-tuned per year), 10 random initializations kept.
- Clusters are named from their own centroid signature *relative to the other clusters in that run*, not fixed absolute thresholds: water is the cluster with the largest (NDWI - NDVI) among clusters with positive NDWI; built-up/bare is the brightest remaining cluster if it also sits below the median NDVI of what's left; grass/farmland is the lowest-NDVI vegetated cluster if a real gap (>0.1 NDVI) separates it from the rest; everything left is ranked by brightness into up to three "dense vegetation" tiers (dark/mid/bright canopy). A fixed NDVI cutoff was tried first and rejected: August vegetation across this AOI is uniformly high-NDVI, so a threshold rule collapsed three spectrally distinct clusters into one label.

**Limitations:**
- Fully unsupervised -- no BRP/BGT ground-truth join yet, so class names are a spectral best guess, not validated against a labelled reference (stated on the dashboard itself on multiple tabs).
- Built-up/bare and grass/farmland sit closer together in spectral space than in reality; bare or just-harvested cropland can get relabelled between them from one year's centroid fit to the next -- named explicitly as the least stable category, versus water/forest which are more spectrally distinct and more trustworthy.
- A hard 6-cluster count can't resolve species-level forest type or sub-classes a satellite with only 4 optical bands and no SWIR wouldn't be able to distinguish anyway.

**Code:** `src/landcover_ml.py (functions classify, _label_clusters)`

---

## SAR backscatter (VV)

**Measures:** Radar reflectivity of the ground -- rough/urban surfaces bright, smooth/water dark.

**Source:** Sentinel-1 RTC (radiometrically terrain-corrected), via Microsoft Planetary Computer's STAC catalog (collection sentinel-1-rtc), VV + VH polarizations, already calibrated to linear gamma0.

**Date/period:** August 2025 (sar_2025, the Overview map's baseline date).

**Resolution:** Not explicitly verified in this pipeline's own code -- see the open question below.

**Processing:**
- VV/VH read as linear gamma0 directly from the STAC asset (no cloud to wait out -- radar).
- Converted to dB: 10 * log10(linear value), only where the linear value is > 0.
- Native grid is UTM zone 32N (EPSG:32632) -- Berg en Dal sits almost exactly on the 6°E UTM zone boundary, so this is a different zone from the Sentinel-2 optical mosaic's own 31N grid. Kept in its native grid at fetch time; reprojected onto the optical grid only where the two are directly compared pixel-to-pixel (the SAR/optical water cross-check).
- Scene selection requires the real footprint (a diagonal parallelogram, not the bounding box) to fully *contain* the AOI polygon, not just intersect its bbox -- Berg en Dal sits close to a swath edge on some passes, where bbox-only filtering would silently accept a scene that only clips one corner of the municipality.

**Limitations:**
- One date, no multi-temporal averaging for this specific layer (unlike the flood-event change-detection product, which does build a multi-date reference composite).
- RTC (radiometric terrain correction) reduces but doesn't eliminate real terrain effects (layover/shadow on the AOI's own ridge slopes) -- not separately quantified here.

**Open questions:**
- Native pixel spacing of Planetary Computer's sentinel-1-rtc product is not asserted or checked anywhere in this pipeline's code (commonly 20m for this specific product, but that number does not appear in src/fetch_sentinel1.py or src/preprocess_sar.py, so it's listed here as unverified rather than stated as fact).

**Code:** `src/fetch_sentinel1.py, src/preprocess_sar.py`

---

## SAR water mask

**Measures:** Open water, detected from how little radar energy a surface reflects back.

**Source:** Derived from the same Sentinel-1 VV backscatter as the entry above, no new fetch.

**Date/period:** August 2025.

**Resolution:** Same as SAR backscatter (see open question there).

**Processing:**
- A pixel is flagged water where VV_dB < -17.0 dB (a fixed threshold): smooth open water is a near-specular reflector at C-band, so it returns very little energy to the sensor.
- The -17 dB cutoff was picked as consistent with the ~5-6% water fraction Sentinel-2's own NDWI/KMeans pass independently found for this AOI -- a plausibility check against a second method, not a value derived from a formal radiometric calibration curve.
- Cross-checked against the optical KMeans 'Water' cluster on the shared clear-ground area: current agreement (IoU, intersection over union) = 59.2% (SAR water 5.0% vs. optical water 5.6% of the same shared pixels).

**Limitations:**
- A fixed absolute dB threshold, the same known weakness the old flood-extent method (below) has -- it doesn't necessarily transfer cleanly to a scene shot under different conditions or from a different orbit, only checked here for internal consistency against the optical estimate, not against ground survey data.
- 59.2% IoU means real disagreement exists at the margins (river-edge vegetation, narrow channels, wet soil that isn't open water) -- not a validated water product on its own.

**Code:** `src/preprocess_sar.py (functions process, cross_check_water)`

---

## Flood extent — fixed threshold (old method)

**Measures:** Where a single high-water SAR scene shows water that a single baseline scene didn't.

**Source:** Derived from two Sentinel-1 scenes' own SAR water masks (see above), no new fetch: sar_2025 (Aug 2025, baseline) and sar_highwater_2024 (Jan 2024, documented Rhine/Waal high water).

**Date/period:** Baseline Aug 2025 vs. one event date, Jan 2024.

**Resolution:** Baseline's own grid; the high-water scene is reprojected (nearest-neighbour) onto it.

**Processing:**
- Each date's water mask (VV_dB < -17dB) computed independently, then compared per pixel: 0 = dry both dates, 1 = water both dates (permanent water), 2 = newly flooded (dry->wet), 3 = baseline water the high-water scene missed (wet->dry).
- Current result: baseline water 5.0%, high-water-scene water 4.7%, newly flooded 2.1%, "newly dry" 2.3% -- net change effectively zero despite this being a documented flood event.

**Limitations:**
- This module's own code prints an explicit warning when "newly dry" exceeds "newly flooded" on a documented flood date (exactly what happens here): "the fixed -17dB threshold doesn't transfer cleanly across scenes shot from different orbit directions/conditions... treat this pass as a pipeline proof, not a validated flood map." stats.json's own threshold_reliable field is set to False for this result.
- Kept in the dashboard for direct comparison against the change-detection method below, specifically *because* it fails -- not presented as a usable flood product on its own.

**Code:** `src/preprocess_sar.py (function flood_extent)`

---

## Flood extent — change detection (new method)

**Measures:** Where the ground dropped well below its own normal radar reflectivity during the flood.

**Source:** Sentinel-1 RTC, 9 scenes total: 4 reference dates + 5 event-window dates, via Planetary Computer (same collection as SAR backscatter above).

**Date/period:** Reference: 4 autumn-2023 dates (2023-11-02, -11-14, -11-26, -12-08), ~12 days apart. Event window: 2023-12-23 (pre-event), 2024-01-04 (rise), 2024-01-13/16/25 (recession, early/mid/late) -- bracketing the documented Lobith gauge peak (14.5-14.7m NAP, 8-9 Jan 2024).

**Resolution:** Reference composite's own grid (the first reference date's grid; the other three reference dates and every event date are reprojected onto it, bilinear).

**Processing:**
- Per-pixel reference level = median VV dB across the 4 reference dates (median, not mean, so one anomalous pass doesn't skew the 'normal' baseline) -- computed only where at least one reference date actually covers that pixel.
- For each event date: delta = that date's VV dB (reprojected onto the reference grid) minus the per-pixel reference level.
- A pixel is flagged flooded where delta < -3.0 dB -- a threshold taken from the SAR flood-mapping literature (e.g. Twele et al. 2016), not tuned against this AOI's own data.
- No Sentinel-1 scene fully covering the AOI exists on the documented peak day itself (checked directly against the STAC catalogue) -- the nearest fully-covering passes on either side stand in for it.
- Current result: 9.1% flooded pre-event -> 21.2% at peak (13 Jan 2024, early recession), net +12.1 percentage points -- a clearly-above-noise signal, unlike the old method's ~0 net change on the same underlying event.

**Limitations:**
- Pre-event (9.1%) is not 0% -- this method has real background noise (speckle, seasonal moisture drift between the autumn-2023 reference dates and the January event window), so the *change* is the signal, not the absolute level.
- The -3dB threshold is a literature convention, not locally calibrated or validated against an independent ground/gauge survey of the actual flood extent.
- Peak lands a few days after Lobith's own gauge peak -- read here as a real floodplain-filling delay, not cross-checked against an independent hydrological model.

**Code:** `src/flood_event.py`

---

## Elevation (AHN DTM)

**Measures:** Bare-earth ground height, with buildings and trees stripped out.

**Source:** AHN (Actueel Hoogtebestand Nederland), the Dutch national LiDAR archive, via PDOK's public WCS, coverage dtm_05m.

**Date/period:** Not stamped with an acquisition date in this pipeline's own output -- AHN is a rolling national resurvey; see the open question below.

**Resolution:** Requested at 5m output grid (the WCS 'scalesize' parameter) directly from the service -- native LiDAR point-cloud resolution is 0.5m, not downloaded at that resolution and resampled locally.

**Processing:**
- GetCoverage request to https://service.pdok.nl/rws/ahn/wcs/v1_0 (WCS 2.0.1), coverageId dtm_05m, subset to the municipal bounding box + 250m pad, scalesize computed from that bbox at 5m/px.
- Clipped to the exact municipal polygon (rasterio.mask, already in the WCS's native CRS, RD New/EPSG:28992).
- Current result: 8.0-95.1m NAP across the municipality (mean 31.2m) -- the real range from the Waal floodplain to the Nijmegen ridge (stuwwal).

**Limitations:**
- 5m output resolution genuinely smooths sub-5m terrain features present in the native 0.5m archive.

**Open questions:**
- This dashboard's own captions elsewhere label the source "AHN4," but src/fetch_ahn.py never checks or asserts a version number against the WCS response (e.g. via GetCapabilities metadata) -- PDOK's current dtm_05m/dsm_05m coverage is AHN4 nationwide as of recent years, but that specific claim is not verified in this pipeline's own code and should be confirmed against the service's own capabilities document rather than assumed.
- No per-fetch acquisition date is recorded in stats.json for this layer (AHN acquisition varies by flight strip, not a single date for the whole municipality) -- listed here as open rather than guessed.

**Code:** `src/fetch_ahn.py`

---

## Canopy / building height (AHN nDSM)

**Measures:** Height of whatever the laser hit first, above the bare ground beneath it.

**Source:** Derived from the same AHN WCS as the DTM entry above (coverage dsm_05m, first-return height), no new fetch beyond the DSM/DTM pair.

**Date/period:** Same as AHN DTM.

**Resolution:** 5m (same grid as DTM/DSM).

**Processing:**
- nDSM = clip(DSM - DTM, 0, None) -- clipped at zero because DSM should never read below DTM on real ground; small negative differences from grid/interpolation noise are floored rather than shown as "negative height."
- A pixel invalid in either the DTM or the DSM is invalid in the nDSM too (nodata propagates from both inputs).
- Current result: mean 1.9m across the municipality, 1.1% of ground reads above 15m (mature forest canopy or tall buildings).

**Limitations:**
- Doesn't distinguish tree canopy from buildings -- both read as "first-return height above ground"; a separate classification (e.g. against BAG building footprints) would be needed to split the two, and isn't done here.
- Same 5m-smoothing limitation as the DTM it's built from.

**Code:** `src/fetch_ahn.py (function fetch_all)`

---

## Air quality — NO2

**Measures:** Modelled annual-mean nitrogen dioxide concentration at ground level.

**Source:** RIVM Atlas Leefomgeving WCS, coverage alo__rivm_nsl_20260401_gm_NO22024 -- the same modelled grids used in official Dutch NSL (Nationaal Samenwerkingsprogramma Luchtkwaliteit) policy reporting, not a satellite product.

**Date/period:** 2024 (the coverage's own name encodes both the data year, 2024, and this particular publication/refresh of the grid, 2026-04-01).

**Resolution:** 1x1 km.

**Processing:**
- GetCoverage request to https://data.rivm.nl/geo/alo/wcs (WCS 2.0.1), clipped to the exact municipal polygon (rasterio.mask).
- A physical sanity bound is applied before averaging: valid = (value != declared_nodata) AND (-1000 < value < 1000). This guards against a real bug found in the equivalent trend fetcher's older-year coverages: some declare nodata=0.0 in their own GDAL profile but actually fill nodata pixels with float32's most-negative value (~-3.4e38) -- excluding only the *declared* nodata value would let that sentinel corrupt the mean to -inf. Applied here too as a matter of consistency, whether or not this specific 2024 coverage turns out to need it.
- mean/min/max computed over the valid pixels only; pct_area_over_who_guideline = share of valid pixels above the WHO 2021 annual guideline (10 ug/m3 for NO2).
- Current result: mean 9.5 ug/m3, range 7.8-16.5 ug/m3, 19.7% of the municipality's area over the WHO guideline.

**Limitations:**
- A national dispersion model output (RIVM's own NSL grids), not a direct sensor measurement at any point in Berg en Dal -- authoritative for policy purposes, but modelled.
- Does not include NH3/nitrogen deposition, the figure Dutch farm-nitrogen permitting actually runs on -- confirmed absent from this specific WCS (checked directly against its capabilities), not simply unfetched; that figure lives in RIVM's separate GDN/AERIUS product as an annual grid download, not a queryable service.

**Code:** `src/fetch_air_quality.py`

---

## Air quality — NH3 (ammonia) concentration

**Measures:** Modelled annual-mean ammonia concentration at ground level -- not nitrogen deposition.

**Source:** RIVM GCN (Grootschalige Concentratiekaarten Nederland), file conc_NH3_<year>.zip, data.rivm.nl/data/gcn/ -- a real, open, directly downloadable grid, distinct from the RIVM Atlas Leefomgeving WCS used for NO2/PM10/PM2.5 above, which was checked directly and confirmed to carry no NH3 or deposition coverage at all.

**Date/period:** 2025 (most recent real year); 2024 kept alongside it for a one-year delta. RIVM's GCN only publishes two real (non-projected) years for NH3 -- 2024 and 2025 -- plus policy-scenario projections for 2030/2035/2040, which are deliberately not fetched here as if they were real historical data.

**Resolution:** 1x1 km.

**Processing:**
- Direct HTTP download of conc_NH3_<year>.zip (no key, no WCS/WFS -- a static file), extracting the Esri ASCII grid (.asc) inside.
- The .asc format carries no CRS of its own; RIVM's own metadata PDF for this product states RD-New (EPSG:28992) explicitly, matching every other RIVM/PDOK layer in this pipeline, so the CRS is assigned from the product's documentation, not guessed or left unset.
- Re-written as a GeoTIFF with that CRS assigned, then clipped to the exact municipal polygon (rasterio.mask), same approach as the WCS-based NO2/PM10/PM2.5 fetch.
- mean/min/max computed over valid pixels (value != nodata, sanity-bounded -100 < value < 1000); no WHO or EU guideline exists for NH3 the way it does for NO2/PM10/PM2.5, so no '% over guideline' figure is computed or shown for it.
- Current result: mean 5.28 ug/m3 (2025), 4.78 ug/m3 (2024), range roughly 3.6-9.5 ug/m3 across the municipality -- the spread reflects real local ammonia sources (livestock farming), not noise.

**Limitations:**
- This is ammonia *concentration in air* (ug/m3), not nitrogen *deposition on nature areas* (mol N/ha/yr) -- a different quantity that Dutch farm-nitrogen permitting actually runs on. Deposition is published separately via RIVM's AERIUS/GDN product (aerius.nl), which has no open download equivalent to this one and is still not fetched by this pipeline.
- RIVM's own stated uncertainty for this product is sigma = 20-25% -- wider than a typical sensor measurement, since it is the output of an atmospheric dispersion model (OPS-pro 5.3.1.0) calibrated against real LML/MAN station measurements, not a direct reading at every point.
- Only two real years exist (2024, 2025), so no multi-year trend can be shown for NH3 the way there is for NO2/PM10/PM2.5/EC elsewhere on this tab.

**Code:** `src/fetch_air_quality.py (function _fetch_nh3)`

---
