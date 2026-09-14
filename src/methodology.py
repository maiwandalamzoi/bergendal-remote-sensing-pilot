"""
Single source of truth for how every indicator on this dashboard is
actually computed -- read directly from the code that computes it
(fetch_cbs.py, preprocess.py, landcover_ml.py, preprocess_sar.py,
flood_event.py, fetch_ahn.py, fetch_air_quality.py, ...), not written
from memory or generalized from what a similar product "usually" does.
Where the code itself doesn't pin something down (a product version, an
exact resolution), that's listed as an open question, not guessed.

Powers two things that must never disagree:
  - dashboard.py's Methodology tab (renders this list directly)
  - METHODOLOGY.md (generated from this file: `python src/methodology.py`)

To add or correct an entry: edit ENTRIES below, then re-run this file to
regenerate METHODOLOGY.md. There is no second copy to keep in sync by hand.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MethodEntry:
    id: str
    title: tuple[str, str]  # (en, nl)
    measures: tuple[str, str]
    source: str  # provider/product/version -- proper nouns/URLs, not translated
    date_range: tuple[str, str]
    resolution: tuple[str, str]
    processing: tuple[list[str], list[str]]
    limitations: tuple[list[str], list[str]]
    code_ref: str
    open_questions: tuple[list[str], list[str]] = field(default_factory=lambda: ([], []))


ENTRIES: list[MethodEntry] = [
    MethodEntry(
        id="population",
        title=("Population", "Inwoners"),
        measures=("Total registered residents of the municipality.",
                   "Totaal aantal geregistreerde inwoners van de gemeente."),
        source="CBS (Statistics Netherlands) via PDOK WFS 'wijkenbuurten', feature type "
               "wijkenbuurten:gemeenten, property aantalInwoners, gemeentecode GM1945.",
        date_range=("2024 (the WFS endpoint itself is published as .../2024/wfs/v1_0; CBS's own "
                     "reference date for this specific figure).",
                     "2024 (het WFS-endpoint zelf is gepubliceerd als .../2024/wfs/v1_0; CBS's eigen "
                     "referentiedatum voor dit specifieke cijfer)."),
        resolution=("Municipality-wide aggregate -- one number for the whole gemeente, no spatial grid.",
                    "Gemeentebreed aggregaat -- één getal voor de hele gemeente, geen ruimtelijk raster."),
        processing=([
            "Live GET to https://service.pdok.nl/cbs/wijkenbuurten/2024/wfs/v1_0 (service=WFS, "
            "version=2.0.0, request=GetFeature, typeName=wijkenbuurten:gemeenten, "
            "outputFormat=application/json).",
            "Filter the returned FeatureCollection to the one feature with gemeentecode == 'GM1945'.",
            "Read the aantalInwoners property directly -- no computation or aggregation performed here.",
            "CBS's own suppression sentinel (-99997) is converted to null (not triggered at "
            "municipality scale; it exists for small-area breakdowns).",
        ], [
            "Live GET naar https://service.pdok.nl/cbs/wijkenbuurten/2024/wfs/v1_0 (service=WFS, "
            "version=2.0.0, request=GetFeature, typeName=wijkenbuurten:gemeenten, "
            "outputFormat=application/json).",
            "De teruggegeven FeatureCollection filteren op de ene feature met gemeentecode == 'GM1945'.",
            "De eigenschap aantalInwoners direct uitlezen -- hier vindt geen berekening of "
            "aggregatie plaats.",
            "CBS's eigen onderdrukkingswaarde (-99997) wordt omgezet naar null (komt op "
            "gemeenteniveau niet voor; is bedoeld voor kleine-gebiedsuitsplitsingen).",
        ]),
        limitations=([
            "A registry snapshot at CBS's own 2024 publication date, not a live count -- the real "
            "population has grown since (the Villages tab's own growth figure and the Forecast tab's "
            "population projection both use this same series and will show later/derived numbers that "
            "legitimately disagree with this exact one).",
            "No margin of error is published by CBS for this field; treated here as exact.",
        ], [
            "Een registratie-momentopname op de publicatiedatum van CBS (2024), geen live telling -- de "
            "werkelijke bevolking is sindsdien gegroeid (het groeicijfer op het tabblad Kernen en de "
            "bevolkingsvoorspelling op Voorspelling gebruiken dezelfde reeks en tonen terecht latere/"
            "afgeleide cijfers die van dit exacte getal afwijken).",
            "CBS publiceert geen foutmarge voor dit veld; hier als exact behandeld.",
        ]),
        code_ref="src/fetch_cbs.py",
    ),
    MethodEntry(
        id="households",
        title=("Households", "Huishoudens"),
        measures=("Total number of households registered in the municipality.",
                   "Totaal aantal geregistreerde huishoudens in de gemeente."),
        source="CBS via the same PDOK WFS as Population, property aantalHuishoudens.",
        date_range=("2024 (same WFS publication as Population).", "2024 (zelfde WFS-publicatie als Inwoners)."),
        resolution=("Municipality-wide aggregate.", "Gemeentebreed aggregaat."),
        processing=([
            "Identical fetch/filter as Population (one shared WFS response supplies every CBS field "
            "this pipeline reads); aantalHuishoudens is read from the same feature.",
        ], [
            "Identieke fetch/filter als Inwoners (één gedeelde WFS-respons levert elk CBS-veld dat deze "
            "pipeline uitleest); aantalHuishoudens wordt uit dezelfde feature gelezen.",
        ]),
        limitations=([
            "Same 2024 registry-vintage caveat as Population.",
        ], [
            "Zelfde 2024-registratiekanttekening als bij Inwoners.",
        ]),
        code_ref="src/fetch_cbs.py",
    ),
    MethodEntry(
        id="registered_farmland",
        title=("Registered farmland (BRP)", "Landbouwgrond (BRP)"),
        measures=("Total area of parcels farmers registered for subsidy this year.",
                   "Totale oppervlakte percelen die boeren dit jaar voor subsidie hebben opgegeven."),
        source="RVO/PDOK BRP (Basisregistratie Gewaspercelen), WFS feature type "
               "brpgewaspercelen:BrpGewas, live current-year registry (no historical WFS -- see "
               "Crop rotation's own 5-year GeoPackage-archive method for years before this one).",
        date_range=("2025 (the registry's own 'jaar' field on each returned feature).",
                     "2025 (het eigen 'jaar'-veld van het register op elke teruggegeven feature)."),
        resolution=("Per-parcel vector polygons (exact registered boundaries, not a raster).",
                    "Vectorpolygonen per perceel (exacte geregistreerde grenzen, geen raster)."),
        processing=([
            "Page through the WFS in 1,000-feature pages (service=WFS, version=2.0.0, "
            "request=GetFeature, bbox filter in EPSG:28992) over the municipal bounding box.",
            "Clip the result to the exact municipal polygon with geopandas.clip (the bbox fetch "
            "over-fetches a rectangle; parcels straddling the boundary are cut to the real line).",
            "area_ha = geometry.area / 10,000, computed in RD New (EPSG:28992, a projected metric "
            "CRS) -- a true planar area, not a geographic-degree approximation.",
            "total_area_ha = sum of every parcel's area_ha; by_category_ha and top_crops_ha are the "
            "same sum grouped by the registry's own 'category'/'gewas' fields.",
        ], [
            "Doorlopen van de WFS in pagina's van 1.000 features (service=WFS, version=2.0.0, "
            "request=GetFeature, bbox-filter in EPSG:28992) over de gemeentelijke bounding box.",
            "Het resultaat uitknippen tot de exacte gemeentegrens met geopandas.clip (de bbox-fetch "
            "haalt een rechthoek te veel op; percelen die de grens overschrijden worden op de echte "
            "lijn afgesneden).",
            "area_ha = geometry.area / 10.000, berekend in RD New (EPSG:28992, een geprojecteerd "
            "metrisch CRS) -- een echte planaire oppervlakte, geen geografische-graden-benadering.",
            "total_area_ha = som van elk perceel's area_ha; by_category_ha en top_crops_ha zijn "
            "dezelfde som gegroepeerd op de eigen 'category'/'gewas'-velden van het register.",
        ]),
        limitations=([
            "Only parcels a farmer actually registered for subsidy -- unmanaged forest, most private "
            "gardens, and non-agricultural land are not in BRP at all, so this is not \"all vegetated "
            "land,\" it's specifically the subsidy-registered farmland footprint (compare against the "
            "KMeans land cover's own, broader vegetation estimate on Land & Crops, which measures a "
            "different thing on purpose).",
        ], [
            "Alleen percelen die een boer daadwerkelijk voor subsidie heeft opgegeven -- onbeheerd bos, "
            "de meeste particuliere tuinen en niet-agrarische grond staan helemaal niet in de BRP, dus "
            "dit is niet \"alle begroeide grond,\" maar specifiek de subsidie-geregistreerde "
            "landbouwvoetafdruk (vergelijk met de eigen, bredere vegetatieschatting van de KMeans-"
            "landgebruikskaart op Land & Gewassen, die met opzet iets anders meet).",
        ]),
        code_ref="src/fetch_brp.py",
    ),
    MethodEntry(
        id="land_area",
        title=("Land area", "Landoppervlakte"),
        measures=("Land area of the municipality, excluding inland water.",
                   "Landoppervlakte van de gemeente, exclusief binnenwater."),
        source="CBS via the same PDOK WFS as Population, property oppervlakteLandInHa.",
        date_range=("2024.", "2024."),
        resolution=("Municipality-wide aggregate.", "Gemeentebreed aggregaat."),
        processing=([
            "Identical fetch/filter as Population; oppervlakteLandInHa read directly from the same "
            "feature. The companion field oppervlakteWaterInHa (689 ha -- the Rhine/Waal floodplain "
            "inside the municipal boundary) is fetched in the same response but shown separately "
            "(the Overview card's own 'More' popover), not added in.",
        ], [
            "Identieke fetch/filter als Inwoners; oppervlakteLandInHa rechtstreeks uit dezelfde feature "
            "gelezen. Het bijbehorende veld oppervlakteWaterInHa (689 ha -- de uiterwaarden van Rijn/"
            "Waal binnen de gemeentegrens) komt uit dezelfde respons maar wordt afzonderlijk getoond "
            "('Meer'-popover van de kaart), niet opgeteld.",
        ]),
        limitations=([
            "A CBS registry figure, not independently re-measured from the municipal boundary polygon "
            "this pipeline also holds (src/aoi.py) -- the two should agree closely since both "
            "ultimately derive from Kadaster/CBS boundary data, but that agreement hasn't been "
            "explicitly cross-checked in code.",
        ], [
            "Een CBS-registratiecijfer, niet onafhankelijk opnieuw gemeten uit de gemeentegrenspolygoon "
            "die deze pipeline ook heeft (src/aoi.py) -- de twee zouden goed moeten overeenkomen omdat "
            "beide uiteindelijk uit Kadaster/CBS-grensdata komen, maar die overeenkomst is niet "
            "expliciet in code gecontroleerd.",
        ]),
        code_ref="src/fetch_cbs.py",
    ),
    MethodEntry(
        id="homes_with_solar",
        title=("Homes with solar", "Zonnepanelen"),
        measures=("Share of homes with rooftop solar generation.",
                   "Aandeel woningen met zonnestroomopwekking op het dak."),
        source="CBS via the same PDOK WFS as Population, property percentageWoningenMetZonnestroom.",
        date_range=("2024.", "2024."),
        resolution=("Municipality-wide aggregate.", "Gemeentebreed aggregaat."),
        processing=([
            "Identical fetch/filter as Population; read directly, no computation.",
        ], [
            "Identieke fetch/filter als Inwoners; rechtstreeks uitgelezen, geen berekening.",
        ]),
        limitations=([
            "CBS's own definition of \"has solar\" (metered feed-in registered) is not restated in "
            "this pipeline's code -- taken as CBS defines it.",
        ], [
            "CBS's eigen definitie van \"heeft zonnestroom\" (geregistreerde teruglevering via de "
            "meter) wordt in deze pipeline niet herhaald -- overgenomen zoals CBS het definieert.",
        ]),
        code_ref="src/fetch_cbs.py",
    ),
    MethodEntry(
        id="ndvi",
        title=("NDVI (vegetation index)", "NDVI (vegetatie-index)"),
        measures=("Vegetation greenness/vigour per pixel, from red and near-infrared reflectance.",
                   "Vegetatiegroenheid/-vitaliteit per pixel, uit rode en nabij-infrarode reflectie."),
        source="Sentinel-2 L2A, via Microsoft Planetary Computer's STAC catalog "
               "(collection sentinel-2-l2a), bands B04 (red) and B08 (NIR); Landsat 5/7/8 Collection 2 "
               "(30m) fills 2005-2017 where Sentinel-2 coverage isn't reliable yet.",
        date_range=("2005-present (one value per year, the least-cloudy date found in a given window). "
                     "Current headline figure (Overview/Land & Crops): August 2025, mean NDVI 0.655 "
                     "(processing baseline 05.11).",
                     "2005-heden (één waarde per jaar, de minst bewolkte datum in een bepaald venster). "
                     "Huidig hoofdcijfer (Overzicht/Land & Gewassen): augustus 2025, gemiddelde NDVI 0,655 "
                     "(processing-baseline 05.11)."),
        resolution=("10m (Sentinel-2 native); 30m (Landsat, 2005-2017).",
                    "10m (Sentinel-2, native); 30m (Landsat, 2005-2017)."),
        processing=([
            "NDVI = (NIR - RED) / (NIR + RED), both bands first converted from digital numbers to "
            "surface reflectance: refl = (DN + offset) / 10000, clipped to [0, 1].",
            "offset = -1000 applied only when the scene's own processing-baseline tag is >= 04.00 "
            "(25 Jan 2022) -- baseline >= 04.00 scenes store DNs with a +1000 additive offset so "
            "reflectance never goes negative; earlier scenes never had it applied and would be wrongly "
            "deflated if the offset were subtracted anyway. A real bug found here: an earlier version "
            "applied the offset unconditionally, producing a false ~0.2 NDVI step exactly at the "
            "baseline-04.00 boundary (2021->2022) that had nothing to do with a real growing season.",
            "Cloud/shadow masking via the Scene Classification Layer (SCL): only codes {2,4,5,6,7,11} "
            "count as clear ground; 0 (no-data), 1 (saturated), 3 (cloud shadow), 8/9 (cloud), "
            "10 (cirrus) are excluded from every mean.",
            "One date per year: the least-cloudy Sentinel-2 scene in a search window, filtered by "
            "whole-scene eo:cloud_cover. The two current headline dates (Aug 2025, Aug 2024, used on "
            "the Overview map and for NDVI change) were fetched at max_cloud=5%; other trend years "
            "(2018-2023, fetched only if not already on disk) use max_cloud=30% for Sentinel-2, or "
            "max_cloud=50% (Landsat, 2005-2017, a whole-scene estimate only -- the per-pixel QA_PIXEL "
            "clear-bit mask, not this ceiling, is what actually excludes cloud from the mean).",
            "Both Sentinel-2 and Landsat years are searched within an August-only date window -- an "
            "earlier version used a wider mid-July-to-September window for Landsat and produced a "
            "second real bug: about half the Landsat years landed in September while every Sentinel-2 "
            "year was August, creating a false trend step at the sensor handover that a same-month "
            "cross-check (two scenes 4 days apart, one per sensor) showed wasn't a real signal.",
        ], [
            "NDVI = (NIR - ROOD) / (NIR + ROOD), beide banden eerst omgezet van digitale getallen naar "
            "oppervlaktereflectie: refl = (DN + offset) / 10000, geclipt naar [0, 1].",
            "offset = -1000 alleen toegepast als de eigen processing-baseline-tag van de opname >= "
            "04.00 is (25 jan 2022) -- opnames met baseline >= 04.00 slaan DN's op met een +1000 "
            "additieve offset zodat reflectie nooit negatief wordt; eerdere opnames hadden dit nooit "
            "toegepast en zouden ten onrechte verlaagd worden als de offset toch werd afgetrokken. Een "
            "echte bug hier gevonden: een eerdere versie paste de offset onvoorwaardelijk toe, wat een "
            "valse NDVI-stap van ~0,2 opleverde precies op de baseline-04.00-grens (2021->2022), niets "
            "te maken met een echt groeiseizoen.",
            "Wolk-/schaduwmaskering via de Scene Classification Layer (SCL): alleen codes "
            "{2,4,5,6,7,11} gelden als onbewolkte grond; 0 (geen data), 1 (verzadigd), 3 "
            "(wolkschaduw), 8/9 (wolk), 10 (cirrus) worden uit elk gemiddelde uitgesloten.",
            "Eén datum per jaar: de minst bewolkte Sentinel-2-opname in een zoekvenster, gefilterd op "
            "hele-scène eo:cloud_cover. De twee huidige hoofddata (aug. 2025, aug. 2024, gebruikt op de "
            "Overzicht-kaart en voor NDVI-verandering) zijn opgehaald bij max_cloud=5%; andere "
            "trendjaren (2018-2023, alleen opgehaald als nog niet op schijf) gebruiken max_cloud=30% "
            "voor Sentinel-2, of max_cloud=50% (Landsat, 2005-2017, slechts een hele-scène-schatting -- "
            "het per-pixel QA_PIXEL-heldere-bit-masker, niet dit plafond, sluit wolken daadwerkelijk "
            "uit van het gemiddelde).",
            "Zowel Sentinel-2 als Landsat-jaren worden alleen binnen een augustus-venster gezocht -- "
            "een eerdere versie gebruikte een breder half-juli-tot-september-venster voor Landsat en "
            "produceerde een tweede echte bug: ongeveer de helft van de Landsat-jaren viel in "
            "september terwijl elk Sentinel-2-jaar augustus was, wat een valse trendstap creëerde bij "
            "de sensorovergang die een gelijke-maand-controle (twee opnames 4 dagen uit elkaar, één "
            "per sensor) liet zien geen echt signaal was.",
        ]),
        limitations=([
            "One snapshot day per year, not a seasonal integral -- read the shape of the 22-year "
            "series, not any single year-to-year jump, as this dashboard's own Trends & Climate tab "
            "already says.",
            "Two different sensors at two different resolutions (10m/30m) share one series; the "
            "handover year (2017->2018) is the single most cross-checkable point and was validated "
            "once (see the window-mismatch bug above), not on an ongoing basis.",
            "Cloud-cover ceiling differs by year (5%/30%/50%) depending on when that year happened to "
            "be fetched -- the per-pixel clear mask is the real quality control, but a year fetched at "
            "the 30%/50% ceiling had fewer candidate scenes to pick the least-cloudy one from.",
        ], [
            "Eén momentopnamedag per jaar, geen seizoensintegraal -- lees de vorm van de 22-jarige "
            "reeks, niet één jaar-op-jaar-sprong, zoals het tabblad Trends & Klimaat zelf al zegt.",
            "Twee verschillende sensoren op twee verschillende resoluties (10m/30m) delen één reeks; "
            "het overgangsjaar (2017->2018) is het enige echt controleerbare punt en is één keer "
            "gevalideerd (zie de vensterbug hierboven), niet doorlopend.",
            "Het bewolkingsplafond verschilt per jaar (5%/30%/50%) afhankelijk van wanneer dat jaar "
            "toevallig is opgehaald -- het per-pixel heldere-masker is de echte kwaliteitscontrole, "
            "maar een jaar opgehaald op het 30%/50%-plafond had minder kandidaatopnames om de minst "
            "bewolkte uit te kiezen.",
        ]),
        code_ref="src/preprocess.py, src/ndvi_trend.py, src/fetch_sentinel2.py, src/fetch_landsat.py, "
                 "src/preprocess_landsat.py",
        open_questions=([
            "The exact day-of-month chosen for each year's scene is stamped on that raster's own file "
            "(data/raw/summer_{year}.tif) but not persisted into stats.json -- the Methodology tab can "
            "state the search window and cloud ceiling, not the literal calendar date, without "
            "re-reading every raster's tags live.",
        ], [
            "De exacte dag van de maand die voor elk jaar's opname is gekozen staat op dat raster's "
            "eigen bestand (data/raw/summer_{year}.tif) maar wordt niet opgeslagen in stats.json -- het "
            "tabblad Methodologie kan het zoekvenster en bewolkingsplafond noemen, niet de letterlijke "
            "kalenderdatum, zonder elk raster's tags live opnieuw te lezen.",
        ]),
    ),
    MethodEntry(
        id="ndvi_change",
        title=("NDVI change", "NDVI-verandering"),
        measures=("Pixel-level vegetation change between two dates.",
                   "Vegetatieverandering per pixel tussen twee data."),
        source="Derived from the same two Sentinel-2 NDVI rasters as the NDVI entry above "
               "(summer_2024 -> summer_2025), no new fetch.",
        date_range=("August 2024 -> August 2025.", "Augustus 2024 -> augustus 2025."),
        resolution=("10m.", "10m."),
        processing=([
            "delta = NDVI(2025) - NDVI(2024), computed only where both dates' clear-ground mask is "
            "valid; pixels invalid in either date are set to NaN, not zero (so a cloud in one year "
            "doesn't register as \"vegetation loss\").",
            "greening = share of valid pixels where delta > +0.05; browning = share where "
            "delta < -0.05 (a fixed absolute threshold on the index itself, not adaptive to the "
            "AOI's own noise level).",
            "Current result: mean delta -0.022, 17.8% of clear ground greening, 28.6% browning.",
        ], [
            "delta = NDVI(2025) - NDVI(2024), alleen berekend waar het heldere-grond-masker van beide "
            "data geldig is; pixels ongeldig in één van beide data worden NaN, niet nul (zodat een "
            "wolk in één jaar niet als \"vegetatieverlies\" wordt geregistreerd.",
            "vergroening = aandeel geldige pixels waar delta > +0,05; verbruining = aandeel waar "
            "delta < -0,05 (een vaste absolute drempel op de index zelf, niet aangepast aan het eigen "
            "ruisniveau van het gebied).",
            "Huidig resultaat: gemiddelde delta -0,022, 17,8% van de onbewolkte grond vergroent, "
            "28,6% verbruint.",
        ]),
        limitations=([
            "Inherits every NDVI limitation above (one snapshot day per date, not a seasonal "
            "integral) -- a two-year delta is even more exposed to single-day weather than the "
            "22-year trend is.",
            "The +-0.05 greening/browning threshold is a round number, not derived from this AOI's own "
            "pixel-to-pixel noise floor.",
        ], [
            "Erft elke NDVI-beperking hierboven (één momentopnamedag per datum, geen "
            "seizoensintegraal) -- een verandering over twee jaar is nog gevoeliger voor "
            "weer-op-één-dag dan de 22-jarige trend.",
            "De +-0,05-vergroenings-/verbruiningsdrempel is een rond getal, niet afgeleid van het "
            "eigen pixel-op-pixel-ruisniveau van dit gebied.",
        ]),
        code_ref="src/landcover_ml.py (function ndvi_change)",
    ),
    MethodEntry(
        id="kmeans_land_cover",
        title=("Land cover (KMeans)", "Landgebruik (KMeans)"),
        measures=("Unsupervised classification of ground into water/built-up/farmland/vegetation-density classes.",
                   "Ongestuurde classificatie van grond in water/bebouwd/landbouw/vegetatiedichtheid-klassen."),
        source="Derived from the same Sentinel-2 reflectance/index bands as NDVI, no new fetch; "
               "clustering via scikit-learn's KMeans.",
        date_range=("One classification per year, 2018-present (Sentinel-2 era); see the Forest & land "
                     "cover change entry for the 2018-2026 spatial comparison this also feeds.",
                     "Eén classificatie per jaar, 2018-heden (Sentinel-2-tijdperk); zie het item Bos & "
                     "landgebruikverandering voor de ruimtelijke vergelijking 2018-2026 die dit ook voedt."),
        resolution=("10m.", "10m."),
        processing=([
            "Seven features per pixel: NDVI, NDWI, red/NIR/green/blue reflectance, and brightness "
            "(mean of red+green+blue) -- standardized (zero mean, unit variance) before clustering so "
            "no one feature's raw scale dominates the distance metric.",
            "KMeans(n_clusters=6, random_state=42, n_init=10) -- a fixed cluster count and a fixed "
            "random seed (reproducible, not re-tuned per year), 10 random initializations kept.",
            "Clusters are named from their own centroid signature *relative to the other clusters in "
            "that run*, not fixed absolute thresholds: water is the cluster with the largest "
            "(NDWI - NDVI) among clusters with positive NDWI; built-up/bare is the brightest remaining "
            "cluster if it also sits below the median NDVI of what's left; grass/farmland is the "
            "lowest-NDVI vegetated cluster if a real gap (>0.1 NDVI) separates it from the rest; "
            "everything left is ranked by brightness into up to three \"dense vegetation\" tiers "
            "(dark/mid/bright canopy). A fixed NDVI cutoff was tried first and rejected: August "
            "vegetation across this AOI is uniformly high-NDVI, so a threshold rule collapsed three "
            "spectrally distinct clusters into one label.",
        ], [
            "Zeven kenmerken per pixel: NDVI, NDWI, rode/NIR/groene/blauwe reflectie, en helderheid "
            "(gemiddelde van rood+groen+blauw) -- gestandaardiseerd (gemiddelde nul, eenheidsvariantie) "
            "vóór clustering zodat geen enkel kenmerk's ruwe schaal de afstandsmaat domineert.",
            "KMeans(n_clusters=6, random_state=42, n_init=10) -- een vast aantal clusters en een vaste "
            "random seed (reproduceerbaar, niet per jaar opnieuw afgesteld), 10 willekeurige "
            "initialisaties behouden.",
            "Clusters worden benoemd op basis van hun eigen centroïde-signatuur *relatief aan de "
            "andere clusters in die run*, geen vaste absolute drempels: water is het cluster met de "
            "grootste (NDWI - NDVI) onder clusters met positieve NDWI; bebouwd/kaal is het helderste "
            "resterende cluster als het ook onder de mediaan-NDVI van de rest zit; grasland/landbouw "
            "is het laagste-NDVI begroeide cluster als een echte kloof (>0,1 NDVI) het van de rest "
            "scheidt; alles wat overblijft wordt op helderheid gerangschikt in tot drie \"dichte "
            "vegetatie\"-niveaus (donker/midden/licht bladerdak). Een vaste NDVI-drempel is eerst "
            "geprobeerd en afgewezen: augustusvegetatie in dit gebied is overal hoog-NDVI, dus een "
            "drempelregel voegde drie spectraal verschillende clusters samen tot één label.",
        ]),
        limitations=([
            "Fully unsupervised -- no BRP/BGT ground-truth join yet, so class names are a spectral "
            "best guess, not validated against a labelled reference (stated on the dashboard itself "
            "on multiple tabs).",
            "Built-up/bare and grass/farmland sit closer together in spectral space than in reality; "
            "bare or just-harvested cropland can get relabelled between them from one year's centroid "
            "fit to the next -- named explicitly as the least stable category, versus water/forest "
            "which are more spectrally distinct and more trustworthy.",
            "A hard 6-cluster count can't resolve species-level forest type or sub-classes a satellite "
            "with only 4 optical bands and no SWIR wouldn't be able to distinguish anyway.",
        ], [
            "Volledig ongestuurd -- nog geen koppeling aan BRP/BGT-grondwaarheid, dus klassenamen zijn "
            "een spectrale inschatting, niet gevalideerd tegen een gelabelde referentie (op meerdere "
            "tabbladen van het dashboard zelf vermeld).",
            "Bebouwd/kaal en grasland/landbouw liggen spectraal dichter bij elkaar dan in werkelijkheid; "
            "kale of net geoogste akkers kunnen tussen deze twee heretiketteerd worden van de "
            "centroïde-fit van het ene jaar naar het volgende -- expliciet benoemd als de minst "
            "stabiele categorie, tegenover water/bos die spectraal duidelijker en betrouwbaarder zijn.",
            "Een vast aantal van 6 clusters kan geen bostype op soortniveau of subklassen oplossen die "
            "een satelliet met maar 4 optische banden en geen SWIR toch niet zou kunnen onderscheiden.",
        ]),
        code_ref="src/landcover_ml.py (functions classify, _label_clusters)",
    ),
    MethodEntry(
        id="sar_backscatter",
        title=("SAR backscatter (VV)", "SAR-terugkaatsing (VV)"),
        measures=("Radar reflectivity of the ground -- rough/urban surfaces bright, smooth/water dark.",
                   "Radarreflectiviteit van de grond -- ruwe/stedelijke oppervlakken licht, glad/water donker."),
        source="Sentinel-1 RTC (radiometrically terrain-corrected), via Microsoft Planetary Computer's "
               "STAC catalog (collection sentinel-1-rtc), VV + VH polarizations, already calibrated to "
               "linear gamma0.",
        date_range=("August 2025 (sar_2025, the Overview map's baseline date).",
                     "Augustus 2025 (sar_2025, de basisdatum van de Overzicht-kaart)."),
        resolution=("Not explicitly verified in this pipeline's own code -- see the open question below.",
                    "Niet expliciet gecontroleerd in de eigen code van deze pipeline -- zie de open vraag hieronder."),
        processing=([
            "VV/VH read as linear gamma0 directly from the STAC asset (no cloud to wait out -- radar).",
            "Converted to dB: 10 * log10(linear value), only where the linear value is > 0.",
            "Native grid is UTM zone 32N (EPSG:32632) -- Berg en Dal sits almost exactly on the 6°E "
            "UTM zone boundary, so this is a different zone from the Sentinel-2 optical mosaic's own "
            "31N grid. Kept in its native grid at fetch time; reprojected onto the optical grid only "
            "where the two are directly compared pixel-to-pixel (the SAR/optical water cross-check).",
            "Scene selection requires the real footprint (a diagonal parallelogram, not the bounding "
            "box) to fully *contain* the AOI polygon, not just intersect its bbox -- Berg en Dal sits "
            "close to a swath edge on some passes, where bbox-only filtering would silently accept a "
            "scene that only clips one corner of the municipality.",
        ], [
            "VV/VH rechtstreeks als lineaire gamma0 uit de STAC-asset gelezen (geen wolk om op te "
            "wachten -- radar).",
            "Omgezet naar dB: 10 * log10(lineaire waarde), alleen waar de lineaire waarde > 0 is.",
            "Het eigen raster is UTM-zone 32N (EPSG:32632) -- Berg en Dal ligt bijna precies op de "
            "6°O-UTM-zonegrens, dus dit is een andere zone dan het eigen 31N-raster van de optische "
            "Sentinel-2-mozaïek. Bij het ophalen in het eigen raster gehouden; alleen herprojecteerd "
            "naar het optische raster waar de twee direct pixel-voor-pixel worden vergeleken (de "
            "SAR/optische waterkruiscontrole).",
            "Scèneselectie vereist dat de echte contour (een diagonaal parallellogram, niet de "
            "bounding box) het AOI-polygoon volledig *bevat*, niet alleen de bbox overlapt -- Berg en "
            "Dal ligt dicht bij een swathrand op sommige passages, waar alleen-bbox-filtering stilletjes "
            "een opname zou accepteren die maar één hoek van de gemeente afsnijdt.",
        ]),
        limitations=([
            "One date, no multi-temporal averaging for this specific layer (unlike the flood-event "
            "change-detection product, which does build a multi-date reference composite).",
            "RTC (radiometric terrain correction) reduces but doesn't eliminate real terrain effects "
            "(layover/shadow on the AOI's own ridge slopes) -- not separately quantified here.",
        ], [
            "Eén datum, geen multi-temporele middeling voor deze specifieke laag (in tegenstelling tot "
            "het verandering-detectieproduct van de overstromingsgebeurtenis, dat wel een "
            "multi-datums referentiecomposiet opbouwt).",
            "RTC (radiometrische terreincorrectie) vermindert maar elimineert niet echte "
            "terreineffecten (layover/schaduw op de eigen stuwwalhellingen van het gebied) -- hier niet "
            "afzonderlijk gekwantificeerd.",
        ]),
        code_ref="src/fetch_sentinel1.py, src/preprocess_sar.py",
        open_questions=([
            "Native pixel spacing of Planetary Computer's sentinel-1-rtc product is not asserted or "
            "checked anywhere in this pipeline's code (commonly 20m for this specific product, but "
            "that number does not appear in src/fetch_sentinel1.py or src/preprocess_sar.py, so it's "
            "listed here as unverified rather than stated as fact).",
        ], [
            "De eigen pixelafstand van het sentinel-1-rtc-product van Planetary Computer wordt nergens "
            "in de code van deze pipeline beweerd of gecontroleerd (vaak 20m voor dit specifieke "
            "product, maar dat getal komt niet voor in src/fetch_sentinel1.py of src/preprocess_sar.py, "
            "dus hier vermeld als ongeverifieerd in plaats van als feit gesteld).",
        ]),
    ),
    MethodEntry(
        id="sar_water_mask",
        title=("SAR water mask", "SAR-watermasker"),
        measures=("Open water, detected from how little radar energy a surface reflects back.",
                   "Open water, gedetecteerd aan hoe weinig radarenergie een oppervlak terugkaatst."),
        source="Derived from the same Sentinel-1 VV backscatter as the entry above, no new fetch.",
        date_range=("August 2025.", "Augustus 2025."),
        resolution=("Same as SAR backscatter (see open question there).",
                    "Zelfde als SAR-terugkaatsing (zie de open vraag daar)."),
        processing=([
            "A pixel is flagged water where VV_dB < -17.0 dB (a fixed threshold): smooth open water "
            "is a near-specular reflector at C-band, so it returns very little energy to the sensor.",
            "The -17 dB cutoff was picked as consistent with the ~5-6% water fraction Sentinel-2's own "
            "NDWI/KMeans pass independently found for this AOI -- a plausibility check against a "
            "second method, not a value derived from a formal radiometric calibration curve.",
            "Cross-checked against the optical KMeans 'Water' cluster on the shared clear-ground area: "
            "current agreement (IoU, intersection over union) = 59.2% (SAR water 5.0% vs. optical "
            "water 5.6% of the same shared pixels).",
        ], [
            "Een pixel wordt als water gemarkeerd waar VV_dB < -17,0 dB (een vaste drempel): glad open "
            "water is een bijna-spiegelende reflector op C-band, dus het kaatst weinig energie terug "
            "naar de sensor.",
            "De -17 dB-drempel is gekozen als consistent met de ~5-6% waterfractie die Sentinel-2's "
            "eigen NDWI/KMeans-doorgang onafhankelijk vond voor dit gebied -- een plausibiliteitscheck "
            "tegen een tweede methode, geen waarde afgeleid van een formele radiometrische "
            "calibratiecurve.",
            "Gekruiscontroleerd tegen het optische KMeans-cluster 'Water' op het gedeelde onbewolkte "
            "gebied: huidige overeenstemming (IoU, intersection over union) = 59,2% (SAR-water 5,0% vs. "
            "optisch water 5,6% van dezelfde gedeelde pixels).",
        ]),
        limitations=([
            "A fixed absolute dB threshold, the same known weakness the old flood-extent method (below) "
            "has -- it doesn't necessarily transfer cleanly to a scene shot under different conditions "
            "or from a different orbit, only checked here for internal consistency against the optical "
            "estimate, not against ground survey data.",
            "59.2% IoU means real disagreement exists at the margins (river-edge vegetation, narrow "
            "channels, wet soil that isn't open water) -- not a validated water product on its own.",
        ], [
            "Een vaste absolute dB-drempel, dezelfde bekende zwakte die de oude overstromingsgebied-"
            "methode (hieronder) heeft -- vertaalt zich niet noodzakelijk goed naar een opname onder "
            "andere omstandigheden of vanuit een andere baan, hier alleen gecontroleerd op interne "
            "consistentie tegen de optische schatting, niet tegen veldonderzoeksdata.",
            "59,2% IoU betekent dat er echte onenigheid bestaat aan de randen (rivieroevervegetatie, "
            "smalle kanalen, natte grond die geen open water is) -- op zichzelf geen gevalideerd "
            "waterproduct.",
        ]),
        code_ref="src/preprocess_sar.py (functions process, cross_check_water)",
    ),
    MethodEntry(
        id="flood_extent_old",
        title=("Flood extent — fixed threshold (old method)", "Overstromingsgebied — vaste drempel (oude methode)"),
        measures=("Where a single high-water SAR scene shows water that a single baseline scene didn't.",
                   "Waar één hoogwater-SAR-opname water toont dat één basisopname niet toonde."),
        source="Derived from two Sentinel-1 scenes' own SAR water masks (see above), no new fetch: "
               "sar_2025 (Aug 2025, baseline) and sar_highwater_2024 (Jan 2024, documented Rhine/Waal "
               "high water).",
        date_range=("Baseline Aug 2025 vs. one event date, Jan 2024.",
                     "Basislijn aug. 2025 vs. één gebeurtenisdatum, jan. 2024."),
        resolution=("Baseline's own grid; the high-water scene is reprojected (nearest-neighbour) onto it.",
                    "Het eigen raster van de basislijn; de hoogwateropname wordt hierop herprojecteerd "
                    "(nearest-neighbour)."),
        processing=([
            "Each date's water mask (VV_dB < -17dB) computed independently, then compared per pixel: "
            "0 = dry both dates, 1 = water both dates (permanent water), 2 = newly flooded "
            "(dry->wet), 3 = baseline water the high-water scene missed (wet->dry).",
            "Current result: baseline water 5.0%, high-water-scene water 4.7%, newly flooded 2.1%, "
            "\"newly dry\" 2.3% -- net change effectively zero despite this being a documented flood "
            "event.",
        ], [
            "Het watermasker van elke datum (VV_dB < -17dB) onafhankelijk berekend, dan per pixel "
            "vergeleken: 0 = droog beide data, 1 = water beide data (permanent water), 2 = nieuw "
            "overstroomd (droog->nat), 3 = basiswater dat de hoogwateropname miste (nat->droog).",
            "Huidig resultaat: basiswater 5,0%, hoogwater-opname-water 4,7%, nieuw overstroomd 2,1%, "
            "\"nieuw droog\" 2,3% -- netto verandering vrijwel nul ondanks dat dit een gedocumenteerde "
            "overstromingsgebeurtenis is.",
        ]),
        limitations=([
            "This module's own code prints an explicit warning when \"newly dry\" exceeds \"newly "
            "flooded\" on a documented flood date (exactly what happens here): \"the fixed -17dB "
            "threshold doesn't transfer cleanly across scenes shot from different orbit directions/"
            "conditions... treat this pass as a pipeline proof, not a validated flood map.\" "
            "stats.json's own threshold_reliable field is set to False for this result.",
            "Kept in the dashboard for direct comparison against the change-detection method below, "
            "specifically *because* it fails -- not presented as a usable flood product on its own.",
        ], [
            "De eigen code van deze module print een expliciete waarschuwing wanneer \"nieuw droog\" "
            "groter is dan \"nieuw overstroomd\" op een gedocumenteerde overstromingsdatum (precies wat "
            "hier gebeurt): \"de vaste -17dB-drempel vertaalt zich niet goed tussen opnames vanuit "
            "verschillende baanrichtingen/omstandigheden... behandel deze doorgang als een "
            "pipeline-bewijs, geen gevalideerde overstromingskaart.\" Het eigen "
            "threshold_reliable-veld in stats.json staat voor dit resultaat op False.",
            "In het dashboard gehouden voor directe vergelijking met de verandering-detectiemethode "
            "hieronder, specifiek *omdat* het faalt -- niet gepresenteerd als een op zichzelf bruikbaar "
            "overstromingsproduct.",
        ]),
        code_ref="src/preprocess_sar.py (function flood_extent)",
    ),
    MethodEntry(
        id="flood_extent_new",
        title=("Flood extent — change detection (new method)", "Overstromingsgebied — verandering-detectie (nieuwe methode)"),
        measures=("Where the ground dropped well below its own normal radar reflectivity during the flood.",
                   "Waar de grond ver onder zijn eigen normale radarreflectiviteit zakte tijdens het hoogwater."),
        source="Sentinel-1 RTC, 9 scenes total: 4 reference dates + 5 event-window dates, via Planetary "
               "Computer (same collection as SAR backscatter above).",
        date_range=("Reference: 4 autumn-2023 dates (2023-11-02, -11-14, -11-26, -12-08), ~12 days apart. "
                     "Event window: 2023-12-23 (pre-event), 2024-01-04 (rise), 2024-01-13/16/25 "
                     "(recession, early/mid/late) -- bracketing the documented Lobith gauge peak "
                     "(14.5-14.7m NAP, 8-9 Jan 2024).",
                     "Referentie: 4 data uit najaar 2023 (2023-11-02, -11-14, -11-26, -12-08), ~12 dagen "
                     "uit elkaar. Gebeurtenisvenster: 2023-12-23 (vóór de gebeurtenis), 2024-01-04 "
                     "(stijging), 2024-01-13/16/25 (recessie, vroeg/midden/laat) -- rondom de "
                     "gedocumenteerde piek van meetstation Lobith (14,5-14,7m NAP, 8-9 jan. 2024)."),
        resolution=("Reference composite's own grid (the first reference date's grid; the other three "
                    "reference dates and every event date are reprojected onto it, bilinear).",
                    "Het eigen raster van het referentiecomposiet (het raster van de eerste "
                    "referentiedatum; de andere drie referentiedata en elke gebeurtenisdatum worden "
                    "hierop herprojecteerd, bilineair)."),
        processing=([
            "Per-pixel reference level = median VV dB across the 4 reference dates (median, not mean, "
            "so one anomalous pass doesn't skew the 'normal' baseline) -- computed only where at least "
            "one reference date actually covers that pixel.",
            "For each event date: delta = that date's VV dB (reprojected onto the reference grid) "
            "minus the per-pixel reference level.",
            "A pixel is flagged flooded where delta < -3.0 dB -- a threshold taken from the SAR "
            "flood-mapping literature (e.g. Twele et al. 2016), not tuned against this AOI's own data.",
            "No Sentinel-1 scene fully covering the AOI exists on the documented peak day itself "
            "(checked directly against the STAC catalogue) -- the nearest fully-covering passes on "
            "either side stand in for it.",
            "Current result: 9.1% flooded pre-event -> 21.2% at peak (13 Jan 2024, early recession), "
            "net +12.1 percentage points -- a clearly-above-noise signal, unlike the old method's "
            "~0 net change on the same underlying event.",
        ], [
            "Referentieniveau per pixel = mediaan VV dB over de 4 referentiedata (mediaan, geen "
            "gemiddelde, zodat één afwijkende passage de 'normale' basislijn niet scheeftrekt) -- "
            "alleen berekend waar minstens één referentiedatum die pixel daadwerkelijk dekt.",
            "Voor elke gebeurtenisdatum: delta = de VV dB van die datum (herprojecteerd naar het "
            "referentieraster) minus het referentieniveau per pixel.",
            "Een pixel wordt als overstroomd gemarkeerd waar delta < -3,0 dB -- een drempel uit de "
            "SAR-overstromingskarteringsliteratuur (bijv. Twele et al. 2016), niet afgesteld op de "
            "eigen data van dit gebied.",
            "Er bestaat geen Sentinel-1-opname die het gebied volledig dekt op de gedocumenteerde "
            "piekdag zelf (rechtstreeks gecontroleerd tegen de STAC-catalogus) -- de dichtstbijzijnde "
            "volledig dekkende passages aan beide zijden vervangen die.",
            "Huidig resultaat: 9,1% overstroomd vóór de gebeurtenis -> 21,2% op de piek (13 jan. 2024, "
            "vroege recessie), netto +12,1 procentpunt -- een signaal duidelijk boven de ruis, anders "
            "dan de ~0 netto verandering van de oude methode op dezelfde onderliggende gebeurtenis.",
        ]),
        limitations=([
            "Pre-event (9.1%) is not 0% -- this method has real background noise (speckle, seasonal "
            "moisture drift between the autumn-2023 reference dates and the January event window), so "
            "the *change* is the signal, not the absolute level.",
            "The -3dB threshold is a literature convention, not locally calibrated or validated against "
            "an independent ground/gauge survey of the actual flood extent.",
            "Peak lands a few days after Lobith's own gauge peak -- read here as a real floodplain-"
            "filling delay, not cross-checked against an independent hydrological model.",
        ], [
            "Vóór de gebeurtenis (9,1%) is niet 0% -- deze methode heeft echte achtergrondruis "
            "(speckle, seizoensgebonden vochtdrift tussen de referentiedata van najaar 2023 en het "
            "gebeurtenisvenster van januari), dus de *verandering* is het signaal, niet het absolute "
            "niveau.",
            "De -3dB-drempel is een literatuurconventie, niet lokaal gecalibreerd of gevalideerd tegen "
            "een onafhankelijk veld-/peilonderzoek van de daadwerkelijke overstromingsomvang.",
            "De piek valt een paar dagen na de eigen piek van meetstation Lobith -- hier gelezen als "
            "een echte vertraging bij het vullen van de uiterwaard, niet gekruiscontroleerd tegen een "
            "onafhankelijk hydrologisch model.",
        ]),
        code_ref="src/flood_event.py",
    ),
    MethodEntry(
        id="ahn_dtm",
        title=("Elevation (AHN DTM)", "Hoogte (AHN DTM)"),
        measures=("Bare-earth ground height, with buildings and trees stripped out.",
                   "Kale grondhoogte, met gebouwen en bomen eruit gefilterd."),
        source="AHN (Actueel Hoogtebestand Nederland), the Dutch national LiDAR archive, via PDOK's "
               "public WCS, coverage dtm_05m.",
        date_range=("Not stamped with an acquisition date in this pipeline's own output -- AHN is a "
                     "rolling national resurvey; see the open question below.",
                     "Niet voorzien van een opnamedatum in de eigen uitvoer van deze pipeline -- AHN is "
                     "een doorlopende landelijke herinventarisatie; zie de open vraag hieronder."),
        resolution=("Requested at 5m output grid (the WCS 'scalesize' parameter) directly from the "
                    "service -- native LiDAR point-cloud resolution is 0.5m, not downloaded at that "
                    "resolution and resampled locally.",
                    "Opgevraagd op een 5m-uitvoerraster (de WCS-parameter 'scalesize') rechtstreeks van "
                    "de dienst -- de native LiDAR-puntenwolkresolutie is 0,5m, niet op die resolutie "
                    "gedownload en lokaal geresampled."),
        processing=([
            "GetCoverage request to https://service.pdok.nl/rws/ahn/wcs/v1_0 (WCS 2.0.1), coverageId "
            "dtm_05m, subset to the municipal bounding box + 250m pad, scalesize computed from that "
            "bbox at 5m/px.",
            "Clipped to the exact municipal polygon (rasterio.mask, already in the WCS's native CRS, "
            "RD New/EPSG:28992).",
            "Current result: 8.0-95.1m NAP across the municipality (mean 31.2m) -- the real range from "
            "the Waal floodplain to the Nijmegen ridge (stuwwal).",
        ], [
            "GetCoverage-verzoek naar https://service.pdok.nl/rws/ahn/wcs/v1_0 (WCS 2.0.1), coverageId "
            "dtm_05m, subset op de gemeentelijke bounding box + 250m marge, scalesize berekend uit die "
            "bbox op 5m/px.",
            "Uitgeknipt tot de exacte gemeentegrens (rasterio.mask, al in het eigen CRS van de WCS, "
            "RD New/EPSG:28992).",
            "Huidig resultaat: 8,0-95,1m NAP over de gemeente (gemiddeld 31,2m) -- het echte bereik van "
            "de Waal-uiterwaarden tot de stuwwal bij Nijmegen.",
        ]),
        limitations=([
            "5m output resolution genuinely smooths sub-5m terrain features present in the native "
            "0.5m archive.",
        ], [
            "5m-uitvoerresolutie vlakt echte sub-5m-terreinkenmerken uit die in het eigen 0,5m-archief "
            "wel aanwezig zijn.",
        ]),
        code_ref="src/fetch_ahn.py",
        open_questions=([
            "This dashboard's own captions elsewhere label the source \"AHN4,\" but src/fetch_ahn.py "
            "never checks or asserts a version number against the WCS response (e.g. via "
            "GetCapabilities metadata) -- PDOK's current dtm_05m/dsm_05m coverage is AHN4 nationwide as "
            "of recent years, but that specific claim is not verified in this pipeline's own code and "
            "should be confirmed against the service's own capabilities document rather than assumed.",
            "No per-fetch acquisition date is recorded in stats.json for this layer (AHN acquisition "
            "varies by flight strip, not a single date for the whole municipality) -- listed here as "
            "open rather than guessed.",
        ], [
            "De eigen bijschriften van dit dashboard elders noemen de bron \"AHN4,\" maar "
            "src/fetch_ahn.py controleert of beweert nooit een versienummer tegen de WCS-respons "
            "(bijv. via GetCapabilities-metadata) -- PDOK's huidige dtm_05m/dsm_05m-dekking is de "
            "afgelopen jaren landelijk AHN4, maar die specifieke claim is niet gecontroleerd in de "
            "eigen code van deze pipeline en zou tegen het eigen capabilities-document van de dienst "
            "bevestigd moeten worden in plaats van aangenomen.",
            "Er wordt geen opnamedatum per fetch vastgelegd in stats.json voor deze laag (AHN-opname "
            "verschilt per vluchtstrook, geen enkele datum voor de hele gemeente) -- hier als open "
            "vermeld in plaats van geraden.",
        ]),
    ),
    MethodEntry(
        id="ahn_ndsm",
        title=("Canopy / building height (AHN nDSM)", "Bladerdak-/gebouwhoogte (AHN nDSM)"),
        measures=("Height of whatever the laser hit first, above the bare ground beneath it.",
                   "Hoogte van wat de laser het eerst raakte, boven de kale grond daaronder."),
        source="Derived from the same AHN WCS as the DTM entry above (coverage dsm_05m, first-return "
               "height), no new fetch beyond the DSM/DTM pair.",
        date_range=("Same as AHN DTM.", "Zelfde als AHN DTM."),
        resolution=("5m (same grid as DTM/DSM).", "5m (zelfde raster als DTM/DSM)."),
        processing=([
            "nDSM = clip(DSM - DTM, 0, None) -- clipped at zero because DSM should never read below "
            "DTM on real ground; small negative differences from grid/interpolation noise are floored "
            "rather than shown as \"negative height.\"",
            "A pixel invalid in either the DTM or the DSM is invalid in the nDSM too (nodata "
            "propagates from both inputs).",
            "Current result: mean 1.9m across the municipality, 1.1% of ground reads above 15m "
            "(mature forest canopy or tall buildings).",
        ], [
            "nDSM = clip(DSM - DTM, 0, None) -- geclipt op nul omdat DSM nooit onder DTM zou moeten "
            "uitlezen op echte grond; kleine negatieve verschillen door raster-/interpolatieruis worden "
            "afgevlakt in plaats van getoond als \"negatieve hoogte\".",
            "Een pixel ongeldig in óf de DTM óf de DSM is ook ongeldig in de nDSM (nodata propageert "
            "vanuit beide invoerlagen).",
            "Huidig resultaat: gemiddeld 1,9m over de gemeente, 1,1% van de grond leest boven 15m "
            "(volwassen bosbladerdak of hoge gebouwen).",
        ]),
        limitations=([
            "Doesn't distinguish tree canopy from buildings -- both read as \"first-return height above "
            "ground\"; a separate classification (e.g. against BAG building footprints) would be "
            "needed to split the two, and isn't done here.",
            "Same 5m-smoothing limitation as the DTM it's built from.",
        ], [
            "Onderscheidt geen bomenbladerdak van gebouwen -- beide lezen als \"eerste-terugkeer-hoogte "
            "boven de grond\"; een aparte classificatie (bijv. tegen BAG-gebouwcontouren) zou nodig "
            "zijn om de twee te scheiden, en gebeurt hier niet.",
            "Zelfde 5m-vervlakkingsbeperking als de DTM waaruit het is opgebouwd.",
        ]),
        code_ref="src/fetch_ahn.py (function fetch_all)",
    ),
    MethodEntry(
        id="no2",
        title=("Air quality — NO2", "Luchtkwaliteit — NO2"),
        measures=("Modelled annual-mean nitrogen dioxide concentration at ground level.",
                   "Gemodelleerde jaargemiddelde stikstofdioxideconcentratie op grondniveau."),
        source="RIVM Atlas Leefomgeving WCS, coverage alo__rivm_nsl_20260401_gm_NO22024 -- the same "
               "modelled grids used in official Dutch NSL (Nationaal Samenwerkingsprogramma "
               "Luchtkwaliteit) policy reporting, not a satellite product.",
        date_range=("2024 (the coverage's own name encodes both the data year, 2024, and this "
                     "particular publication/refresh of the grid, 2026-04-01).",
                     "2024 (de naam van de dekking codeert zowel het datajaar, 2024, als deze "
                     "specifieke publicatie/verversing van het raster, 2026-04-01)."),
        resolution=("1x1 km.", "1x1 km."),
        processing=([
            "GetCoverage request to https://data.rivm.nl/geo/alo/wcs (WCS 2.0.1), clipped to the exact "
            "municipal polygon (rasterio.mask).",
            "A physical sanity bound is applied before averaging: valid = (value != declared_nodata) "
            "AND (-1000 < value < 1000). This guards against a real bug found in the equivalent trend "
            "fetcher's older-year coverages: some declare nodata=0.0 in their own GDAL profile but "
            "actually fill nodata pixels with float32's most-negative value (~-3.4e38) -- excluding "
            "only the *declared* nodata value would let that sentinel corrupt the mean to -inf. Applied "
            "here too as a matter of consistency, whether or not this specific 2024 coverage turns out "
            "to need it.",
            "mean/min/max computed over the valid pixels only; pct_area_over_who_guideline = share of "
            "valid pixels above the WHO 2021 annual guideline (10 ug/m3 for NO2).",
            "Current result: mean 9.5 ug/m3, range 7.8-16.5 ug/m3, 19.7% of the municipality's area "
            "over the WHO guideline.",
        ], [
            "GetCoverage-verzoek naar https://data.rivm.nl/geo/alo/wcs (WCS 2.0.1), uitgeknipt tot de "
            "exacte gemeentegrens (rasterio.mask).",
            "Een fysieke plausibiliteitsgrens wordt toegepast vóór het middelen: geldig = (waarde != "
            "gedeclareerde nodata) EN (-1000 < waarde < 1000). Dit beschermt tegen een echte bug "
            "gevonden in de oudere-jaren-dekkingen van de equivalente trend-fetcher: sommige declareren "
            "nodata=0,0 in hun eigen GDAL-profiel maar vullen nodata-pixels feitelijk met de meest-"
            "negatieve waarde van float32 (~-3,4e38) -- alleen de *gedeclareerde* nodata-waarde "
            "uitsluiten zou dat toelaten dat die waarde het gemiddelde naar -inf corrumpeert. Hier ook "
            "toegepast uit consistentie, of deze specifieke 2024-dekking het nu nodig heeft of niet.",
            "gemiddelde/min/max berekend over alleen de geldige pixels; "
            "pct_area_over_who_guideline = aandeel geldige pixels boven de WHO-2021-jaarrichtlijn "
            "(10 ug/m3 voor NO2).",
            "Huidig resultaat: gemiddelde 9,5 ug/m3, bereik 7,8-16,5 ug/m3, 19,7% van het "
            "gemeenteoppervlak boven de WHO-richtlijn.",
        ]),
        limitations=([
            "A national dispersion model output (RIVM's own NSL grids), not a direct sensor "
            "measurement at any point in Berg en Dal -- authoritative for policy purposes, but modelled.",
            "Does not include NH3/nitrogen deposition, the figure Dutch farm-nitrogen permitting "
            "actually runs on -- confirmed absent from this specific WCS (checked directly against its "
            "capabilities), not simply unfetched; that figure lives in RIVM's separate GDN/AERIUS "
            "product as an annual grid download, not a queryable service.",
        ], [
            "Uitvoer van een nationaal verspreidingsmodel (RIVM's eigen NSL-rasters), geen directe "
            "sensormeting op enig punt in Berg en Dal -- gezaghebbend voor beleidsdoeleinden, maar "
            "gemodelleerd.",
            "Bevat geen NH3/stikstofdepositie, het cijfer waar de Nederlandse stikstofvergunning voor "
            "de landbouw daadwerkelijk op draait -- rechtstreeks bevestigd afwezig in deze specifieke "
            "WCS (rechtstreeks gecontroleerd tegen de eigen capabilities), niet simpelweg niet-"
            "opgehaald; dat cijfer zit in RIVM's aparte GDN/AERIUS-product als jaarlijkse "
            "rasterdownload, geen bevraagbare dienst.",
        ]),
        code_ref="src/fetch_air_quality.py",
    ),
]


def _render_markdown() -> str:
    lines = [
        "# Methodology",
        "",
        "One entry per indicator on the Berg en Dal dashboard: what it measures, "
        "the exact source and processing, and its known limitations — read "
        "directly from the code that computes it "
        "(`src/methodology.py` is the single source of truth this file is "
        "generated from; run `python src/methodology.py` after editing that "
        "file to regenerate this one).",
        "",
        "Every number below reflects this repository's state when generated; "
        "see each entry's own \"Code\" line for where to check the live logic, "
        "and `data/processed/stats.json` for the live current value.",
        "",
    ]
    for e in ENTRIES:
        lines.append(f"## {e.title[0]}")
        lines.append("")
        lines.append(f"**Measures:** {e.measures[0]}")
        lines.append("")
        lines.append(f"**Source:** {e.source}")
        lines.append("")
        lines.append(f"**Date/period:** {e.date_range[0]}")
        lines.append("")
        lines.append(f"**Resolution:** {e.resolution[0]}")
        lines.append("")
        lines.append("**Processing:**")
        for step in e.processing[0]:
            lines.append(f"- {step}")
        lines.append("")
        lines.append("**Limitations:**")
        for lim in e.limitations[0]:
            lines.append(f"- {lim}")
        lines.append("")
        if e.open_questions[0]:
            lines.append("**Open questions:**")
            for q in e.open_questions[0]:
                lines.append(f"- {q}")
            lines.append("")
        lines.append(f"**Code:** `{e.code_ref}`")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def write_methodology_md() -> Path:
    out_path = ROOT / "METHODOLOGY.md"
    out_path.write_text(_render_markdown(), encoding="utf-8")
    print(f"wrote {out_path} ({len(ENTRIES)} entries)")
    return out_path


if __name__ == "__main__":
    write_methodology_md()
