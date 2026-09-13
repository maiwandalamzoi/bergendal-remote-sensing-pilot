"""
Unsupervised land-cover classification (KMeans) over the Berg en Dal AOI,
plus a one-year NDVI change layer -- the two ML/analysis outputs the
concept note proposed for the "spatial planning" and "nature" lenses.

KMeans is deliberately unsupervised: there is no labelled training set yet
(that's the BRP/BGT join proposed as Phase 2), so clusters are grouped by
spectral behaviour and then labelled by their own NDVI/NDWI/brightness
signature -- e.g. "the cluster that's dark, wet, and has negative NDVI is
water" -- rather than trained against ground truth.
"""
from pathlib import Path

import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from statsutil import update_stats

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

N_CLUSTERS = 6
RANDOM_STATE = 42


def _load_indices(label: str):
    path = PROC_DIR / f"{label}_indices.tif"
    with rasterio.open(path) as src:
        bands = src.read()
        profile = src.profile.copy()
    names = ["NDVI", "NDWI", "red", "nir", "green", "blue", "valid"]
    data = dict(zip(names, bands))
    return data, profile


def _label_clusters(centers: np.ndarray, feature_names: list[str]) -> dict[int, str]:
    """Name each cluster from its centroid signature, relative to the other
    clusters in this run, rather than fixed absolute thresholds.

    Fixed NDVI cutoffs turned out to be the wrong tool here: August
    vegetation across the AOI is uniformly high-NDVI (forest, orchard and
    healthy pasture all score above the naive 0.55 "forest" cutoff), so a
    threshold rule collapsed three spectrally distinct clusters into one
    label. Ranking clusters against each other on NDVI/brightness/NIR
    keeps the distinction the four-band data can actually support --
    telling dark, high-NIR canopy apart from brighter, lower-NIR canopy --
    without overclaiming a species-level forest type a 4-band optical
    sensor can't resolve on its own (that needs the SWIR bands or the BRP
    ground truth Phase 2 of the roadmap proposes).
    """
    ndvi_i = feature_names.index("NDVI")
    ndwi_i = feature_names.index("NDWI")
    bright_i = feature_names.index("brightness")

    remaining = set(range(len(centers)))
    labels: dict[int, str] = {}

    # 1. Water: clearest spectral signature (NDWI positive, NDVI negative) --
    #    identify it first so it can't steal the "darkest cluster" slot below.
    water_candidates = [i for i in remaining if centers[i][ndwi_i] > 0]
    if water_candidates:
        wi = max(water_candidates, key=lambda i: centers[i][ndwi_i] - centers[i][ndvi_i])
        labels[wi] = "Water"
        remaining.discard(wi)

    # 2. Built-up / bare: brightest surviving cluster, if it's also below-
    #    median NDVI (bright *and* green together would just be farmland).
    if remaining:
        med_ndvi = np.median([centers[i][ndvi_i] for i in remaining])
        bi = max(remaining, key=lambda i: centers[i][bright_i])
        if centers[bi][ndvi_i] < med_ndvi:
            labels[bi] = "Built-up / bare"
            remaining.discard(bi)

    # 3. Grass / farmland: lowest-NDVI vegetated cluster, if it sits in a
    #    clear gap below the rest (otherwise everything left is forest-like).
    if remaining:
        by_ndvi = sorted(remaining, key=lambda i: centers[i][ndvi_i])
        gi = by_ndvi[0]
        gap_ok = len(by_ndvi) == 1 or (centers[by_ndvi[1]][ndvi_i] - centers[gi][ndvi_i]) > 0.1
        if gap_ok:
            labels[gi] = "Grass / farmland"
            remaining.discard(gi)

    # 4. Whatever's left is dense, high-NDVI vegetation -- differentiate by
    #    brightness (dark canopy first, since conifers/shade run darker in
    #    both visible and NIR than open broadleaf canopy or dense crops).
    if remaining:
        by_bright = sorted(remaining, key=lambda i: centers[i][bright_i])
        tiers = ["Dense vegetation, dark canopy", "Dense vegetation, mid canopy", "Dense vegetation, bright canopy"]
        for rank, i in enumerate(by_bright):
            labels[i] = tiers[min(rank, len(tiers) - 1)] if len(by_bright) > 1 else "Dense vegetation (forest)"

    return labels


def classify(label: str) -> Path:
    data, profile = _load_indices(label)
    valid = data["valid"] > 0
    h, w = valid.shape

    feature_names = ["NDVI", "NDWI", "red", "nir", "green", "blue", "brightness"]
    brightness = (data["red"] + data["green"] + data["blue"]) / 3.0
    feats = np.stack([data["NDVI"], data["NDWI"], data["red"], data["nir"],
                       data["green"], data["blue"], brightness])

    X = feats[:, valid].T  # (n_valid_px, n_features)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_ids = km.fit_predict(Xs)

    centers_raw = scaler.inverse_transform(km.cluster_centers_)
    names = _label_clusters(centers_raw, feature_names)

    out = np.full((h, w), 255, dtype=np.uint8)  # 255 = nodata/cloud
    out[valid] = cluster_ids.astype(np.uint8)

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=255)
    out_path = OUT_DIR / f"{label}_landcover.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(out[np.newaxis, :, :])

    # Sentinel-2 10m grid = 100 m2/px = 0.01 ha/px -- hectares (not just %
    # of that year's clear ground) is what a year-over-year "how much did
    # X actually grow" comparison needs, since clear-ground coverage
    # itself varies year to year (cloud) and a %-of-clear-ground figure
    # alone would silently conflate "more built-up" with "less cloud."
    HA_PER_PX = 0.01
    print(f"[{label}] land cover clusters:")
    class_pcts, class_ha = {}, {}
    for cid in range(N_CLUSTERS):
        n_px = int((cluster_ids == cid).sum())
        pct = n_px / len(cluster_ids) * 100
        ha = n_px * HA_PER_PX
        print(f"  cluster {cid:>2} = {names[cid]:<26} {pct:5.1f}% of clear ground  ({ha:7.1f} ha)")
        class_pcts[names[cid]] = class_pcts.get(names[cid], 0.0) + round(float(pct), 1)
        class_ha[names[cid]] = class_ha.get(names[cid], 0.0) + round(float(ha), 1)

    legend_path = OUT_DIR / f"{label}_landcover_legend.txt"
    with open(legend_path, "w") as f:
        for cid in range(N_CLUSTERS):
            f.write(f"{cid}\t{names[cid]}\n")

    update_stats(f"landcover_{label}", {
        "n_clusters": N_CLUSTERS, "class_pct": class_pcts, "class_ha": class_ha,
    })
    return out_path


def ndvi_change(label_new: str, label_old: str) -> Path:
    """Pixel-wise NDVI difference between two dates, clear-ground only."""
    new_data, profile = _load_indices(label_new)
    old_data, _ = _load_indices(label_old)

    both_valid = (new_data["valid"] > 0) & (old_data["valid"] > 0)
    delta = np.zeros_like(new_data["NDVI"])
    delta[both_valid] = new_data["NDVI"][both_valid] - old_data["NDVI"][both_valid]
    delta[~both_valid] = np.nan

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", nodata=np.nan)
    out_path = OUT_DIR / f"ndvi_change_{label_old}_to_{label_new}.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(delta[np.newaxis, :, :])

    valid_delta = delta[both_valid]
    mean_delta = float(np.nanmean(valid_delta))
    greening = float((valid_delta > 0.05).mean() * 100)
    browning = float((valid_delta < -0.05).mean() * 100)
    print(f"NDVI change {label_old} -> {label_new}: "
          f"mean={mean_delta:+.3f}  greening(>+0.05)={greening:.1f}%  browning(<-0.05)={browning:.1f}%")
    update_stats("ndvi_change", {
        "from": label_old, "to": label_new,
        "mean_delta": round(mean_delta, 3),
        "greening_pct": round(greening, 1), "browning_pct": round(browning, 1),
    })
    return out_path


if __name__ == "__main__":
    classify("summer_2025")
    ndvi_change("summer_2025", "summer_2024")
