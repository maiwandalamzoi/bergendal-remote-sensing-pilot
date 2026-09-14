"""
A real, downloadable print report -- the QGIS/ArcGIS "print composer"
equivalent this project didn't have. The sidebar's "Print this tab" button
only ever triggered the browser's own print dialog on whatever tab
happened to be open (no choice of what's on the page, no legend baked in
reliably across browsers) -- this module builds an actual multi-page PDF
instead: pick which real layers to include, get one page per layer with a
title block, the layer's real geographic extent, a legend that matches
what's actually on it (categorical swatches for land cover/crops, a
colour-ramp bar for continuous indices), a north arrow, and a scale bar
computed from that page's own real extent -- not a browser screenshot of
the interactive map.
"""
import datetime
import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib import colormaps
from PIL import Image

from statsutil import load_stats
from visualize import (
    LANDCOVER_COLORS, landcover_label, CROP_FAMILIES,
    true_color_png, ndvi_png, landcover_png, ndvi_change_png,
    sar_png, sar_water_png, flood_extent_png, elevation_png, ndsm_png,
    air_quality_png, brp_png,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _continuous_legend(ax_leg, cmap_name: str, vmin: float, vmax: float, unit: str, lang: str) -> None:
    """A horizontal colour-ramp legend for a continuous raster (NDVI, SAR,
    NO2, elevation) -- min/max labelled at the ends, matching the real
    stretch each PNG function above was rendered with, not a generic 0-1
    bar that would misrepresent what the colours actually mean."""
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax_leg.imshow(grad, aspect="auto", cmap=colormaps[cmap_name], extent=[vmin, vmax, 0, 1])
    ax_leg.set_yticks([])
    ax_leg.set_xticks([vmin, (vmin + vmax) / 2, vmax])
    ax_leg.set_xlabel(unit, fontsize=8)
    ax_leg.tick_params(labelsize=7)
    for spine in ax_leg.spines.values():
        spine.set_visible(False)


def _categorical_legend(ax_leg, entries: list[tuple[str, str]]) -> None:
    """entries: list of (label, hex colour). Small coloured squares + text,
    stacked -- the same information a folium legend box carries, drawn
    into the PDF page itself so it survives being printed/emailed as a
    static file. Uses a fixed-size marker (not an axes-fraction Rectangle)
    for the swatch -- this legend column is much taller than it is wide,
    so a "square" sized in axes-fraction units would render as a tall
    thin bar instead of a square, a real bug caught by actually looking
    at the rendered PDF page, not just running the code."""
    ax_leg.axis("off")
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    n = len(entries)
    for i, (label, color) in enumerate(entries):
        y = 1 - (i + 0.5) / max(n, 1)
        ax_leg.plot([0.05], [y], marker="s", markersize=11, color=color,
                    markeredgewidth=0, transform=ax_leg.transAxes, clip_on=False)
        ax_leg.text(0.14, y, label, transform=ax_leg.transAxes, fontsize=8, va="center")


def _north_arrow(ax) -> None:
    ax.annotate("N", xy=(0.95, 0.90), xytext=(0.95, 0.80), xycoords="axes fraction",
                textcoords="axes fraction", ha="center", fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6),
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=0.8))


def _scale_bar(ax, bounds: list) -> None:
    """A real scale bar sized to this page's own extent -- 1 degree of
    longitude is ~111.32*cos(latitude) km, not a fixed length reused
    across every layer regardless of how zoomed in it is."""
    (south, west), (north, east) = bounds
    lat_mid = (south + north) / 2
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat_mid))
    width_km = (east - west) * km_per_deg_lon
    # round to a "nice" bar length: 1/5 of the width, snapped to 1/2/5x10^n
    raw = width_km / 5
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for mult in (1, 2, 5, 10):
        bar_km = mult * magnitude
        if bar_km >= raw:
            break
    bar_frac = bar_km / width_km if width_km > 0 else 0.2
    x0, y0 = 0.04, 0.05
    ax.plot([x0, x0 + bar_frac], [y0, y0], transform=ax.transAxes, color="black", lw=3,
            solid_capstyle="butt")
    ax.text(x0 + bar_frac / 2, y0 + 0.02, f"{bar_km:g} km", transform=ax.transAxes,
            ha="center", fontsize=8)


def _layer_page(pdf: PdfPages, title: str, subtitle: str, png_path: Path, bounds: list,
                 legend_kind: str, legend_data, lang: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    gs = fig.add_gridspec(1, 5, wspace=0.05)
    ax_map = fig.add_subplot(gs[0, :4])
    ax_leg = fig.add_subplot(gs[0, 4])

    (south, west), (north, east) = bounds
    img = np.asarray(Image.open(png_path))
    ax_map.imshow(img, extent=[west, east, south, north])
    ax_map.set_xlim(west, east)
    ax_map.set_ylim(south, north)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_linewidth(0.8)
    _north_arrow(ax_map)
    _scale_bar(ax_map, bounds)

    if legend_kind == "categorical":
        _categorical_legend(ax_leg, legend_data)
    elif legend_kind == "continuous":
        cmap_name, vmin, vmax, unit = legend_data
        ax_leg.set_position([ax_leg.get_position().x0, 0.45, ax_leg.get_position().width, 0.05])
        _continuous_legend(ax_leg, cmap_name, vmin, vmax, unit, lang)
    else:
        ax_leg.axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.06, ha="left", y=0.97)
    fig.text(0.06, 0.935, subtitle, fontsize=9, color="#51604F")
    fig.text(0.06, 0.02,
              ("Berg en Dal remote-sensing pilot -- real satellite/LiDAR/registry data, "
               if lang == "en" else
               "Berg en Dal aardobservatie-pilot -- echte satelliet-/LiDAR-/registratiedata, ")
              + datetime.date.today().isoformat(),
              fontsize=7, color="#9aa295")
    pdf.savefig(fig)
    plt.close(fig)


# key -> (title_en, title_nl, fetch callable returning (path, bounds[, extra]))
_LAYER_BUILDERS = {
    "true_color": ("True colour (Sentinel-2, Aug 2025)", "Ware kleur (Sentinel-2, aug. 2025)",
                   lambda: true_color_png("summer_2025")),
    "ndvi": ("NDVI -- vegetation health", "NDVI -- vegetatievitaliteit",
             lambda: ndvi_png("summer_2025")),
    "ndvi_change": ("NDVI change, 2024→2025", "NDVI-verandering, 2024→2025",
                     lambda: ndvi_change_png("summer_2024", "summer_2025")),
    "landcover": ("Land cover (KMeans)", "Landgebruik (KMeans)",
                  lambda: landcover_png("summer_2025")),
    "sar": ("SAR backscatter (VV)", "SAR-terugkaatsing (VV)", sar_png),
    "sar_water": ("SAR water mask", "SAR-watermasker", sar_water_png),
    "flood": ("Flood extent (SAR)", "Overstromingsgebied (SAR)", flood_extent_png),
    "elevation": ("Elevation (AHN DTM)", "Hoogte (AHN DTM)", elevation_png),
    "ndsm": ("Canopy/building height (AHN nDSM)", "Bladerdak-/gebouwhoogte (AHN nDSM)", ndsm_png),
    "no2": ("Air quality (NO2, RIVM)", "Luchtkwaliteit (NO2, RIVM)", lambda: air_quality_png("no2")),
    "brp": ("Farm parcels (BRP)", "Landbouwpercelen (BRP)", brp_png),
}

# What's shown to the person picking layers, and the order they appear in
# the generated PDF -- deliberately the same order as the map's own
# grouped layer control, so a report and the live map read the same way.
REPORT_LAYER_CHOICES: list[tuple[str, tuple[str, str]]] = [
    ("true_color", ("True colour", "Ware kleur")),
    ("ndvi", ("NDVI (vegetation health)", "NDVI (vegetatievitaliteit)")),
    ("ndvi_change", ("NDVI change", "NDVI-verandering")),
    ("landcover", ("Land cover (KMeans)", "Landgebruik (KMeans)")),
    ("sar", ("SAR backscatter", "SAR-terugkaatsing")),
    ("sar_water", ("SAR water mask", "SAR-watermasker")),
    ("flood", ("Flood extent", "Overstromingsgebied")),
    ("elevation", ("Elevation (AHN DTM)", "Hoogte (AHN DTM)")),
    ("ndsm", ("Canopy/building height", "Bladerdak-/gebouwhoogte")),
    ("no2", ("Air quality (NO2)", "Luchtkwaliteit (NO2)")),
    ("brp", ("Farm parcels (BRP)", "Landbouwpercelen (BRP)")),
]


def build_report_pdf(selected_keys: list[str], lang: str = "en") -> bytes:
    """Builds the actual PDF in memory and returns its bytes -- the caller
    (dashboard.py) hands these straight to st.download_button, no temp
    file left behind. Raises KeyError for an unknown layer key rather than
    silently skipping it, since a silently-shorter report is worse than a
    loud failure while this is still being wired up."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Cover page
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.5, 0.6, "Berg en Dal", fontsize=34, fontweight="bold", ha="center", family="serif")
        fig.text(0.5, 0.52,
                  "Remote-sensing pilot -- printed report" if lang == "en" else
                  "Aardobservatie-pilot -- afgedrukt rapport",
                  fontsize=15, ha="center", color="#51604F")
        fig.text(0.5, 0.46, datetime.date.today().isoformat(), fontsize=11, ha="center", color="#9aa295")
        layer_titles = [(_LAYER_BUILDERS[k][0] if lang == "en" else _LAYER_BUILDERS[k][1]) for k in selected_keys]
        # Fixed band (0.38 down to 0.12) for the TOC regardless of how many
        # layers were picked -- a line spacing computed from len(layer_titles)
        # rather than a fixed 0.03 step, so 11 items and 2 items both fit
        # without the last line ever colliding with the footer below it
        # (a real bug in the first version of this page, caught by actually
        # rendering it with every layer selected, not just a couple).
        toc_top, toc_bottom = 0.38, 0.12
        step = min(0.03, (toc_top - toc_bottom) / max(len(layer_titles), 1))
        fig.text(0.5, toc_top + 0.03, ("Contents:" if lang == "en" else "Inhoud:"), fontsize=10, ha="center", fontweight="bold")
        for i, title in enumerate(layer_titles):
            fig.text(0.5, toc_top - i * step, f"{i + 1}. {title}", fontsize=9, ha="center", color="#16221C")
        fig.text(0.5, 0.05,
                  ("Real satellite/LiDAR/registry data -- nothing on this report is simulated." if lang == "en" else
                   "Echte satelliet-, LiDAR- en registratiedata -- niets op dit rapport is gesimuleerd."),
                  fontsize=8, ha="center", style="italic", color="#7a8177")
        pdf.savefig(fig)
        plt.close(fig)

        for key in selected_keys:
            if key not in _LAYER_BUILDERS:
                raise KeyError(f"Unknown report layer '{key}'")
            title_en, title_nl, fn = _LAYER_BUILDERS[key]
            title = title_en if lang == "en" else title_nl
            result = fn()
            path, bounds = result[0], result[1]

            if key == "landcover":
                id_to_name = result[2]
                entries = [(landcover_label(name, lang), LANDCOVER_COLORS.get(name, "#999"))
                           for name in sorted(set(id_to_name.values()))]
                _layer_page(pdf, title, "KMeans, 6 clusters, Aug 2025", path, bounds, "categorical", entries, lang)
            elif key == "brp":
                brp_colors = result[2]
                entries = [(fam_en if lang == "en" else fam_nl, color)
                           for fam, (color, _icon, fam_en, fam_nl) in CROP_FAMILIES.items()]
                _layer_page(pdf, title, "BRP 2025", path, bounds, "categorical", entries, lang)
            elif key == "ndvi":
                _layer_page(pdf, title, "Sentinel-2, Aug 2025", path, bounds, "continuous",
                            ("RdYlGn", -0.2, 0.9, "NDVI"), lang)
            elif key == "ndvi_change":
                _layer_page(pdf, title, "Sentinel-2, Aug 2024 -> Aug 2025", path, bounds, "continuous",
                            ("BrBG", -0.3, 0.3, "Δ NDVI"), lang)
            elif key == "sar":
                _layer_page(pdf, title, "Sentinel-1 RTC, Aug 2025", path, bounds, "continuous",
                            ("gray", -22, -2, "dB"), lang)
            elif key == "elevation":
                _layer_page(pdf, title, "AHN LiDAR DTM", path, bounds, "none", None, lang)
            elif key == "ndsm":
                _layer_page(pdf, title, "AHN LiDAR nDSM", path, bounds, "none", None, lang)
            elif key == "no2":
                _layer_page(pdf, title, "RIVM, 2024", path, bounds, "none", None, lang)
            else:
                _layer_page(pdf, title, "", path, bounds, "none", None, lang)

    return buf.getvalue()
