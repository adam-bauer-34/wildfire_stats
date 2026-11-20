"""Land-cover figures: fire maps by class, class counts, and confusion matrices."""

import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wildfire_stats.config import resolve
from wildfire_stats.plotting.presets import finish_fig, slugify
from wildfire_stats.reporting.tables import confusion_to_labels, period_metrics

# Default North America view for map panels without usable bounds
DEFAULT_EXTENT = [-170, -50, 15, 75]


def _add_map_features(ax):
    """Land, coastlines, and borders on a Cartopy axis."""
    ax.add_feature(cfeature.LAND, facecolor="lightgrey")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)


# TODO(bug): total_bounds is in the geometry CRS (EPSG:6933 metres for MTBS), but
# set_extent treats it as lon/lat; works only for lon/lat inputs.
def plot_biome_facets(
    mtbs_classified: pd.DataFrame | gpd.GeoDataFrame, save_figs: bool = False
) -> None:
    """Create a 3×3 faceted map of MTBS fire locations by individual biome.

    One panel per biome (up to 9 panels); here we use 8 MODIS classes:
      - Deciduous Broadleaf forest
      - Evergreen Broadleaf forest
      - Mixed forest
      - Savannas
      - Woody savannas
      - Evergreen Needleleaf forest
      - Grasslands
      - Open shrublands

    ``save_figs=True`` saves the figure to ``FIGS_DIR``.
    """
    # Silence runtime warnings from Cartopy/geopandas
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Order of individual biomes to plot (one per panel)
    biomes = [
        "Deciduous Broadleaf forest",
        "Evergreen Broadleaf forest",
        "Mixed forest",
        "Savannas",
        "Woody savannas",
        "Evergreen Needleleaf forest",
        "Grasslands",
        "Open shrublands",
    ]

    # Assign a distinct color to each biome
    biome_colors = {
        "Deciduous Broadleaf forest": "tab:orange",
        "Evergreen Broadleaf forest": "tab:green",
        "Mixed forest": "tab:brown",
        "Savannas": "tab:olive",
        "Woody savannas": "tab:gray",
        "Evergreen Needleleaf forest": "tab:blue",
        "Grasslands": "tab:red",
        "Open shrublands": "tab:purple",
    }

    # Copy input to avoid mutating the caller's object
    df = mtbs_classified.copy()

    # Drop rows with missing MODIS classification or non-finite coordinates
    df = df[df["modis_class_static"].notna()].copy()
    df = df[df["LATITUDE"].apply(np.isfinite) & df["LONGITUDE"].apply(np.isfinite)]

    # If not already a GeoDataFrame, construct geometry points from lat/lon
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE),
            crs="EPSG:4326",
        )

    # Remove invalid or empty geometries
    df = df[df.geometry.notna() & (~df.geometry.is_empty) & df.geometry.is_valid].copy()
    if df.empty:
        print("No valid fire locations found for plotting.")
        return

    # 3×3 grid of panels
    # Create a faceted 3×3 grid of Cartopy subplots
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 12),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes_flat = axes.ravel()

    # Plot each biome in its own panel
    for ax, biome in zip(axes_flat, biomes):
        subset = df[df["modis_class_static"] == biome].copy()

        # Add geographic context (land, coastlines, borders)
        _add_map_features(ax)

        if subset.empty:
            # If no fires for this biome, display "no data" and default extent
            ax.set_title(f"{biome}\n(no data)", fontsize=11, pad=8)
            ax.set_extent(DEFAULT_EXTENT, crs=ccrs.PlateCarree())
        else:
            # Scatter plot all fire locations for this biome
            ax.scatter(
                subset["LONGITUDE"],
                subset["LATITUDE"],
                color=biome_colors.get(biome, "black"),
                s=15,
                alpha=0.6,
                transform=ccrs.PlateCarree(),
                label=biome,
            )

            # Zoom to data with a little padding
            try:
                minx, miny, maxx, maxy = subset.total_bounds
                pad_x = max((maxx - minx) * 0.05, 5)
                pad_y = max((maxy - miny) * 0.05, 5)
                ax.set_extent(
                    [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y],
                    crs=ccrs.PlateCarree(),
                )
            except Exception:
                # If bounds are invalid, use a default North America view
                ax.set_extent(DEFAULT_EXTENT, crs=ccrs.PlateCarree())

            ax.set_title(biome, fontsize=11, pad=8)

        # Gridlines (labels only on left/bottom to avoid clutter)
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3)
        gl.top_labels = False
        gl.right_labels = False

    # Turn off any unused axes (9th panel, since we only have 8 biomes)
    if len(biomes) < len(axes_flat):
        for ax in axes_flat[len(biomes) :]:
            ax.set_visible(False)

    plt.tight_layout()
    finish_fig("biome_facets", save_figs, fig=fig)


def plot_category_facets(
    fires_evt: gpd.GeoDataFrame | pd.DataFrame,
    col_name: str,
    min_fires: int | None = None,
    save_figs: bool = False,
    pad_frac: float = 0.2,
    min_span: tuple[float, float] = (25.0, 15.0),
) -> None:
    """
    Create a faceted map of MTBS fire locations for EVT_NAME classes
    that have more than `min_fires` fires (e.g., >400).

    Each panel shows fire centroids (LAT/LONG) for one EVT_NAME, zoomed to the
    data plus ``pad_frac`` of its extent on each side, but never smaller than
    ``min_span`` = (lon degrees, lat degrees). Raise either to zoom out further.
    ``min_fires`` defaults to ``min_fires`` in the yaml. (Was notebook 7's
    ``plot_evtname_facets``.) ``save_figs=True`` saves the figure to ``FIGS_DIR``.
    """
    min_fires = resolve(min_fires, "min_fires")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = fires_evt.copy()
    df = df[df[col_name].notna()].copy()
    df = df[df["LATITUDE"].apply(np.isfinite) & df["LONGITUDE"].apply(np.isfinite)]

    if df.empty:
        print("No valid fire locations found for plotting.")
        return

    counts = df[col_name].value_counts()
    top_evts = counts[counts > min_fires].index.tolist()

    if not top_evts:
        print(f"No categories with more than {min_fires} fires.")
        return

    top_evts = sorted(top_evts, key=lambda name: counts[name], reverse=True)

    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE),
            crs="EPSG:4326",
        )

    df = df[df.geometry.notna() & (~df.geometry.is_empty) & df.geometry.is_valid].copy()
    if df.empty:
        print("No valid fire geometries after cleaning.")
        return

    n = len(top_evts)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel()
    else:
        axes_flat = [axes]

    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(i % cmap.N) for i, name in enumerate(top_evts)}

    for ax, evt_name in zip(axes_flat, top_evts):
        subset = df[df[col_name] == evt_name].copy()

        _add_map_features(ax)

        if subset.empty:
            ax.set_title(f"{evt_name}\n(no data)", fontsize=9, pad=6)
            ax.set_extent(DEFAULT_EXTENT, crs=ccrs.PlateCarree())
        else:
            ax.scatter(
                subset["LONGITUDE"],
                subset["LATITUDE"],
                color=colors[evt_name],
                s=8,
                alpha=0.6,
                transform=ccrs.PlateCarree(),
            )

            try:
                minx, miny, maxx, maxy = subset[["LONGITUDE", "LATITUDE"]].agg(
                    ["min", "max"]
                ).values.ravel()
                # pad the data bbox, then enforce a minimum window so tight
                # clusters still show enough surrounding geography to orient by
                cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
                span_x = max((maxx - minx) * (1 + 2 * pad_frac), min_span[0])
                span_y = max((maxy - miny) * (1 + 2 * pad_frac), min_span[1])
                # crop to the North America window so AK+CONUS groups don't
                # pad out past the dateline into Asia
                x0 = max(cx - span_x / 2, DEFAULT_EXTENT[0])
                x1 = min(cx + span_x / 2, DEFAULT_EXTENT[1])
                y0 = max(cy - span_y / 2, DEFAULT_EXTENT[2])
                y1 = min(cy + span_y / 2, DEFAULT_EXTENT[3])
                ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
            except Exception:
                ax.set_extent(DEFAULT_EXTENT, crs=ccrs.PlateCarree())

            count_here = counts[evt_name]
            ax.set_title(f"{evt_name}\n(n={count_here})", fontsize=9, pad=6)

        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.3)
        gl.top_labels = False
        gl.right_labels = False

    if len(top_evts) < len(axes_flat):
        for ax in axes_flat[len(top_evts) :]:
            ax.set_visible(False)

    plt.tight_layout()
    finish_fig(f"category_facets_{slugify(col_name)}_minfires{min_fires}", save_figs, fig=fig)


def plot_category_counts(df, col="EVT_class", figsize=(12, 7), save_figs=False):
    """Bar chart of fires per category, sorted (NaN included).

    ``save_figs=True`` saves the figure to ``FIGS_DIR``.
    """
    counts = df[col].value_counts(dropna=False).sort_values(ascending=False)

    plt.figure(figsize=figsize)
    counts.plot(kind="bar")

    plt.xlabel("EVT Class Code")
    plt.ylabel("Number of Fires")
    plt.title("Number of Fires Classified by EVT Class (Sorted)")
    plt.tight_layout()
    finish_fig(f"category_counts_{slugify(col)}", save_figs)
    return counts


def _annotate_cells(ax, values, fmt, fontsize):
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iat[i, j]
            if value > 0:
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=fontsize, color="black")


def plot_confusion_matrix(confusion: pd.DataFrame, normalize: bool = False, save_figs: bool = False):
    """Heatmap of a GFA (rows) × MODIS (columns) confusion table.

    With ``normalize=True`` rows are divided by their totals. ``save_figs=True``
    saves the figure to ``FIGS_DIR``.
    """
    values = confusion.div(confusion.sum(axis=1), axis=0) if normalize else confusion

    plt.figure(figsize=(10, 8))
    plt.imshow(values, cmap="Blues", aspect="auto")
    plt.xticks(range(len(confusion.columns)), confusion.columns, rotation=45, ha="right")
    plt.yticks(range(len(confusion.index)), confusion.index)

    plt.colorbar(label="Proportion" if normalize else "Count")
    plt.title("Normalized Confusion Matrix (Row-wise)" if normalize else "Confusion Matrix Heatmap (GFA vs MODIS)",
              fontsize=14)
    plt.xlabel("MODIS class")
    plt.ylabel("GFA class")

    _annotate_cells(plt.gca(), values, ".2f" if normalize else "", 8)

    plt.tight_layout()
    finish_fig("confusion_matrix_normalized" if normalize else "confusion_matrix", save_figs)


def plot_confusion_by_period(df, periods: dict, ref_col="gfa_class", pred_col="modis_class", year_col="year",
                             save_figs: bool = False):
    """Row-normalized confusion matrices for each ``{label: (start, end)}`` period.

    ``save_figs=True`` saves the figure to ``FIGS_DIR``.

    Returns:
        ``{label: metrics dict or None}`` (see ``reporting.tables.period_metrics``).
    """
    fig, axes = plt.subplots(1, len(periods), figsize=(6 * len(periods), 6), sharey=True)
    metrics = {}

    for ax, (label, (start, end)) in zip(np.atleast_1d(axes), periods.items()):
        subset = df[(df[year_col] >= start) & (df[year_col] <= end)]
        confusion = pd.crosstab(subset[ref_col], subset[pred_col])
        confusion_normalized = confusion.div(confusion.sum(axis=1), axis=0).fillna(0)

        im = ax.imshow(confusion_normalized, cmap="Blues", aspect="auto")
        ax.set_title(label, fontsize=14)
        ax.set_xticks(range(len(confusion.columns)))
        ax.set_xticklabels(confusion.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(confusion.index)))
        ax.set_yticklabels(confusion.index, fontsize=8)

        _annotate_cells(ax, confusion_normalized, ".2f", 6)

        true_labels, pred_labels = confusion_to_labels(confusion)
        metrics[label] = period_metrics(true_labels, pred_labels) if true_labels else None

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04, label="Proportion")
    plt.suptitle("Normalized Confusion Matrices by Time Period", fontsize=16)
    plt.tight_layout()
    finish_fig("confusion_by_period", save_figs, fig=fig)
    return metrics
