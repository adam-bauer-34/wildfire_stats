"""Figures comparing fire-size distributions across datasets (notebook 1)."""

import matplotlib.pyplot as plt
import numpy as np

from wildfire_stats.fitting.stats_helpers.kuiper import kuiper_scan
from wildfire_stats.plotting.presets import finish_fig, get_presets, slugify


def _fig_tag(labels):
    """Filename-safe tag built from the two dataset labels, e.g. ``MTBS_vs_Idaho``."""
    return f"{slugify(labels[0])}_vs_{slugify(labels[1])}"


def _plot_ecdf_pair(df1, df2, labels, area_col, title, name, save_figs):
    plotting_presets, _ = get_presets()
    with plt.rc_context(plotting_presets):
        plt.figure()

        # Compute and plot empirical CDFs
        for data, label in [(df1, labels[0]), (df2, labels[1])]:
            sizes = np.sort(data[area_col].values)
            cdf = np.arange(1, len(sizes) + 1) / len(sizes)
            plt.plot(sizes, cdf, label=label)

        plt.xscale("log")
        plt.xlabel("Fire size (km², log scale)")
        plt.ylabel("CDF")
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        finish_fig(name, save_figs)


def plot_cdf_vs_area(
    df1, df2, area_col="area_km2", labels=("MTBS", "Idaho"), save_figs=False
):
    """Plot empirical CDFs of fire sizes for two datasets on a log-x axis.

    Args:
        df1, df2 (DataFrame): Datasets containing the fire size column.
        area_col (str, optional): Fire area column (km²).
        labels (tuple[str, str]): Legend labels for ``df1`` and ``df2``.
        save_figs (bool, optional): If True, save the figure to ``FIGS_DIR``.
    """
    _plot_ecdf_pair(
        df1,
        df2,
        labels,
        area_col,
        "CDF of Fire Sizes",
        name=f"cdf_vs_area_{_fig_tag(labels)}",
        save_figs=save_figs,
    )


def plot_alpha_vs_kuiper_with_min_cdf(
    df1, df2, alphas, area_col="area_km2", labels=("MTBS", "Idaho"), save_figs=False
):
    """Analyze sensitivity of Kuiper statistic across fire size thresholds.

    For a range of `alpha` thresholds, computes the Kuiper statistic, plots it
    against alpha, highlights the minimizing alpha, and re-plots both CDFs
    filtered at that alpha. This is where ``min_fire`` comes from.

    Args:
        df1, df2 (DataFrame): Datasets containing the fire size column.
        alphas (Iterable[float]): Threshold values (km²) to evaluate.
        area_col (str, optional): Fire size column.
        labels (tuple[str, str]): Legend labels for ``df1`` and ``df2``.
        save_figs (bool, optional): If True, save both figures to ``FIGS_DIR``.

    Returns:
        tuple: (alpha minimizing the Kuiper statistic, minimum Kuiper value).
    """
    kuipers, min_alpha, min_kuiper = kuiper_scan(df1, df2, alphas, area_col)
    tag = _fig_tag(labels)

    # Plot alpha vs. Kuiper statistic
    plotting_presets, _ = get_presets()
    with plt.rc_context(plotting_presets):
        plt.figure()
        plt.plot(alphas, kuipers, marker="o")
        plt.axvline(
            min_alpha,
            color="red",
            linestyle="--",
            label=f"Min Kuiper at alpha={min_alpha:.2f}",
        )
        plt.xlabel("Alpha (km² threshold)")
        plt.ylabel("Kuiper Statistic")
        plt.title("Alpha vs. Kuiper Statistic")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        finish_fig(f"alpha_vs_kuiper_{tag}", save_figs)

    print(
        f"Minimum Kuiper statistic {min_kuiper:.4f} occurs at alpha = {min_alpha:.4f} km²"
    )

    # Plot CDFs for datasets filtered at the minimizing alpha
    _plot_ecdf_pair(
        df1[df1[area_col] > min_alpha],
        df2[df2[area_col] > min_alpha],
        labels,
        area_col,
        f"CDF of Fire Sizes for alpha > {min_alpha:.4f} km² (min Kuiper)",
        name=f"cdf_vs_area_min_kuiper_{tag}",
        save_figs=save_figs,
    )

    return min_alpha, min_kuiper
