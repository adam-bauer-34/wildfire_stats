"""Plotting preset function.

Here I write a simple function which returns a set of matplotlib parameters and
two base file names to save figures.

Adam Michael Bauer
University of Illinois Urbana-Champaign
World Bank Group
6.19.2023
"""

import os
import re
import datetime

import matplotlib.pyplot as plt

# Default resolution for every saved figure across the plotting package.
FIG_DPI = 300


def slugify(text):
    """Filename-safe version of ``text`` (runs of non-alphanumerics -> ``-``)."""
    return re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-")


def finish_fig(
    name,
    save_figs=False,
    fig=None,
    save_dir=None,
    date_prefix=True,
    dpi=FIG_DPI,
    show=True,
    tag=True,
    **savefig_kwargs,
):
    """Save a figure to ``FIGS_DIR`` (if ``save_figs``), then show it.

    Args:
        name (str): Filename stem (``.png`` is appended).
        save_figs (bool): If True, write the figure to disk.
        fig (Figure, optional): Figure to save; defaults to the current figure.
        save_dir (path, optional): Output directory; defaults to ``paths.figs``.
        date_prefix (bool): Prepend the ``YYYY-M-D-`` prefix from ``get_presets``.
        dpi (int): Output resolution (default ``FIG_DPI`` = 300).
        show (bool): Call ``plt.show()`` afterwards.
        tag (bool): Append ``_<minfire tag>`` (e.g. ``_minfire6p45``) to the
            filename so runs with different ``min_fire`` don't overwrite each other.
        **savefig_kwargs: Extra keyword arguments for ``Figure.savefig``.

    Returns:
        str | None: Path of the saved file, or None if not saved.
    """
    save_path = None
    if save_figs:
        from wildfire_stats.config import load_paths
        from wildfire_stats.data.io import minfire_tag

        if tag:
            name = f"{name}_{minfire_tag()}"
        fig = plt.gcf() if fig is None else fig
        save_dir = os.fspath(load_paths().figs if save_dir is None else save_dir)
        os.makedirs(save_dir, exist_ok=True)
        prefix = os.path.basename(get_presets()[1]) if date_prefix else ""
        save_path = os.path.join(save_dir, f"{prefix}{name}.png")
        savefig_kwargs.setdefault("bbox_inches", "tight")
        fig.savefig(save_path, dpi=dpi, **savefig_kwargs)
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return save_path


def get_presets(markers=False):
    """Get plotting presets.

    Parameters
    ----------
    markers: bool (default=False)
        do you want a marker on each data point?

    Returns
    -------
    plotting_presents: dict
        plotting presets to be passed to plt.rcParams.update

    basefile: string
        filename to save figure to 'figs' folder
    """

    linestyle_list = ["solid", "dashed", "dashdot", "dotted"] * 2
    color_list = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]
    marker_list = ["o", "s", "P", "+", "D", "v", "3", ","]

    if markers:
        plotting_presets = {
            "axes.linewidth": 3,
            "axes.axisbelow": False,
            "axes.edgecolor": "black",
            "axes.facecolor": "None",
            "axes.grid": False,
            "axes.labelcolor": "black",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "axes.titlelocation": "left",
            "figure.facecolor": "white",
            "figure.figsize": (18, 10),
            "lines.solid_capstyle": "round",
            "lines.linewidth": 2.5,
            "patch.edgecolor": "w",
            "patch.force_edgecolor": True,
            "text.color": "black",
            "legend.frameon": False,
            "xtick.bottom": True,
            "xtick.major.width": 3,
            "xtick.major.size": 6,
            "xtick.color": "black",
            "xtick.top": False,
            "ytick.color": "black",
            "ytick.direction": "out",
            "ytick.left": True,
            "ytick.right": False,
            "ytick.color": "black",
            "ytick.major.width": 3,
            "ytick.major.size": 6,
            "axes.prop_cycle": plt.cycler(
                linestyle=linestyle_list, color=color_list, marker=marker_list
            ),
            "font.size": 16,
            "font.family": "sans",
            "savefig.dpi": FIG_DPI,
        }

    else:
        plotting_presets = {
            "axes.linewidth": 3,
            "axes.axisbelow": False,
            "axes.edgecolor": "black",
            "axes.facecolor": "None",
            "axes.grid": False,
            "axes.labelcolor": "black",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "axes.titlelocation": "left",
            "figure.facecolor": "white",
            "figure.figsize": (12, 5.5),
            "lines.solid_capstyle": "round",
            "lines.linewidth": 2.5,
            "patch.edgecolor": "w",
            "patch.force_edgecolor": True,
            "text.color": "black",
            "legend.frameon": False,
            "xtick.bottom": True,
            "xtick.major.width": 3,
            "xtick.major.size": 6,
            "xtick.color": "black",
            "xtick.top": False,
            "ytick.color": "black",
            "ytick.direction": "out",
            "ytick.left": True,
            "ytick.right": False,
            "ytick.color": "black",
            "ytick.major.width": 3,
            "ytick.major.size": 6,
            "axes.prop_cycle": plt.cycler(linestyle=linestyle_list, color=color_list),
            "font.size": 16,
            "font.family": "sans",
            "savefig.dpi": FIG_DPI,
        }

    today = datetime.datetime.now()
    year = str(today.year)
    day = str(today.day)
    month = str(today.month)

    cwd = os.getcwd()
    basefile = "".join([cwd, "/figs/", year, "-", month, "-", day, "-"])

    return plotting_presets, basefile
