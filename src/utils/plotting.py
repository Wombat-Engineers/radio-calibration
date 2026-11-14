"""Shared matplotlib style for figures in a scientific report."""

import matplotlib as mpl

# Widths of a figure in a two-column IEEE paper, in inches: 3.5 in for one column and
# 7.16 in for two columns (IEEE Author Center, "Preparing your graphics").
FIGURE_WIDTH_SINGLE = 3.5
FIGURE_WIDTH_DOUBLE = 7.16

# Colors that persons with a color vision deficiency can tell apart
# (Wong, "Points of view: Color blindness", Nature Methods, 2011, doi:10.1038/nmeth.1618).
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def configure_matplotlib() -> None:
    """Apply the project figure style to the global matplotlib rcParams.

    The style gives figures that are legible at the width of a paper column: a serif font,
    8 pt to 9 pt text, thin lines, and colors that are safe for color vision deficiencies.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.constrained_layout.use": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.4,
            "axes.prop_cycle": mpl.cycler(color=COLORS),
            "image.cmap": "viridis",
        }
    )
