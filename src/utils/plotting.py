"""Shared matplotlib style."""

import matplotlib as mpl


def configure_matplotlib() -> None:
    """Apply the project figure style to the global matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "image.cmap": "viridis",
        }
    )
