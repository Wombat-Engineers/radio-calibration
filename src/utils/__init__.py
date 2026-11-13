"""Shared utilities: paths, figure style, seeds, device, and notebook setup."""

from src.utils.notebook import setup_notebook
from src.utils.paths import DATA_DIR, FIGURES_DIR, OUTPUTS_DIR, PROJECT_ROOT
from src.utils.plotting import (
    COLORS,
    FIGURE_WIDTH_DOUBLE,
    FIGURE_WIDTH_SINGLE,
    configure_matplotlib,
    save_figure,
)
from src.utils.reproducibility import get_device, seed_everything

__all__ = [
    "COLORS",
    "DATA_DIR",
    "FIGURES_DIR",
    "FIGURE_WIDTH_DOUBLE",
    "FIGURE_WIDTH_SINGLE",
    "OUTPUTS_DIR",
    "PROJECT_ROOT",
    "configure_matplotlib",
    "get_device",
    "save_figure",
    "seed_everything",
    "setup_notebook",
]
