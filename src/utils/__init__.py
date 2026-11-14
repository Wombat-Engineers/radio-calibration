"""Shared utilities: paths, configuration, figures, tables, seeds, device, notebook setup."""

from src.utils.config import load_config
from src.utils.notebook import setup_notebook
from src.utils.output import configure_output, save_figure, save_table
from src.utils.paths import CONFIG_DIR, DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT
from src.utils.plotting import (
    COLORS,
    FIGURE_WIDTH_DOUBLE,
    FIGURE_WIDTH_SINGLE,
    configure_matplotlib,
)
from src.utils.reproducibility import get_device, seed_everything

__all__ = [
    "COLORS",
    "CONFIG_DIR",
    "DATA_DIR",
    "FIGURE_WIDTH_DOUBLE",
    "FIGURE_WIDTH_SINGLE",
    "OUTPUTS_DIR",
    "PROJECT_ROOT",
    "configure_matplotlib",
    "configure_output",
    "get_device",
    "load_config",
    "save_figure",
    "save_table",
    "seed_everything",
    "setup_notebook",
]
