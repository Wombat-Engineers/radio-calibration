"""Shared utilities: paths, configuration, figures, tables, seeds, device, notebook setup."""

from utils.config import calibration_settings, load_config, scene_source, simulation_settings
from utils.notebook import setup_notebook
from utils.output import configure_output, save_figure, save_table
from utils.paths import CONFIG_DIR, DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT, SCENES_DIR
from utils.plotting import (
    COLORS,
    FIGURE_WIDTH_DOUBLE,
    FIGURE_WIDTH_SINGLE,
    configure_matplotlib,
)
from utils.reproducibility import get_device, seed_everything

__all__ = [
    "COLORS",
    "CONFIG_DIR",
    "DATA_DIR",
    "FIGURE_WIDTH_DOUBLE",
    "FIGURE_WIDTH_SINGLE",
    "OUTPUTS_DIR",
    "PROJECT_ROOT",
    "SCENES_DIR",
    "calibration_settings",
    "configure_matplotlib",
    "configure_output",
    "get_device",
    "load_config",
    "save_figure",
    "save_table",
    "scene_source",
    "seed_everything",
    "setup_notebook",
    "simulation_settings",
]
