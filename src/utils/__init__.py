"""Shared utilities: paths, figure style, seeds, device, and notebook setup."""

from src.utils.notebook import setup_notebook
from src.utils.paths import DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT
from src.utils.plotting import configure_matplotlib
from src.utils.reproducibility import get_device, seed_everything

__all__ = [
    "DATA_DIR",
    "OUTPUTS_DIR",
    "PROJECT_ROOT",
    "configure_matplotlib",
    "get_device",
    "seed_everything",
    "setup_notebook",
]
