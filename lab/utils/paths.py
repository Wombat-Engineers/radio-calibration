"""Absolute project paths, so code works from any working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "lab" / "data"
OUTPUTS_DIR = PROJECT_ROOT / "lab" / "outputs"
CONFIG_DIR = PROJECT_ROOT / "lab" / "configs"
SCENES_DIR = DATA_DIR / "external" / "scenes"
