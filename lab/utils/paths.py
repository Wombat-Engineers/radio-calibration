"""Absolute project paths, so code works from any working directory."""

import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "lab" / "data"
OUTPUTS_DIR = PROJECT_ROOT / "lab" / "outputs"
CONFIG_DIR = PROJECT_ROOT / "lab" / "configs"
NOTEBOOK_DIR = PROJECT_ROOT / "lab" / "notebooks"
SCENES_DIR = DATA_DIR / "external" / "scenes"


@dataclass(frozen=True)
class ScenarioDirs:
    """Directories of one scenario.

    Attributes:
        raw: Generated data: `cells.csv`, `mdt.csv`, `maps.npz`.
        processed: Validated and divided MDT records.
        outputs: Models, histories, notebook copies, and Hydra runs of the scenario.
        notebooks: The notebooks of the scenario.
    """

    raw: Path
    processed: Path
    outputs: Path
    notebooks: Path


def scenario_dirs(name: str) -> ScenarioDirs:
    """Return the directories of a scenario.

    Raises:
        ValueError: If the name has a character that is not a letter, a digit, or `_`.
    """
    # The name becomes a part of a path. This check prevents a path outside the directories.
    if not re.fullmatch(r"\w+", name):
        raise ValueError(f"The scenario name '{name}' must have only letters, digits, and '_'.")
    return ScenarioDirs(
        raw=DATA_DIR / name / "raw",
        processed=DATA_DIR / name / "processed",
        outputs=OUTPUTS_DIR / name,
        notebooks=NOTEBOOK_DIR / name,
    )
