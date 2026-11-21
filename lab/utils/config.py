"""Composition of the configuration in `lab/configs/`, and its change to typed settings."""

import re
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from radio_calibration.core import CalibrationSettings, SimulationSettings
from utils import paths


def load_config(*overrides: str) -> DictConfig:
    """Compose `lab/configs/config.yaml` as `task scenario` composes it.

    The result has the keys `seed`, `scenario`, `simulation`, `calibration`, `evaluation`,
    and `output`. It has no `hydra` key.

    Args:
        overrides: Hydra overrides, for example `"scenario=munich_new_site"` or
            `"scenario.sites.isd_m=60"`.

    Returns:
        The configuration.

    Raises:
        hydra.errors.ConfigCompositionException: If an override names a key that is not in
            the configuration.
    """
    with initialize_config_dir(config_dir=str(paths.CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def simulation_settings(cfg: DictConfig) -> SimulationSettings:
    """Make the simulation settings from the key `simulation` of the configuration.

    Args:
        cfg: The configuration from `load_config`.

    Returns:
        The settings. `radio_map.seed` has the value of the root `seed`.

    Raises:
        ValueError: If a key is missing, or if a value is out of its range.
    """
    return SimulationSettings.from_mapping(OmegaConf.to_container(cfg.simulation, resolve=True))


def calibration_settings(cfg: DictConfig) -> CalibrationSettings:
    """Make the calibration settings from the key `calibration` of the configuration.

    Args:
        cfg: The configuration from `load_config`.

    Returns:
        The settings.

    Raises:
        ValueError: If a key is missing, or if a value is out of its range.
    """
    return CalibrationSettings.from_mapping(OmegaConf.to_container(cfg.calibration, resolve=True))


def scene_source(cfg: DictConfig) -> str | Path:
    """Give the argument of `load_rt_scene` for the scene of the configuration.

    A file `<name>.xml` in `lab/data/external/scenes/` has priority over a scene of
    Sionna RT with the same name.

    Args:
        cfg: The configuration from `load_config`. The function uses `scenario.scene`.

    Returns:
        The path of the local scene file if it exists, else the name.

    Raises:
        ValueError: If the name has a character that is not a letter, a digit, or `_`.
    """
    name = cfg.scenario.scene
    # The name becomes a part of a file path. This check prevents a path outside SCENES_DIR.
    if not re.fullmatch(r"\w+", name):
        raise ValueError(f"The scene name '{name}' must have only letters, digits, and '_'.")
    path = paths.SCENES_DIR / f"{name}.xml"
    return path if path.is_file() else name
