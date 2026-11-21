"""Data of one scenario, as the notebooks 03, 04, and 05 read it."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from preprocessing import SPLIT_NAMES, MapsFile, mdt_to_arrays, read_cells, read_maps
from radio_calibration.calibration import sensitivity_floor_db
from radio_calibration.core import Cell
from utils import ScenarioDirs, scenario_dirs


@dataclass(frozen=True)
class ScenarioData:
    """The data of one scenario.

    Attributes:
        dirs: The directories of the scenario.
        cells: The cells, in the order of the maps and of the RSRP columns.
        parts: The MDT records of each part of `SPLIT_NAMES`, in the long format.
        arrays: The positions `[n, 3]` and the RSRP `[n, n_cells]` of each part.
        maps: The content of `maps.npz`.
    """

    dirs: ScenarioDirs
    cells: list[Cell]
    parts: dict[str, pd.DataFrame]
    arrays: dict[str, tuple[np.ndarray, np.ndarray]]
    maps: MapsFile

    @property
    def power_rs_dbm(self) -> np.ndarray:
        """Reference-signal power of each cell, in dBm."""
        return np.array([cell.power_rs_dbm for cell in self.cells], dtype=float)


def measurable_db(cfg: DictConfig, data: ScenarioData) -> float:
    """Return the smallest path gain in dB that a UE can report for the strongest cell."""
    return float(cfg.scenario.measurement.min_rsrp_dbm - data.power_rs_dbm.max())


def floor_db(cfg: DictConfig, data: ScenarioData) -> float:
    """Return the floor of the path gain of the model, from `sensitivity_floor_db`."""
    return sensitivity_floor_db(cfg.scenario.measurement.min_rsrp_dbm, data.power_rs_dbm)


def load_scenario(cfg: DictConfig) -> ScenarioData:
    """Read `cells.csv` and `maps.npz` of `raw/`, and the parts of `processed/`.

    Run `task scenario` and the notebook `02_preprocessing.ipynb` of the scenario first.

    Args:
        cfg: The configuration from `load_config`. The function uses `scenario.name`.

    Returns:
        The data.
    """
    dirs = scenario_dirs(cfg.scenario.name)
    cells = read_cells(dirs.raw / "cells.csv")
    parts = {name: pd.read_parquet(dirs.processed / f"{name}.parquet") for name in SPLIT_NAMES}
    arrays = {name: mdt_to_arrays(part, cells) for name, part in parts.items()}
    return ScenarioData(dirs, cells, parts, arrays, read_maps(dirs.raw / "maps.npz"))


def first_records(
    arrays: tuple[np.ndarray, np.ndarray], size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first `size` records of one random order of the records.

    The same seed gives the same order for each size, thus a smaller set is a part of a
    larger set.

    Raises:
        ValueError: If `size` is less than 1 or more than the number of records.
    """
    positions, rsrp = arrays
    if not 1 <= size <= len(positions):
        raise ValueError(f"size must be from 1 to {len(positions)}, but it is {size}.")
    order = np.random.default_rng(seed).permutation(len(positions))[:size]
    return positions[order], rsrp[order]
