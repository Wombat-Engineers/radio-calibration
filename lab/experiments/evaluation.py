"""Baseline maps and error tables of the models of one scenario.

Each model gives one path gain map in dB for each cell. The RSRP of a record comes from the
grid cell that holds its reported position (`rsrp_from_map`), thus all models use the same
rule. The functions do no ray tracing.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from experiments.data import ScenarioData, floor_db, measurable_db
from radio_calibration.calibration import base_path_gain_db
from radio_calibration.core import rsrp_from_map
from radio_calibration.evaluation import (
    SpmModel,
    error_statistics,
    fit_offset,
    grouped_statistics,
    map_statistics,
    spm_inputs,
)

SIONNA_RT = "Sionna RT"
SIONNA_RT_OFFSET = "Sionna RT + offset"
SPM_ATOLL = "SPM (Atoll form)"


def antenna_heights(cfg: DictConfig, data: ScenarioData) -> np.ndarray:
    """Return the antenna height of each cell above the ground, in meters."""
    ground_z = data.maps.target.grid.z - cfg.scenario.ue_height_m
    return np.array([cell.z - ground_z for cell in data.cells], dtype=float)


def map_rsrp_dbm(data: ScenarioData, path_gain_db: np.ndarray, part: str) -> np.ndarray:
    """Return the RSRP of the records of a part, from a path gain map in dB."""
    positions, _ = data.arrays[part]
    return rsrp_from_map(path_gain_db, data.maps.target.grid, positions, data.power_rs_dbm)


def spm_map_db(
    cfg: DictConfig, data: ScenarioData, train: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Fit the SPM model on train records, then give its path gain map in dB.

    Args:
        cfg: The configuration.
        data: The data of the scenario.
        train: Positions `[n, 3]` and RSRP `[n, n_cells]` of the fit.

    Returns:
        The path gain in dB, with shape `[n_cells, ny, nx]`.
    """
    target = data.maps.target
    heights = antenna_heights(cfg, data)
    positions, rsrp = train
    distance, height, los, clutter, inside = spm_inputs(target, heights, positions)
    path_loss = data.power_rs_dbm - rsrp
    valid = np.isfinite(path_loss) & inside[:, None]
    model = SpmModel.fit(
        distance[valid], height[valid], los[valid], clutter[valid], path_loss[valid]
    )
    centers = target.grid.centers().reshape(-1, 3)
    distance, height, los, clutter, _ = spm_inputs(target, heights, centers)
    path_gain = -model.path_loss_db(distance, height, los, clutter)
    return path_gain.T.reshape(target.num_cells, target.grid.ny, target.grid.nx)


def baseline_maps(
    cfg: DictConfig, data: ScenarioData, train: tuple[np.ndarray, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """Return the path gain maps in dB of the three baselines.

    - Sionna RT on the original scene. A grid cell with no ray or a path gain below the floor
      gets the floor, as in the input of the U-Net.
    - Sionna RT plus one offset that the train records give.
    - The SPM model with coefficients from the train records.

    Args:
        cfg: The configuration.
        data: The data of the scenario.
        train: Positions and RSRP of the fits. The default is the full train part. Give a
            smaller set to compare with a model that was fine-tuned on the same set.

    Returns:
        One map with shape `[n_cells, ny, nx]` for each baseline name.
    """
    positions, rsrp = data.arrays["train"] if train is None else train
    sionna = base_path_gain_db(data.maps.target, floor_db(cfg, data))
    simulated = rsrp_from_map(sionna, data.maps.target.grid, positions, data.power_rs_dbm)
    return {
        SIONNA_RT: sionna,
        SIONNA_RT_OFFSET: sionna + fit_offset(rsrp, simulated),
        SPM_ATOLL: spm_map_db(cfg, data, (positions, rsrp)),
    }


def evaluated_cells(cfg: DictConfig, data: ScenarioData) -> np.ndarray:
    """Return the indices of the cells of the evaluation.

    With the split method `site`, these are the cells of the test sites. Else all cells.
    """
    split = cfg.scenario.split
    if split.method == "site":
        sites = set(split.test_sites)
        return np.array([i for i, cell in enumerate(data.cells) if cell.site in sites])
    return np.arange(len(data.cells))


def mdt_table(
    cfg: DictConfig, data: ScenarioData, maps_db: Mapping[str, np.ndarray], part: str = "test"
) -> pd.DataFrame:
    """Return the error statistics of each model on the records of a part, one row each."""
    _, measured = data.arrays[part]
    rows = {
        name: error_statistics(
            measured, map_rsrp_dbm(data, path_gain_db, part), cfg.evaluation.quantiles
        )
        for name, path_gain_db in maps_db.items()
    }
    return pd.DataFrame(rows).T


def cell_table(
    cfg: DictConfig, data: ScenarioData, maps_db: Mapping[str, np.ndarray], part: str = "test"
) -> pd.DataFrame:
    """Return the RMSE in dB of each model (rows) for each cell (columns) on a part."""
    _, measured = data.arrays[part]
    names = np.broadcast_to(np.array([cell.name for cell in data.cells]), measured.shape)
    columns = {
        name: grouped_statistics(
            measured, map_rsrp_dbm(data, path_gain_db, part), names, cfg.evaluation.quantiles
        )["rmse_db"]
        for name, path_gain_db in maps_db.items()
    }
    return pd.DataFrame(columns).T


def map_table(
    cfg: DictConfig, data: ScenarioData, maps_db: Mapping[str, np.ndarray]
) -> pd.DataFrame | None:
    """Return the error statistics of each model against the true map, one row each.

    The statistics use the cells of `evaluated_cells`, and only the grid cells where a UE
    can measure the true path gain (`measurable_db`). The result is None if the scenario has
    no true map.
    """
    if data.maps.true_path_gain is None:
        return None
    cells = evaluated_cells(cfg, data)
    truth = data.maps.true_path_gain[cells]
    threshold = measurable_db(cfg, data)
    rows = {
        name: map_statistics(truth, path_gain_db[cells], cfg.evaluation.quantiles, threshold)
        for name, path_gain_db in maps_db.items()
    }
    return pd.DataFrame(rows).T
