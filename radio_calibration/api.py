"""Interface for another system: pretrain, fine-tune, save, load, and use a radio-map model.

The model corrects the Sionna RT path gain map of each cell with a U-Net. The caller
calculates the input maps one time with `radio_calibration.simulation.compute_map_inputs`
and gives them to each method. The methods do no ray tracing.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from radio_calibration.calibration import (
    FEATURE_NAMES,
    Normalization,
    UNet,
    fine_tune,
    load_model,
    log_error,
    predict_maps_db,
    pretrain,
    save_model,
    sensitivity_floor_db,
)
from radio_calibration.core import (
    CalibrationSettings,
    Cell,
    MapInputs,
    SimulationSettings,
    check_cells,
    check_measurements,
    check_positions,
    rsrp_from_map,
)
from radio_calibration.evaluation import error_statistics, grouped_statistics, map_statistics
from radio_calibration.simulation import compute_map_inputs, load_rt_scene

__all__ = [
    "CalibrationSettings",
    "Cell",
    "MapInputs",
    "RadioMapModel",
    "SimulationSettings",
    "compute_map_inputs",
    "error_statistics",
    "grouped_statistics",
    "load_rt_scene",
    "map_statistics",
    "sensitivity_floor_db",
]


def _check_cells_of(inputs: MapInputs, cells: Sequence[Cell]) -> None:
    check_cells(cells)
    if len(cells) != inputs.num_cells:
        raise ValueError(f"There are {len(cells)} cells, but the maps have {inputs.num_cells}.")


class RadioMapModel:
    """A U-Net that corrects simulated path gain maps.

    Args:
        network: The network.
        normalization: The statistics of the pretraining inputs.
        device: The PyTorch device of the network.
        channels: The input channels of the network.
        residual: If False (the default), the network output is the normalized path gain in
            dB. If True, the network output is added to the simulated path gain in dB.
    """

    def __init__(
        self,
        network: UNet,
        normalization: Normalization,
        device: torch.device,
        channels: Sequence[str] = FEATURE_NAMES,
        residual: bool = False,
    ) -> None:
        self.network = network.to(device)
        self.normalization = normalization
        self.device = device
        self.channels = tuple(channels)
        self.residual = residual

    @classmethod
    def pretrain(
        cls,
        inputs: Sequence[MapInputs],
        labels: Sequence[np.ndarray],
        calibration: CalibrationSettings,
        seed: int,
        device: torch.device,
        floor_db: float,
        channels: Sequence[str] = FEATURE_NAMES,
        residual: bool = False,
    ) -> tuple["RadioMapModel", pd.DataFrame]:
        """Train a new model on simulated scene variants.

        Args:
            inputs: The input maps of each variant.
            labels: The target linear path gain of each variant.
            calibration: The settings.
            seed: The seed.
            device: The PyTorch device.
            floor_db: The floor of the path gain in dB. Use `sensitivity_floor_db` with the
                smallest RSRP that a UE reports and the powers of the cells.
            channels: The input channels.
            residual: The output type.

        Returns:
            The model and the history of the pretraining.

        Raises:
            ValueError: As for `radio_calibration.calibration.pretrain`.
        """
        result = pretrain(inputs, labels, calibration, seed, device, floor_db, channels, residual)
        model = cls(result.network, result.normalization, device, channels, residual)
        return model, result.history

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "RadioMapModel":
        """Read a model from a `.pt` file of `save`."""
        network, normalization, channels, residual = load_model(path, device)
        return cls(network, normalization, device, channels, residual)

    def save(self, path: Path) -> None:
        """Write the model to a `.pt` file."""
        save_model(path, self.network, self.normalization, self.channels, self.residual)

    def fine_tune(
        self,
        inputs: MapInputs,
        positions: np.ndarray,
        rsrp_dbm: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray],
        cells: Sequence[Cell],
        calibration: CalibrationSettings,
        seed: int,
    ) -> pd.DataFrame:
        """Change all parameters of the model to agree with the measured RSRP.

        Args:
            inputs: The input maps of the cells.
            positions: Train positions in meters, with shape `[n, 3]`.
            rsrp_dbm: Measured train RSRP in dBm, with shape `[n, len(cells)]`. NaN shows
                that the record does not have the cell.
            validation: Validation positions and RSRP, with the same format.
            cells: The cells, in the order of the maps.
            calibration: The settings.
            seed: The seed of the batch order.

        Returns:
            The history of the fine-tuning.

        Raises:
            ValueError: If the data is not in the fixed format, or if no train record has a
                received pair in the grid.
        """
        _check_cells_of(inputs, cells)
        check_measurements(positions, rsrp_dbm, len(cells))
        check_measurements(*validation, len(cells))
        return fine_tune(
            self.network,
            self.normalization,
            inputs,
            positions,
            rsrp_dbm,
            validation,
            np.array([cell.power_rs_dbm for cell in cells], dtype=float),
            calibration,
            seed,
            self.device,
            self.channels,
            self.residual,
        )

    def predict_path_gain_db(self, inputs: MapInputs) -> np.ndarray:
        """Calculate the corrected path gain map of each cell.

        Returns:
            The path gain in dB, with shape `[n_cells, ny, nx]`.
        """
        return predict_maps_db(
            self.network, self.normalization, inputs, self.device, self.channels, self.residual
        )

    def predict_rsrp_dbm(
        self, inputs: MapInputs, positions: np.ndarray, cells: Sequence[Cell]
    ) -> np.ndarray:
        """Calculate the RSRP of each cell at each position.

        The RSRP is the reference-signal power of the cell plus the corrected path gain of
        the grid cell that holds the position.

        Args:
            inputs: The input maps of the cells.
            positions: Positions in meters, with shape `[n, 3]`.
            cells: The cells, in the order of the maps.

        Returns:
            The RSRP in dBm, with shape `[n, len(cells)]`. A position outside the grid gives
            NaN.

        Raises:
            ValueError: If the positions or the cells are not in the fixed format.
        """
        _check_cells_of(inputs, cells)
        check_positions(positions)
        power_rs_dbm = np.array([cell.power_rs_dbm for cell in cells], dtype=float)
        return rsrp_from_map(
            self.predict_path_gain_db(inputs), inputs.grid, positions, power_rs_dbm
        )

    def errors_db(
        self,
        inputs: MapInputs,
        positions: np.ndarray,
        rsrp_dbm: np.ndarray,
        cells: Sequence[Cell],
    ) -> np.ndarray:
        """Calculate the measured minus the predicted RSRP of each pair, in dB.

        Returns:
            The error with shape `[n, len(cells)]`. NaN where there is no measurement or the
            position is outside the grid.

        Raises:
            ValueError: If the data is not in the fixed format.
        """
        check_measurements(positions, rsrp_dbm, len(cells))
        return log_error(rsrp_dbm, self.predict_rsrp_dbm(inputs, positions, cells))
