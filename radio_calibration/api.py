"""Interface for another system: calibrate, save, load, and use a neural material.

The data has a fixed format:

- `cells`: a sequence of `Cell`. All cells have the same antenna array.
- `positions`: UE positions in meters, with shape `[n, 3]`.
- `rsrp_dbm`: measured RSRP in dBm, with shape `[n, len(cells)]`. NaN shows that the
  record does not have the cell.

The import of this module sets the Mitsuba variant `cuda_ad_mono_polarized` for the full
process. The functions need an NVIDIA GPU.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from radio_calibration.calibration import calibrate, load_calibration, log_error, save_calibration
from radio_calibration.core import (
    CalibrationSettings,
    Cell,
    SimulationSettings,
    check_cells,
    check_measurements,
    check_positions,
)
from radio_calibration.evaluation import error_statistics, grouped_statistics, nearest_distance
from radio_calibration.simulation import (
    NeuralRadioMaterial,
    apply_neural_material,
    assign_material,
    compute_rsrp_dbm,
    load_rt_scene,
    make_material_network,
)

if TYPE_CHECKING:
    import sionna.rt

__all__ = [
    "CalibratedModel",
    "CalibrationSettings",
    "Cell",
    "SimulationSettings",
    "error_statistics",
    "grouped_statistics",
    "load_rt_scene",
    "nearest_distance",
]

_LOADED_MATERIAL_NAME = "calibrated"


class CalibratedModel:
    """A scene, its cells, and a calibrated neural material with its power scale.

    Args:
        scene: The Sionna RT scene, from `load_rt_scene`. It must have no transmitter and
            no receiver.
        cells: The cells.
        material: A calibrated material. Its scale must not be None.
        simulation: The settings of the simulation.

    Raises:
        ValueError: If the cells are not valid (see `check_cells`), or if the material has
            no power scale.
    """

    def __init__(
        self,
        scene: "sionna.rt.Scene",
        cells: Sequence[Cell],
        material: NeuralRadioMaterial,
        simulation: SimulationSettings,
    ) -> None:
        check_cells(cells)
        if material.scale is None:
            raise ValueError("The material has no power scale. Calibrate it before you use it.")
        self.scene = scene
        self.cells = list(cells)
        self.material = material
        self.simulation = simulation

    @classmethod
    def fit(
        cls,
        scene: "sionna.rt.Scene",
        cells: Sequence[Cell],
        positions: np.ndarray,
        rsrp_dbm: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray],
        calibration: CalibrationSettings,
        simulation: SimulationSettings,
        seed: int,
    ) -> tuple["CalibratedModel", pd.DataFrame]:
        """Calibrate a new neural material against measured RSRP (arXiv:2311.18558).

        The function gives the new material to all objects of `scene`. It replaces the
        materials of the scene file.

        Args:
            scene: The Sionna RT scene. It must have no transmitter and no receiver.
            cells: The cells.
            positions: Positions of the train records in meters, with shape `[n, 3]`.
            rsrp_dbm: Measured RSRP of the train records in dBm, with shape
                `[n, len(cells)]`.
            validation: Positions and measured RSRP of the validation records, with the
                same format.
            calibration: The settings of the calibration.
            simulation: The settings of the simulation.
            seed: Seed of the initial weights and of the random order of the records.

        Returns:
            The calibrated model, and the history of the steps (one row for each step, with
            the columns `step`, `epoch`, `loss`, `scale`, `validation_loss`, and
            `step_seconds`).

        Raises:
            ValueError: If the cells or the measurements are not valid, if there is no train
                record, or if all simulated powers of a batch are 0.
        """
        check_cells(cells)
        check_measurements(positions, rsrp_dbm, len(cells))
        check_measurements(*validation, len(cells))
        network = make_material_network(
            calibration.encoding_octaves,
            calibration.hidden_layers,
            calibration.hidden_units,
            seed=seed,
        )
        material = apply_neural_material(scene, network)
        result = calibrate(
            scene,
            cells,
            material,
            np.asarray(positions, dtype=float),
            np.asarray(rsrp_dbm, dtype=float),
            (np.asarray(validation[0], dtype=float), np.asarray(validation[1], dtype=float)),
            calibration,
            simulation,
            seed=seed,
        )
        return cls(scene, cells, material, simulation), result.history

    @classmethod
    def load(
        cls,
        scene: "sionna.rt.Scene",
        cells: Sequence[Cell],
        path: Path,
        simulation: SimulationSettings,
    ) -> "CalibratedModel":
        """Read a model that `save` wrote.

        Args:
            scene: The Sionna RT scene of the calibration.
            cells: The cells.
            path: Path of the `.npz` file.
            simulation: The settings of the simulation.

        Returns:
            The model.

        Raises:
            ValueError: If the file is not a valid calibration file, or if the cells are not
                valid.
        """
        return cls(scene, cells, load_calibration(Path(path), _LOADED_MATERIAL_NAME), simulation)

    def save(self, path: Path) -> None:
        """Write the material and its power scale to one `.npz` file.

        Args:
            path: Path of the file. Its suffix must be `.npz`.

        Raises:
            ValueError: If the suffix of `path` is not `.npz`.
        """
        save_calibration(Path(path), self.material)

    def predict_rsrp_dbm(self, positions: np.ndarray) -> np.ndarray:
        """Calculate the RSRP of each cell at each position, with the power scale.

        The function gives the material of the model to all objects of the scene before the
        calculation. Thus two models can use the same scene.

        Args:
            positions: UE positions in meters, with shape `[n, 3]`.

        Returns:
            The RSRP in dBm, with shape `[n, len(cells)]`. A value is NaN if no path
            connects the cell and the position.

        Raises:
            ValueError: If `positions` is not valid (see `check_positions`).
        """
        check_positions(positions)
        assign_material(self.scene, self.material)
        return compute_rsrp_dbm(
            self.scene,
            self.cells,
            np.asarray(positions, dtype=float),
            self.simulation,
            self.material.scale,
        )

    def errors_db(self, positions: np.ndarray, rsrp_dbm: np.ndarray) -> np.ndarray:
        """Calculate the signed error between the measured and the predicted RSRP.

        Args:
            positions: UE positions in meters, with shape `[n, 3]`.
            rsrp_dbm: Measured RSRP in dBm, with shape `[n, len(cells)]`.

        Returns:
            The measured RSRP minus the predicted RSRP, in dB, with shape
            `[n, len(cells)]`. A value is NaN if the record does not have the cell or if no
            path connects the cell and the position.

        Raises:
            ValueError: If the measurements are not valid (see `check_measurements`).
        """
        check_measurements(positions, rsrp_dbm, len(self.cells))
        return log_error(np.asarray(rsrp_dbm, dtype=float), self.predict_rsrp_dbm(positions))
