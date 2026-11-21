"""Baseline models that the U-Net must be better than.

- Ray tracing with a constant offset: the simulated RSRP plus one offset that the train
  records give. Geo2SigMap (arXiv:2312.14303) calibrates its ray-tracing baseline in the
  same way.
- A model with the form of the Standard Propagation Model (SPM) of the Atoll planning tool
  (Forsk), with the coefficients from a least-squares fit to the train records. Atoll
  calibrates the SPM from measurements in the same way. The form follows "Calibrating the
  Standard Path Loss Model for Urban Environments using Field Measurements and Geospatial
  Data", WCE 2017.

The functions do no ray tracing.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from radio_calibration.core import MapInputs

# A shorter distance becomes this distance, in meters, before the logarithm. Assumption.
_MIN_DISTANCE_M = 1.0


def fit_offset(measured_dbm: np.ndarray, simulated_dbm: np.ndarray) -> float:
    """Calculate the constant that best moves the simulated RSRP to the measured RSRP.

    Args:
        measured_dbm: Measured RSRP in dBm. NaN shows no measurement.
        simulated_dbm: Simulated RSRP in dBm, with the same shape. NaN shows no simulated
            value.

    Returns:
        The mean of `measured - simulated` over the pairs where both values are finite, in
        dB. Add it to the simulated RSRP.

    Raises:
        ValueError: If the shapes are not equal, or if no pair has both values.
    """
    measured_dbm = np.asarray(measured_dbm, dtype=float)
    simulated_dbm = np.asarray(simulated_dbm, dtype=float)
    if measured_dbm.shape != simulated_dbm.shape:
        raise ValueError(
            f"The shapes must be equal, but they are {measured_dbm.shape} and "
            f"{simulated_dbm.shape}."
        )
    difference = measured_dbm - simulated_dbm
    valid = np.isfinite(difference)
    if not valid.any():
        raise ValueError("The offset needs a minimum of one pair with two finite values.")
    return float(np.mean(difference[valid]))


@dataclass(frozen=True)
class SpmModel:
    """Path loss in dB with the form of the Standard Propagation Model.

    `L = K1 + K2 log10(d) + K3 log10(h) + K5 log10(d) log10(h) + K_los LoS + K_clutter c`.
    `d` is the horizontal distance in meters, `h` is the antenna height above the ground
    in meters, `LoS` is 1 for a line of sight, and `c` is the building fraction of the grid
    cell. The diffraction term K4 is not in the model: the LoS term replaces it. The UE
    height term K6 is not in the model: the UE height is constant, thus K1 includes it.

    Attributes:
        coefficients: `[K1, K2, K3, K5, K_los, K_clutter]`.
    """

    coefficients: np.ndarray

    @staticmethod
    def _design(
        distance_m: np.ndarray, height_m: np.ndarray, los: np.ndarray, clutter: np.ndarray
    ) -> np.ndarray:
        log_distance = np.log10(np.maximum(np.asarray(distance_m, dtype=float), _MIN_DISTANCE_M))
        log_height = np.log10(np.asarray(height_m, dtype=float))
        return np.stack(
            [
                np.ones_like(log_distance),
                log_distance,
                log_height,
                log_distance * log_height,
                np.asarray(los, dtype=float),
                np.asarray(clutter, dtype=float),
            ],
            axis=-1,
        )

    @classmethod
    def fit(
        cls,
        distance_m: np.ndarray,
        height_m: np.ndarray,
        los: np.ndarray,
        clutter: np.ndarray,
        path_loss_db: np.ndarray,
    ) -> "SpmModel":
        """Fit the coefficients with least squares.

        `numpy.linalg.lstsq` gives the solution with the smallest norm. Thus a coefficient
        that the data cannot identify (for example K3 when all antennas have the same
        height) causes no error.

        Args:
            distance_m: Horizontal distance of each pair, in meters, with shape `[n]`.
            height_m: Antenna height above the ground of each pair, in meters, more than 0.
            los: Line of sight of each pair, 0 or 1.
            clutter: Building fraction of the grid cell of each pair, from 0 to 1.
            path_loss_db: Measured path loss of each pair, in dB.

        Returns:
            The fitted model.

        Raises:
            ValueError: If there is no pair, or if a value is not finite.
        """
        design = cls._design(distance_m, height_m, los, clutter).reshape(-1, 6)
        target = np.asarray(path_loss_db, dtype=float).ravel()
        if target.size == 0 or not (np.isfinite(design).all() and np.isfinite(target).all()):
            raise ValueError("The fit needs a minimum of one pair and only finite values.")
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return cls(coefficients)

    def path_loss_db(
        self, distance_m: np.ndarray, height_m: np.ndarray, los: np.ndarray, clutter: np.ndarray
    ) -> np.ndarray:
        """Calculate the path loss in dB, with the shape of the inputs."""
        return self._design(distance_m, height_m, los, clutter) @ self.coefficients


def spm_inputs(
    inputs: MapInputs, heights_m: Sequence[float], positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the SPM inputs of each (position, cell) pair from the input maps.

    The line of sight and the building fraction are those of the grid cell that holds the
    position.

    Args:
        inputs: The input maps of the cells.
        heights_m: Antenna height above the ground of each cell, in meters.
        positions: Positions in meters, with shape `[n, 2]` or `[n, 3]`.

    Returns:
        The distance, the antenna height, the line of sight, and the building fraction, each
        with shape `[n, n_cells]`, and a boolean array with shape `[n]` that is True for a
        position in the grid.
    """
    positions = np.asarray(positions, dtype=float)
    row, column, inside = inputs.grid.cell_index(positions)
    offset = positions[:, None, :2] - inputs.cell_xy[None, :, :]
    distance = np.linalg.norm(offset, axis=-1)
    height = np.broadcast_to(np.asarray(heights_m, dtype=float), distance.shape)
    los = inputs.los[:, row, column].T
    clutter = np.broadcast_to(inputs.building_fraction[row, column][:, None], distance.shape)
    return distance, height, los, clutter, inside
