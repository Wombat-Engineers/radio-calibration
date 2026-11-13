"""RSRP calculation with the Sionna RT path solver."""

import math
from collections.abc import Sequence

import numpy as np
import sionna.rt
from omegaconf import DictConfig

from src.core import Cell

_ARRAY_FIELDS = (
    "num_rows",
    "num_cols",
    "vertical_spacing",
    "horizontal_spacing",
    "pattern",
    "polarization",
)


def _tx_array(cells: Sequence[Cell]) -> sionna.rt.PlanarArray:
    """Make the array that all cells use.

    Raises:
        ValueError: If the cells do not have the same array configuration.
    """
    arrays = {tuple(getattr(cell, name) for name in _ARRAY_FIELDS) for cell in cells}
    # Sionna RT has one `scene.tx_array` for all transmitters.
    if len(arrays) != 1:
        raise ValueError(f"All cells must have the same array, but there are {len(arrays)}.")
    return sionna.rt.PlanarArray(**dict(zip(_ARRAY_FIELDS, arrays.pop(), strict=True)))


def _path_gain(scene: sionna.rt.Scene, positions: np.ndarray, solver: DictConfig) -> np.ndarray:
    """Return the path gain from each transmitter to each position, with shape `[n, num_tx]`."""
    names = [f"ue{i}" for i in range(len(positions))]
    for name, position in zip(names, positions, strict=True):
        scene.add(sionna.rt.Receiver(name, position=position.tolist()))
    try:
        paths = sionna.rt.PathSolver()(scene, **solver)
        # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths].
        a_real, a_imag = (part.numpy() for part in paths.a)
    finally:
        for name in names:
            scene.remove(name)
    # Equal weights 1/sqrt(num_tx_ant) on all antennas. This is the default precoding
    # vector of the Sionna RT RadioMapSolver.
    num_tx_ant = a_real.shape[3]
    power = (a_real.sum(axis=3) ** 2 + a_imag.sum(axis=3) ** 2) / num_tx_ant
    # Sum of the path powers, not of the path amplitudes: layer-3 filtering removes
    # fast fading from a reported RSRP.
    return power.sum(axis=(1, 3))


def compute_rsrp(
    scene: sionna.rt.Scene,
    cells: Sequence[Cell],
    positions: np.ndarray,
    simulation: DictConfig,
) -> np.ndarray:
    """Calculate the RSRP of each cell at each position.

    The function sets the frequency, the bandwidth, and the arrays of the scene. It adds
    transmitters and receivers and removes them before it returns.

    Args:
        scene: The scene. It must have no transmitter and no receiver.
        cells: The cells. All cells must have the same array configuration.
        positions: Receiver positions in meters, with shape `[n, 3]`.
        simulation: The configuration from `configs/simulation.yaml`. The function
            uses `radio` (`frequency_hz`, `subcarrier_spacing_hz`), `receiver` (the
            arguments of the receiver `PlanarArray`), `solver` (the arguments of the
            Sionna RT `PathSolver`), and `rx_batch_size`.

    Returns:
        The RSRP in dBm, with shape `[n, len(cells)]`. A value is NaN if no path
        connects the cell and the position.

    Raises:
        ValueError: If `positions` does not have the shape `[n, 3]`, or if the cells do
            not have the same array configuration.
        TypeError: If `solver` has a key that is not an argument of `PathSolver`.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have the shape [n, 3], but it is {positions.shape}.")
    scene.tx_array = _tx_array(cells)
    scene.rx_array = sionna.rt.PlanarArray(**simulation.receiver)
    scene.frequency = simulation.radio.frequency_hz
    # One resource element. The bandwidth changes only the thermal noise, not the RSRP.
    scene.bandwidth = simulation.radio.subcarrier_spacing_hz
    for cell in cells:
        scene.add(
            sionna.rt.Transmitter(
                cell.name,
                position=[cell.x, cell.y, cell.z],
                # Sionna RT angles are in radians. A positive pitch points the array down.
                orientation=[math.radians(cell.azimuth_deg), math.radians(cell.downtilt_deg), 0.0],
                power_dbm=cell.power_rs_dbm,
            )
        )
    batch = simulation.rx_batch_size
    try:
        gains = [
            _path_gain(scene, positions[start : start + batch], simulation.solver)
            for start in range(0, len(positions), batch)
        ]
    finally:
        for cell in cells:
            scene.remove(cell.name)
    gain = np.concatenate(gains) if gains else np.empty((0, len(cells)))
    power_rs_dbm = np.array([cell.power_rs_dbm for cell in cells])
    with np.errstate(divide="ignore"):
        rsrp = power_rs_dbm + 10.0 * np.log10(gain)
    return np.where(gain > 0, rsrp, np.nan)
