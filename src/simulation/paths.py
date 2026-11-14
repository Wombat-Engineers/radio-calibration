"""RSRP calculation with the Sionna RT path solver."""

import math
from collections.abc import Sequence

import drjit as dr
import mitsuba as mi
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


def _check_positions(positions: np.ndarray) -> None:
    """Raise a ValueError if `positions` does not have the shape `[n, 3]`."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have the shape [n, 3], but it is {positions.shape}.")


def _add_cells(scene: sionna.rt.Scene, cells: Sequence[Cell], simulation: DictConfig) -> None:
    """Set the frequency, the bandwidth, and the arrays, and add one transmitter for each cell.

    Raises:
        ValueError: If the cells do not have the same array configuration.
    """
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


def _remove_cells(scene: sionna.rt.Scene, cells: Sequence[Cell]) -> None:
    for cell in cells:
        scene.remove(cell.name)


def _gain_tensor(
    scene: sionna.rt.Scene, positions: np.ndarray, solver: DictConfig, loop_mode: str
) -> mi.TensorXf:
    """Return the path gain from each transmitter to each position, with shape `[n, num_tx]`.

    `loop_mode` is the loop mode of the Sionna RT `PathSolver`. Only "evaluated" lets
    Dr.Jit calculate gradients through the result.
    """
    names = [f"ue{i}" for i in range(len(positions))]
    for name, position in zip(names, positions, strict=True):
        scene.add(sionna.rt.Receiver(name, position=position.tolist()))
    try:
        path_solver = sionna.rt.PathSolver()
        path_solver.loop_mode = loop_mode
        paths = path_solver(scene, **solver)
        # Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths].
        a_real, a_imag = paths.a
        if a_real.shape[4] == 0:
            return dr.zeros(mi.TensorXf, shape=(a_real.shape[0], a_real.shape[2]))
        # Equal weights 1/sqrt(num_tx_ant) on all antennas. This is the default precoding
        # vector of the Sionna RT RadioMapSolver.
        num_tx_ant = a_real.shape[3]
        power = (dr.square(dr.sum(a_real, axis=3)) + dr.square(dr.sum(a_imag, axis=3))) / num_tx_ant
        # Sum of the path powers, not of the path amplitudes: layer-3 filtering removes
        # fast fading from a reported RSRP.
        gain = dr.sum(power, axis=(1, 3))
        dr.eval(gain)
    finally:
        for name in names:
            scene.remove(name)
    return gain


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
    _check_positions(positions)
    _add_cells(scene, cells, simulation)
    batch = simulation.rx_batch_size
    try:
        gains = [
            _gain_tensor(
                scene, positions[start : start + batch], simulation.solver, "symbolic"
            ).numpy()
            for start in range(0, len(positions), batch)
        ]
    finally:
        _remove_cells(scene, cells)
    gain = np.concatenate(gains) if gains else np.empty((0, len(cells)))
    power_rs_dbm = np.array([cell.power_rs_dbm for cell in cells])
    with np.errstate(divide="ignore"):
        rsrp = power_rs_dbm + 10.0 * np.log10(gain)
    return np.where(gain > 0, rsrp, np.nan)


def compute_rsrp_mw(
    scene: sionna.rt.Scene,
    cells: Sequence[Cell],
    positions: np.ndarray,
    simulation: DictConfig,
) -> mi.TensorXf:
    """Calculate the RSRP of each cell at each position, with gradients.

    Dr.Jit can calculate the gradient of the result for the parameters of the radio
    materials. The function uses one call of the path solver: the caller divides the
    positions into batches. It changes the scene as `compute_rsrp` does.

    Args:
        scene: The scene. It must have no transmitter and no receiver.
        cells: The cells. All cells must have the same array configuration.
        positions: Receiver positions in meters, with shape `[n, 3]` and `n` more than 0.
        simulation: The configuration from `configs/simulation.yaml`. The function
            uses `radio`, `receiver`, and `solver`, as `compute_rsrp` does.

    Returns:
        The RSRP in milliwatts, with shape `[n, len(cells)]`. A value is 0 if no path
        connects the cell and the position.

    Raises:
        ValueError: If `positions` does not have the shape `[n, 3]`, if it has no row, or
            if the cells do not have the same array configuration.
        TypeError: If `solver` has a key that is not an argument of `PathSolver`.
    """
    _check_positions(positions)
    if len(positions) == 0:
        raise ValueError("positions must have a minimum of one row.")
    _add_cells(scene, cells, simulation)
    try:
        gain = _gain_tensor(scene, positions, simulation.solver, "evaluated")
    finally:
        _remove_cells(scene, cells)
    power_rs_mw = np.array([10.0 ** (cell.power_rs_dbm / 10.0) for cell in cells], dtype="f4")
    return gain * mi.TensorXf(power_rs_mw[np.newaxis, :])
