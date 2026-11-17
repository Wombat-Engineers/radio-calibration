"""Checks of the input data at the boundary of the package.

Data from another system goes through these checks before the simulation or the
calibration uses it.
"""

from collections.abc import Sequence

import numpy as np

from radio_calibration.core.cell import Cell


def check_cells(cells: Sequence[Cell]) -> None:
    """Make sure that the cells are a valid input of the simulation.

    Args:
        cells: The cells.

    Raises:
        ValueError: If there is no cell, if two cells have the same name, or if a cell
            position is not finite.
    """
    if len(cells) == 0:
        raise ValueError("The cells must have a minimum of one cell.")
    names = [cell.name for cell in cells]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Each cell name must be unique. These names are not: {duplicates}.")
    positions = np.array([[cell.x, cell.y, cell.z] for cell in cells], dtype=float)
    finite = np.isfinite(positions).all(axis=1)
    not_finite = [cell.name for cell, ok in zip(cells, finite, strict=True) if not ok]
    if not_finite:
        raise ValueError(f"These cells have a position that is not finite: {not_finite}.")


def check_positions(positions: np.ndarray) -> None:
    """Make sure that the UE positions have the shape `[n, 3]` and only finite values.

    Args:
        positions: UE positions in meters.

    Raises:
        ValueError: If the shape is not `[n, 3]`, or if a value is not finite.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have the shape [n, 3], but it is {positions.shape}.")
    if not np.isfinite(positions).all():
        raise ValueError("positions must have only finite values.")


def check_measurements(positions: np.ndarray, rsrp_dbm: np.ndarray, num_cells: int) -> None:
    """Make sure that the measurements have the fixed data format.

    Args:
        positions: UE positions in meters, with shape `[n, 3]`.
        rsrp_dbm: Measured RSRP in dBm, with shape `[n, num_cells]`. NaN shows that the
            record does not have the cell.
        num_cells: The number of cells.

    Raises:
        ValueError: If `positions` is not valid (see `check_positions`), if `rsrp_dbm` does
            not have the shape `[n, num_cells]`, or if `rsrp_dbm` has an infinite value.
    """
    check_positions(positions)
    rsrp_dbm = np.asarray(rsrp_dbm, dtype=float)
    expected = (len(positions), num_cells)
    if rsrp_dbm.shape != expected:
        raise ValueError(f"rsrp_dbm must have the shape {expected}, but it is {rsrp_dbm.shape}.")
    if np.isinf(rsrp_dbm).any():
        raise ValueError("rsrp_dbm must have no infinite value. Use NaN for a missing cell.")
