"""Transmitters of the cells in a Sionna RT scene."""

import math
from collections.abc import Sequence

import sionna.rt

from radio_calibration.core import Cell

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


def add_cells(scene: sionna.rt.Scene, cells: Sequence[Cell], frequency_hz: float) -> None:
    """Set the frequency and the transmitter array, and add one transmitter for each cell.

    Raises:
        ValueError: If the cells do not have the same array configuration.
    """
    scene.tx_array = _tx_array(cells)
    scene.frequency = frequency_hz
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


def remove_cells(scene: sionna.rt.Scene, cells: Sequence[Cell]) -> None:
    """Remove the transmitters of the cells from the scene."""
    for cell in cells:
        scene.remove(cell.name)
