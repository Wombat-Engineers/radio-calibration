"""Grid of a radio map, and the input maps of the U-Net for one scene."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadioMapGrid:
    """A horizontal grid of square cells at one height.

    Row `i` and column `j` is the cell with its center at
    `(x_min + (j + 0.5) * cell_size, y_min + (i + 0.5) * cell_size, z)`.

    Attributes:
        x_min: Smallest x of the grid, in meters.
        y_min: Smallest y of the grid, in meters.
        z: Height of the measurement plane, in meters.
        cell_size: Length of one side of a cell, in meters.
        nx: Number of columns.
        ny: Number of rows.

    Raises:
        ValueError: If `cell_size` is not more than 0, or if `nx` or `ny` is less than 1.
    """

    x_min: float
    y_min: float
    z: float
    cell_size: float
    nx: int
    ny: int

    def __post_init__(self) -> None:
        if not self.cell_size > 0:
            raise ValueError(f"cell_size must be more than 0, but it is {self.cell_size}.")
        if self.nx < 1 or self.ny < 1:
            raise ValueError(f"nx and ny must be 1 or more, but they are {self.nx}, {self.ny}.")

    @property
    def diagonal(self) -> float:
        """Length of the diagonal of the grid, in meters."""
        return float(np.hypot(self.nx * self.cell_size, self.ny * self.cell_size))

    def centers(self) -> np.ndarray:
        """Return the cell centers in meters, with shape `[ny, nx, 3]`."""
        x = self.x_min + (np.arange(self.nx) + 0.5) * self.cell_size
        y = self.y_min + (np.arange(self.ny) + 0.5) * self.cell_size
        xx, yy = np.meshgrid(x, y)
        return np.stack([xx, yy, np.full_like(xx, self.z)], axis=-1)

    def cell_index(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the cell that holds each position. The function uses only x and y.

        Args:
            positions: Positions in meters, with shape `[n, 2]` or `[n, 3]`.

        Returns:
            The row index, the column index, and a boolean array that is True for a
            position in the grid. Each array has shape `[n]`. The indices of a position
            outside the grid are clipped to the nearest cell.
        """
        positions = np.asarray(positions, dtype=float)
        column = np.floor((positions[:, 0] - self.x_min) / self.cell_size).astype(int)
        row = np.floor((positions[:, 1] - self.y_min) / self.cell_size).astype(int)
        inside = (column >= 0) & (column < self.nx) & (row >= 0) & (row < self.ny)
        return np.clip(row, 0, self.ny - 1), np.clip(column, 0, self.nx - 1), inside


@dataclass(frozen=True)
class MapInputs:
    """The maps that the U-Net needs for the cells of one scene.

    Attributes:
        grid: The grid of all maps.
        path_gain: Simulated linear path gain of each cell, with shape `[n_cells, ny, nx]`.
            0 shows a grid cell that no ray reaches.
        los: True where the antenna of the cell has a line of sight to the grid cell, with
            shape `[n_cells, ny, nx]`.
        max_height: Height of the highest building surface in each grid cell, in meters,
            with shape `[ny, nx]`.
        building_fraction: Fraction of the area of each grid cell with a building, from 0
            to 1, with shape `[ny, nx]`.
        cell_xy: Horizontal position of the antenna of each cell, in meters, with shape
            `[n_cells, 2]`.

    Raises:
        ValueError: If a shape does not agree with the grid or with the number of cells,
            or if a path gain is negative or not finite.
    """

    grid: RadioMapGrid
    path_gain: np.ndarray
    los: np.ndarray
    max_height: np.ndarray
    building_fraction: np.ndarray
    cell_xy: np.ndarray

    def __post_init__(self) -> None:
        plane = (self.grid.ny, self.grid.nx)
        num_cells = len(self.cell_xy)
        expected = {
            "path_gain": (num_cells, *plane),
            "los": (num_cells, *plane),
            "max_height": plane,
            "building_fraction": plane,
            "cell_xy": (num_cells, 2),
        }
        for name, shape in expected.items():
            actual = np.shape(getattr(self, name))
            if actual != shape:
                raise ValueError(f"{name} must have the shape {shape}, but it is {actual}.")
        path_gain = np.asarray(self.path_gain)
        if not np.all(np.isfinite(path_gain)) or np.any(path_gain < 0):
            raise ValueError("path_gain must be finite and not negative.")

    @property
    def num_cells(self) -> int:
        """Number of cells (transmitters)."""
        return len(self.cell_xy)

    def subset(self, indices: np.ndarray) -> "MapInputs":
        """Return the maps of the cells with the given indices, in this order."""
        indices = np.asarray(indices, dtype=int)
        return MapInputs(
            grid=self.grid,
            path_gain=self.path_gain[indices],
            los=self.los[indices],
            max_height=self.max_height,
            building_fraction=self.building_fraction,
            cell_xy=self.cell_xy[indices],
        )


def rsrp_from_map(
    path_gain_db: np.ndarray,
    grid: RadioMapGrid,
    positions: np.ndarray,
    power_rs_dbm: np.ndarray,
) -> np.ndarray:
    """Read the RSRP of each cell at each position from a path gain map.

    The RSRP is the reference-signal power of the cell plus the path gain of the grid cell
    that holds the position.

    Args:
        path_gain_db: Path gain in dB, with shape `[n_cells, ny, nx]`. A value that is not
            finite shows a grid cell with no path gain.
        grid: The grid of the map.
        positions: Positions in meters, with shape `[n, 2]` or `[n, 3]`.
        power_rs_dbm: Reference-signal power of each cell, in dBm, with shape `[n_cells]`.

    Returns:
        The RSRP in dBm, with shape `[n, n_cells]`. NaN where the position is outside the
        grid or the path gain is not finite.

    Raises:
        ValueError: If the shape of `path_gain_db` does not agree with the grid and the
            powers.
    """
    path_gain_db = np.asarray(path_gain_db, dtype=float)
    power = np.asarray(power_rs_dbm, dtype=float)
    expected = (len(power), grid.ny, grid.nx)
    if path_gain_db.shape != expected:
        raise ValueError(
            f"path_gain_db must have the shape {expected}, but it is {path_gain_db.shape}."
        )
    row, column, inside = grid.cell_index(positions)
    rsrp = power + path_gain_db[:, row, column].T
    return np.where(inside[:, None] & np.isfinite(rsrp), rsrp, np.nan)


def path_gain_to_db(path_gain: np.ndarray) -> np.ndarray:
    """Change a linear path gain to dB. A path gain of 0 gives NaN."""
    path_gain = np.asarray(path_gain, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(path_gain > 0, 10.0 * np.log10(path_gain), np.nan)
