"""The cell record and the functions that read and write `cells.csv`."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Cell:
    """One sector of a site. Sionna RT simulates each cell as one transmitter.

    Attributes:
        name: Unique name of the cell. The `cell` column of `cells.csv` holds this value.
        site: Index of the site that holds the cell.
        x: Position on the x axis of the scene, in meters.
        y: Position on the y axis of the scene, in meters.
        z: Position on the z axis of the scene, in meters.
        azimuth_deg: Rotation about the z axis, in degrees. 0 points to +x.
        downtilt_deg: Tilt below the horizon, in degrees.
        power_rs_dbm: Reference-signal power for each resource element, in dBm.
        num_rows: Number of rows of the planar array.
        num_cols: Number of columns of the planar array.
        vertical_spacing: Distance between rows, in wavelengths.
        horizontal_spacing: Distance between columns, in wavelengths.
        pattern: Name of the Sionna RT antenna pattern.
        polarization: Name of the Sionna RT polarization.
    """

    name: str
    site: int
    x: float
    y: float
    z: float
    azimuth_deg: float
    downtilt_deg: float
    power_rs_dbm: float
    num_rows: int
    num_cols: int
    vertical_spacing: float
    horizontal_spacing: float
    pattern: str
    polarization: str


# The file uses `cell` because a network export gives the cell identifier this name.
_NAME_COLUMN = "cell"
_COLUMNS = [_NAME_COLUMN if f.name == "name" else f.name for f in fields(Cell)]


def write_cells(cells: Sequence[Cell], path: Path) -> None:
    """Write the cells to a CSV file. Each row is one cell.

    Args:
        cells: The cells to write.
        path: Path of the CSV file. The function replaces a file that exists.
    """
    frame = pd.DataFrame([asdict(cell) for cell in cells]).rename(columns={"name": _NAME_COLUMN})
    frame.to_csv(path, index=False, columns=_COLUMNS)


def read_cells(path: Path) -> list[Cell]:
    """Read the cells from a CSV file that `write_cells` wrote.

    Args:
        path: Path of the CSV file.

    Returns:
        The cells, in the order of the rows.

    Raises:
        ValueError: If a column is missing.
    """
    frame = pd.read_csv(path)
    missing = [column for column in _COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"The file {path.name} does not have these columns: {missing}.")
    return [
        Cell(**{f.name: f.type(row[_COLUMNS[i]]) for i, f in enumerate(fields(Cell))})
        for row in frame.to_dict("records")
    ]
