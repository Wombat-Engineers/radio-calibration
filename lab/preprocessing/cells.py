"""The format of `cells.csv`."""

from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd

from radio_calibration.core import Cell

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
