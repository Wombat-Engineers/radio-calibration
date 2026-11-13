"""Tests for src.core."""

import pandas as pd
import pytest

from src.core import MDT_POSITION_COLUMNS, Cell, read_cells, rsrp_column, write_cells


def make_cell(name: str = "site0_sector0", **changes) -> Cell:
    values = {
        "name": name,
        "site": 0,
        "x": 1.5,
        "y": -2.0,
        "z": 25.0,
        "azimuth_deg": 120.0,
        "downtilt_deg": 5.0,
        "power_rs_dbm": 4.85,
        "num_rows": 8,
        "num_cols": 8,
        "vertical_spacing": 0.5,
        "horizontal_spacing": 0.5,
        "pattern": "tr38901",
        "polarization": "cross",
    }
    return Cell(**(values | changes))


def test_cells_roundtrip(tmp_path):
    cells = [make_cell(f"site{i // 3}_sector{i % 3}", site=i // 3, x=10.0 * i) for i in range(9)]
    path = tmp_path / "cells.csv"
    write_cells(cells, path)

    assert read_cells(path) == cells
    assert pd.read_csv(path).columns[0] == "cell"


def test_read_cells_missing_column(tmp_path):
    path = tmp_path / "cells.csv"
    write_cells([make_cell()], path)
    pd.read_csv(path).drop(columns="pattern").to_csv(path, index=False)

    with pytest.raises(ValueError, match="pattern"):
        read_cells(path)


def test_rsrp_column():
    assert rsrp_column("site0_sector0") == "rsrp_site0_sector0"


def test_mdt_position_columns():
    assert MDT_POSITION_COLUMNS == ("timestamp", "x", "y", "z")
