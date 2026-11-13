"""Tests for src.data."""

import numpy as np
import pandas as pd
import pytest

from src.core import Cell
from src.data import mdt_to_long, validate_cells, validate_mdt

MIN_RSRP = -140.0
RESOLUTION = 1.0


def make_cell(name: str, **changes) -> Cell:
    values = {
        "name": name,
        "site": 0,
        "x": 0.0,
        "y": 0.0,
        "z": 25.0,
        "azimuth_deg": 0.0,
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


CELLS = [make_cell("a"), make_cell("b", azimuth_deg=120.0)]


def make_mdt() -> pd.DataFrame:
    """Make 3 valid records with 4 received cells."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-05 00:00", "2026-01-05 00:15", "2026-01-05 00:30"]
            ),
            "x": [10.0, 20.0, 30.0],
            "y": [-5.0, 0.0, 5.0],
            "z": [1.5, 1.6, 1.4],
            "rsrp_a": [-90.0, np.nan, -100.0],
            "rsrp_b": [-95.0, -110.0, np.nan],
        }
    )


def validate(mdt: pd.DataFrame) -> dict[str, int]:
    return validate_mdt(mdt, CELLS, MIN_RSRP, RESOLUTION)


# --- validate_cells ---


def test_validate_cells_valid():
    assert all(count == 0 for count in validate_cells(CELLS).values())


def test_validate_cells_duplicate_name():
    assert validate_cells([make_cell("a"), make_cell("a")])["Duplicate cell name"] == 1


def test_validate_cells_nan_position():
    cells = [make_cell("a", x=np.nan), make_cell("b")]
    assert validate_cells(cells)["Cell position is not finite"] == 1


# --- validate_mdt ---


def test_validate_mdt_valid():
    assert all(count == 0 for count in validate(make_mdt()).values())


def test_validate_mdt_missing_column():
    with pytest.raises(ValueError, match="rsrp_b"):
        validate(make_mdt().drop(columns="rsrp_b"))


def test_validate_mdt_extra_column():
    with pytest.raises(ValueError, match="foo"):
        validate(make_mdt().assign(foo=0.0))


def test_validate_mdt_timestamp_not_datetime():
    mdt = make_mdt()
    mdt["timestamp"] = mdt["timestamp"].astype(str)
    with pytest.raises(ValueError, match="timestamp"):
        validate(mdt)


def test_validate_mdt_missing_timestamp():
    mdt = make_mdt()
    mdt.loc[1, "timestamp"] = pd.NaT
    assert validate(mdt)["Timestamp is missing"] == 1


def test_validate_mdt_non_finite_position():
    mdt = make_mdt()
    mdt.loc[0, "x"] = np.nan
    mdt.loc[1, "z"] = np.inf
    assert validate(mdt)["Position is not finite"] == 2


def test_validate_mdt_record_without_cell():
    mdt = make_mdt()
    mdt.loc[2, "rsrp_a"] = np.nan
    assert validate(mdt)["Record has no cell"] == 1


def test_validate_mdt_below_threshold():
    mdt = make_mdt()
    mdt.loc[0, "rsrp_b"] = MIN_RSRP - RESOLUTION
    assert validate(mdt)["RSRP is below min_rsrp_dbm"] == 1


def test_validate_mdt_at_threshold():
    mdt = make_mdt()
    mdt.loc[0, "rsrp_b"] = MIN_RSRP
    assert validate(mdt)["RSRP is below min_rsrp_dbm"] == 0


def test_validate_mdt_off_resolution():
    mdt = make_mdt()
    mdt.loc[0, "rsrp_a"] = -100.5
    assert validate(mdt)["RSRP is not a multiple of the resolution"] == 1


def test_validate_mdt_duplicate_record():
    mdt = make_mdt()
    mdt.iloc[2] = mdt.iloc[1]
    assert validate(mdt)["Record is a duplicate"] == 1


# --- mdt_to_long ---


def test_mdt_to_long_rows():
    long = mdt_to_long(make_mdt(), CELLS)

    assert len(long) == 4
    assert list(long.columns) == ["record", "timestamp", "x", "y", "z", "cell", "rsrp_dbm"]


def test_mdt_to_long_values():
    mdt = make_mdt()
    long = mdt_to_long(mdt, CELLS)
    row = long[(long["record"] == 0) & (long["cell"] == "b")].iloc[0]

    assert row["rsrp_dbm"] == mdt.loc[0, "rsrp_b"]
    assert row["x"] == mdt.loc[0, "x"]
    assert row["timestamp"] == mdt.loc[0, "timestamp"]


def test_mdt_to_long_order():
    long = mdt_to_long(make_mdt(), CELLS)

    assert long["record"].tolist() == [0, 0, 1, 2]
    assert long["cell"].tolist() == ["a", "b", "b", "a"]


def test_mdt_to_long_all_missing_cell():
    mdt = make_mdt()
    mdt["rsrp_b"] = np.nan
    long = mdt_to_long(mdt, CELLS)

    assert "b" not in set(long["cell"])
    assert len(long) == 2


def test_mdt_to_long_parquet_roundtrip(tmp_path):
    long = mdt_to_long(make_mdt(), CELLS)
    path = tmp_path / "mdt.parquet"
    long.to_parquet(path, index=False)

    pd.testing.assert_frame_equal(pd.read_parquet(path), long)
