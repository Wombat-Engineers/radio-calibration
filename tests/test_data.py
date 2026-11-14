"""Tests for src.data."""

import numpy as np
import pandas as pd
import pytest

from src.core import Cell
from src.data import (
    SPLIT_NAMES,
    mdt_to_arrays,
    mdt_to_long,
    split_records,
    validate_cells,
    validate_mdt,
)

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


# --- split_records ---

FRACTIONS = [0.6, 0.2, 0.2]


def make_long(num_records: int = 10) -> pd.DataFrame:
    """Make records in the long format. Each record has the cell `a`. Record 0 also has `b`."""
    mdt = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2026-01-05") + pd.to_timedelta(np.arange(num_records), "m"),
            "x": np.arange(num_records, dtype=float),
            "y": 0.0,
            "z": 1.5,
            "rsrp_a": -90.0 - np.arange(num_records),
            "rsrp_b": [-95.0] + [np.nan] * (num_records - 1),
        }
    )
    return mdt_to_long(mdt, CELLS)


def test_split_records_disjoint_and_complete():
    long = make_long()
    parts = split_records(long, FRACTIONS, seed=0)

    assert list(parts) == list(SPLIT_NAMES)
    together = pd.concat(parts.values()).sort_values(["record", "cell"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(together, long)
    record_sets = [set(part["record"]) for part in parts.values()]
    assert sum(len(records) for records in record_sets) == long["record"].nunique()


def test_split_records_counts():
    parts = split_records(make_long(), FRACTIONS, seed=0)

    assert [part["record"].nunique() for part in parts.values()] == [6, 2, 2]


def test_split_records_seed():
    long = make_long()
    first = split_records(long, FRACTIONS, seed=0)

    pd.testing.assert_frame_equal(first["train"], split_records(long, FRACTIONS, seed=0)["train"])
    other_seeds = [split_records(long, FRACTIONS, seed=seed)["train"] for seed in range(1, 6)]
    assert any(set(other["record"]) != set(first["train"]["record"]) for other in other_seeds)


def test_split_records_keeps_record_together():
    parts = split_records(make_long(), FRACTIONS, seed=0)

    assert sorted(len(part[part["record"] == 0]) for part in parts.values()) == [0, 0, 2]


def test_split_records_empty():
    long = make_long()
    parts = split_records(long.iloc[:0], FRACTIONS, seed=0)

    assert all(len(part) == 0 for part in parts.values())
    assert all(list(part.columns) == list(long.columns) for part in parts.values())


@pytest.mark.parametrize("fractions", [[0.5, 0.2, 0.2], [1.2, -0.1, -0.1], [0.5, 0.5]])
def test_split_records_bad_fractions(fractions):
    with pytest.raises(ValueError, match="fractions"):
        split_records(make_long(), fractions, seed=0)


# --- mdt_to_arrays ---


def test_mdt_to_arrays_round_trip():
    mdt = make_mdt()
    positions, rsrp = mdt_to_arrays(mdt_to_long(mdt, CELLS), CELLS)

    np.testing.assert_array_equal(positions, mdt[["x", "y", "z"]].to_numpy())
    np.testing.assert_array_equal(rsrp, mdt[["rsrp_a", "rsrp_b"]].to_numpy())


def test_mdt_to_arrays_missing_cell():
    _, rsrp = mdt_to_arrays(mdt_to_long(make_mdt(), CELLS), CELLS)

    assert np.isnan(rsrp[1, 0])
    assert np.isnan(rsrp[2, 1])


def test_mdt_to_arrays_part_of_records():
    long = mdt_to_long(make_mdt(), CELLS)
    positions, rsrp = mdt_to_arrays(long[long["record"] != 1], CELLS)

    assert positions[:, 0].tolist() == [10.0, 30.0]
    assert rsrp[1, 0] == -100.0


def test_mdt_to_arrays_empty():
    positions, rsrp = mdt_to_arrays(mdt_to_long(make_mdt(), CELLS).iloc[:0], CELLS)

    assert positions.shape == (0, 3)
    assert rsrp.shape == (0, len(CELLS))


def test_mdt_to_arrays_unknown_cell():
    with pytest.raises(ValueError, match="b"):
        mdt_to_arrays(mdt_to_long(make_mdt(), CELLS), CELLS[:1])
