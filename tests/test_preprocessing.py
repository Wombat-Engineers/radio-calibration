"""Tests for preprocessing."""

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    MDT_POSITION_COLUMNS,
    SPLIT_NAMES,
    MapsFile,
    mdt_to_arrays,
    mdt_to_long,
    read_cells,
    read_maps,
    rsrp_column,
    split_by_site,
    split_mdt,
    split_records,
    split_spatial,
    validate_cells,
    validate_mdt,
    write_cells,
    write_maps,
)
from radio_calibration.core import Cell, MapInputs, RadioMapGrid

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


# --- split_by_site, split_spatial, split_mdt ---

# Cell `a` is on site 0 and cell `b` is on site 1.
SITE_CELLS = [make_cell("a", site=0), make_cell("b", site=1)]


def test_split_by_site_no_leak():
    long = make_long()

    parts = split_by_site(long, SITE_CELLS, test_sites=[1], validation_fraction=0.2, seed=0)

    assert set(parts["test"]["cell"]) == {"b"}
    assert set(parts["train"]["cell"]) | set(parts["validation"]["cell"]) == {"a"}
    assert sum(len(part) for part in parts.values()) == len(long)
    assert set(parts["train"]["record"]).isdisjoint(parts["validation"]["record"])


def test_split_by_site_keeps_all_rows():
    # 11 records and the fraction 0.15: int(0.15 * 11) = 1 validation record, 10 train records.
    long = make_long(num_records=11)

    parts = split_by_site(long, SITE_CELLS, test_sites=[1], validation_fraction=0.15, seed=0)

    assert sum(len(part) for part in parts.values()) == len(long)
    assert parts["validation"]["record"].nunique() == 1
    assert parts["train"]["record"].nunique() == 10


@pytest.mark.parametrize("test_sites", [[5], [0, 1]])
def test_split_by_site_bad_sites(test_sites):
    with pytest.raises(ValueError, match="test sites"):
        split_by_site(make_long(), SITE_CELLS, test_sites, 0.2, seed=0)


def test_split_spatial_blocks_disjoint():
    # x goes from 0 to 9 m, thus blocks of 2 m give 5 blocks.
    parts = split_spatial(make_long(), block_size_m=2.0, fractions=FRACTIONS, seed=0)
    blocks = [set(np.floor(part["x"] / 2.0)) for part in parts.values()]

    assert blocks[0].isdisjoint(blocks[1]) and blocks[0].isdisjoint(blocks[2])
    assert blocks[1].isdisjoint(blocks[2])


def test_split_spatial_fractions():
    parts = split_spatial(make_long(), block_size_m=2.0, fractions=FRACTIONS, seed=0)

    # 5 blocks: int(0.6 * 5) = 3 train blocks, int(0.2 * 5) = 1 validation block, 1 test block.
    assert [part["x"].floordiv(2.0).nunique() for part in parts.values()] == [3, 1, 1]


def test_split_spatial_bad_block():
    with pytest.raises(ValueError, match="block_size_m"):
        split_spatial(make_long(), 0.0, FRACTIONS, seed=0)


def test_split_mdt_methods():
    long = make_long()
    random = split_mdt(long, SITE_CELLS, {"method": "random", "fractions": FRACTIONS}, seed=0)
    site = split_mdt(
        long,
        SITE_CELLS,
        {"method": "site", "test_sites": [1], "validation_fraction": 0.2},
        seed=0,
    )

    pd.testing.assert_frame_equal(random["train"], split_records(long, FRACTIONS, 0)["train"])
    assert set(site["test"]["cell"]) == {"b"}


def test_split_mdt_unknown_method():
    with pytest.raises(ValueError, match="random, site, or spatial"):
        split_mdt(make_long(), SITE_CELLS, {"method": "nope"}, seed=0)


# --- maps.npz ---


def make_maps(true_map: bool = True) -> MapsFile:
    grid = RadioMapGrid(0.0, 0.0, 1.5, 10.0, 3, 2)
    rng = np.random.default_rng(0)

    def inputs(num_cells: int) -> MapInputs:
        return MapInputs(
            grid,
            10 ** rng.uniform(-12, -8, size=(num_cells, 2, 3)),
            rng.random((num_cells, 2, 3)) < 0.5,
            np.ones((2, 3)),
            np.full((2, 3), 0.25),
            rng.uniform(0, 30, size=(num_cells, 2)),
        )

    variants = [inputs(2), inputs(2)]
    return MapsFile(
        target=inputs(2),
        true_path_gain=inputs(2).path_gain if true_map else None,
        variants=variants,
        variant_labels=[variant.path_gain * 2 for variant in variants],
        variant_scenes=np.array([0, 1]),
    )


@pytest.mark.parametrize("true_map", [True, False])
def test_maps_round_trip(tmp_path, true_map):
    maps = make_maps(true_map)

    write_maps(tmp_path / "maps.npz", maps)
    loaded = read_maps(tmp_path / "maps.npz")

    assert loaded.target.grid == maps.target.grid
    np.testing.assert_allclose(loaded.target.path_gain, maps.target.path_gain, rtol=1e-6)
    np.testing.assert_array_equal(loaded.target.los, maps.target.los)
    assert (loaded.true_path_gain is None) == (not true_map)
    assert len(loaded.variants) == 2
    np.testing.assert_allclose(loaded.variant_labels[1], maps.variant_labels[1], rtol=1e-6)
    assert loaded.variant_scenes.tolist() == [0, 1]


def test_read_maps_missing_key(tmp_path):
    np.savez(tmp_path / "maps.npz", grid=np.zeros(6))

    with pytest.raises(ValueError, match="variant_label"):
        read_maps(tmp_path / "maps.npz")


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


# --- cells.csv and the columns of mdt.csv ---


def test_cells_roundtrip(tmp_path):
    cells = [make_cell(f"site{i // 3}_sector{i % 3}", site=i // 3, x=10.0 * i) for i in range(9)]
    path = tmp_path / "cells.csv"
    write_cells(cells, path)

    assert read_cells(path) == cells
    assert pd.read_csv(path).columns[0] == "cell"


def test_read_cells_missing_column(tmp_path):
    path = tmp_path / "cells.csv"
    write_cells([make_cell("site0_sector0")], path)
    pd.read_csv(path).drop(columns="pattern").to_csv(path, index=False)

    with pytest.raises(ValueError, match="pattern"):
        read_cells(path)


def test_rsrp_column():
    assert rsrp_column("site0_sector0") == "rsrp_site0_sector0"


def test_mdt_position_columns():
    assert MDT_POSITION_COLUMNS == ("timestamp", "x", "y", "z")
