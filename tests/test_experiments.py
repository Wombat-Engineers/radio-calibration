"""Tests for experiments. They use small fake data and the CPU."""

import numpy as np
import pandas as pd
import pytest

from experiments import (
    SIONNA_RT,
    SIONNA_RT_OFFSET,
    SPM_ATOLL,
    ScenarioData,
    baseline_maps,
    cell_table,
    channels_without,
    evaluated_cells,
    first_records,
    floor_db,
    map_table,
    mdt_table,
    measurable_db,
)
from preprocessing import MapsFile
from radio_calibration.calibration import FEATURE_NAMES
from radio_calibration.core import MapInputs, RadioMapGrid
from tests.test_core import make_cell
from utils import load_config, scenario_dirs

GRID = RadioMapGrid(x_min=0.0, y_min=0.0, z=1.5, cell_size=10.0, nx=6, ny=5)


def make_data(true_map: bool = True) -> ScenarioData:
    rng = np.random.default_rng(0)
    cells = [
        make_cell("a", site=0, x=5.0, y=5.0, z=25.0, power_rs_dbm=0.0),
        make_cell("b", site=1, x=55.0, y=45.0, z=25.0, power_rs_dbm=0.0),
    ]
    path_gain = 10 ** rng.uniform(-12, -8, size=(2, GRID.ny, GRID.nx))
    path_gain[0, 0, 0] = 0.0
    target = MapInputs(
        GRID,
        path_gain,
        rng.random((2, GRID.ny, GRID.nx)) < 0.5,
        rng.uniform(0, 20, size=(GRID.ny, GRID.nx)),
        rng.uniform(0, 1, size=(GRID.ny, GRID.nx)),
        np.array([[5.0, 5.0], [55.0, 45.0]]),
    )
    maps = MapsFile(
        target=target,
        true_path_gain=path_gain * 2 if true_map else None,
        variants=[target, target],
        variant_labels=[path_gain, path_gain],
        variant_scenes=np.array([0, 1]),
    )

    def part(num: int, seed: int):
        part_rng = np.random.default_rng(seed)
        positions = np.column_stack(
            [part_rng.uniform(0, 60, num), part_rng.uniform(0, 50, num), np.full(num, 1.5)]
        )
        row, column, _ = GRID.cell_index(positions)
        with np.errstate(divide="ignore"):
            rsrp = 10 * np.log10(2 * path_gain[:, row, column].T)
        return positions, np.where(np.isfinite(rsrp), np.round(rsrp), np.nan)

    arrays = {"train": part(60, 1), "validation": part(10, 2), "test": part(30, 3)}
    return ScenarioData(scenario_dirs("fake"), cells, {}, arrays, maps)


@pytest.fixture(scope="module")
def cfg():
    return load_config("scenario=munich_map_errors")


def test_channels_without():
    assert channels_without(["los"]) == tuple(name for name in FEATURE_NAMES if name != "los")


def test_channels_without_unknown():
    with pytest.raises(ValueError, match="nope"):
        channels_without(["nope"])


def test_first_records_nested():
    arrays = make_data().arrays["train"]

    small = first_records(arrays, 5, seed=0)
    large = first_records(arrays, 20, seed=0)

    np.testing.assert_array_equal(small[0], large[0][:5])


def test_first_records_bad_size():
    with pytest.raises(ValueError, match="size"):
        first_records(make_data().arrays["train"], 0, seed=0)


def test_floor_and_measurable(cfg):
    data = make_data()

    assert measurable_db(cfg, data) == cfg.scenario.measurement.min_rsrp_dbm
    assert floor_db(cfg, data) == measurable_db(cfg, data) - 10.0


def test_baseline_maps_keys_and_floor(cfg):
    data = make_data()

    maps = baseline_maps(cfg, data)

    assert list(maps) == [SIONNA_RT, SIONNA_RT_OFFSET, SPM_ATOLL]
    assert all(value.shape == (2, GRID.ny, GRID.nx) for value in maps.values())
    # The grid cell with no ray gets the floor.
    assert maps[SIONNA_RT][0, 0, 0] == floor_db(cfg, data)
    # The measured path gain is 3 dB above the simulated one, thus the offset is near 3 dB.
    offset = maps[SIONNA_RT_OFFSET] - maps[SIONNA_RT]
    assert offset[0, 1, 1] == pytest.approx(3.0, abs=0.6)


def test_baseline_maps_given_train(cfg):
    data = make_data()
    small = first_records(data.arrays["train"], 10, seed=0)

    assert not np.allclose(
        baseline_maps(cfg, data, small)[SPM_ATOLL], baseline_maps(cfg, data)[SPM_ATOLL]
    )


def test_mdt_table_columns(cfg):
    data = make_data()

    table = mdt_table(cfg, data, baseline_maps(cfg, data))

    assert list(table.index) == [SIONNA_RT, SIONNA_RT_OFFSET, SPM_ATOLL]
    assert {"rmse_db", "mae_db", "bias_db", "smape"} <= set(table.columns)


def test_cell_table_shape(cfg):
    data = make_data()

    table = cell_table(cfg, data, baseline_maps(cfg, data))

    assert table.shape == (3, 2)


def test_map_table_skips_without_truth(cfg):
    data = make_data(true_map=False)

    assert map_table(cfg, data, baseline_maps(cfg, data)) is None


def test_map_table_values(cfg):
    data = make_data()

    table = map_table(cfg, data, baseline_maps(cfg, data))

    assert isinstance(table, pd.DataFrame)
    assert table.loc[SIONNA_RT, "bias_db"] == pytest.approx(10 * np.log10(2), abs=1e-6)


def test_evaluated_cells_site_split():
    data = make_data()
    cfg = load_config("scenario=munich_new_site", "scenario.split.test_sites=[1]")

    assert evaluated_cells(cfg, data).tolist() == [1]
    assert evaluated_cells(load_config(), data).tolist() == [0, 1]
