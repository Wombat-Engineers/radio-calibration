"""Tests for radio_calibration.core."""

import subprocess
import sys

import numpy as np
import pytest

from radio_calibration.core import (
    CalibrationSettings,
    Cell,
    MapInputs,
    RadioMapGrid,
    SimulationSettings,
    check_cells,
    check_measurements,
    check_positions,
    path_gain_to_db,
    rsrp_from_map,
)


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


# The structure of lab/configs/simulation.yaml and lab/configs/calibration.yaml.
SIMULATION_MAPPING = {
    "radio": {"frequency_hz": 2.6e9},
    "radio_map_cell_size_m": 10.0,
    "radio_map_tx_batch_size": 4,
    "radio_map": {"max_depth": 2, "samples_per_tx": 10_000},
}
CALIBRATION_MAPPING = {
    "network": {"base_channels": 8, "depth": 2, "groups": 4},
    "validation_fraction": 0.2,
    "pretraining": {"learning_rate": 1e-3, "batch_size": 4, "num_epochs": 2},
    "fine_tuning": {"learning_rate": 1e-4, "batch_size": 16, "num_epochs": 3},
}


def test_simulation_settings_from_mapping():
    settings = SimulationSettings.from_mapping(SIMULATION_MAPPING)

    assert settings.frequency_hz == 2.6e9
    assert settings.radio_map["max_depth"] == 2
    assert settings.radio_map_cell_size_m == 10.0
    assert settings.radio_map_tx_batch_size == 4


@pytest.mark.parametrize("key", ["radio", "radio_map", "radio_map_tx_batch_size"])
def test_simulation_settings_missing_key(key):
    mapping = {name: value for name, value in SIMULATION_MAPPING.items() if name != key}

    with pytest.raises(ValueError, match=key):
        SimulationSettings.from_mapping(mapping)


def test_simulation_settings_missing_nested_key():
    with pytest.raises(ValueError, match="radio.frequency_hz"):
        SimulationSettings.from_mapping(SIMULATION_MAPPING | {"radio": {}})


@pytest.mark.parametrize(
    ("key", "value"), [("radio_map_tx_batch_size", 0), ("radio_map_cell_size_m", 0.0)]
)
def test_simulation_settings_bad_value(key, value):
    with pytest.raises(ValueError, match=key):
        SimulationSettings.from_mapping(SIMULATION_MAPPING | {key: value})


def test_calibration_settings_from_mapping():
    settings = CalibrationSettings.from_mapping(CALIBRATION_MAPPING)

    assert (settings.base_channels, settings.depth, settings.groups) == (8, 2, 4)
    assert settings.pretraining.batch_size == 4
    assert settings.fine_tuning.num_epochs == 3


def test_calibration_settings_missing_key():
    mapping = CALIBRATION_MAPPING | {"fine_tuning": {"learning_rate": 1e-4, "batch_size": 16}}

    with pytest.raises(ValueError, match="fine_tuning.num_epochs"):
        CalibrationSettings.from_mapping(mapping)


@pytest.mark.parametrize("name", ["base_channels", "depth", "groups"])
def test_calibration_settings_bad_count(name):
    mapping = CALIBRATION_MAPPING | {"network": CALIBRATION_MAPPING["network"] | {name: 0}}

    with pytest.raises(ValueError, match=name):
        CalibrationSettings.from_mapping(mapping)


def test_calibration_settings_bad_batch():
    mapping = CALIBRATION_MAPPING | {
        "pretraining": {"learning_rate": 1e-3, "batch_size": 0, "num_epochs": 2}
    }

    with pytest.raises(ValueError, match="batch_size"):
        CalibrationSettings.from_mapping(mapping)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_calibration_settings_bad_fraction(fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        CalibrationSettings.from_mapping(CALIBRATION_MAPPING | {"validation_fraction": fraction})


# --- grid and input maps ---

GRID = RadioMapGrid(x_min=0.0, y_min=0.0, z=1.5, cell_size=20.0, nx=2, ny=1)


def make_inputs(path_gain=None) -> MapInputs:
    path_gain = np.array([[[1e-10, 0.0]]]) if path_gain is None else path_gain
    return MapInputs(
        grid=GRID,
        path_gain=path_gain,
        los=np.array([[[True, False]]]),
        max_height=np.array([[0.0, 12.0]]),
        building_fraction=np.array([[0.0, 0.5]]),
        cell_xy=np.array([[5.0, 5.0]]),
    )


def test_grid_centers():
    centers = GRID.centers()

    assert centers.shape == (1, 2, 3)
    assert centers[0, :, 0].tolist() == [10.0, 30.0]
    assert centers[0, 0, 2] == 1.5


def test_grid_cell_index_inside_outside():
    row, column, inside = GRID.cell_index(np.array([[25.0, 5.0], [-1.0, 5.0], [41.0, 5.0]]))

    assert inside.tolist() == [True, False, False]
    assert column.tolist() == [1, 0, 1]
    assert row.tolist() == [0, 0, 0]


def test_grid_diagonal():
    assert GRID.diagonal == pytest.approx(np.hypot(40.0, 20.0))


def test_grid_bad_size():
    with pytest.raises(ValueError, match="cell_size"):
        RadioMapGrid(0.0, 0.0, 1.5, 0.0, 1, 1)


def test_map_inputs_bad_shape():
    with pytest.raises(ValueError, match="path_gain"):
        make_inputs(np.zeros((1, 2, 2)))


def test_map_inputs_negative_path_gain():
    with pytest.raises(ValueError, match="negative"):
        make_inputs(np.array([[[-1.0, 0.0]]]))


def test_map_inputs_subset():
    inputs = make_inputs()
    two = MapInputs(
        GRID,
        np.concatenate([inputs.path_gain, inputs.path_gain * 2]),
        np.concatenate([inputs.los, inputs.los]),
        inputs.max_height,
        inputs.building_fraction,
        np.array([[5.0, 5.0], [6.0, 6.0]]),
    )

    subset = two.subset([1])

    assert subset.num_cells == 1
    assert subset.cell_xy.tolist() == [[6.0, 6.0]]
    assert subset.path_gain[0, 0, 0] == 2e-10


def test_rsrp_from_map():
    path_gain_db = path_gain_to_db(make_inputs().path_gain)
    positions = np.array([[5.0, 5.0], [25.0, 5.0], [-5.0, 5.0]])

    rsrp = rsrp_from_map(path_gain_db, GRID, positions, np.array([4.85]))

    # 4.85 dBm + 10 log10(1e-10) = -95.15 dBm. A path gain of 0 and a position outside the
    # grid give NaN.
    assert rsrp[0, 0] == pytest.approx(-95.15)
    assert np.isnan(rsrp[1:, 0]).all()


def test_rsrp_from_map_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        rsrp_from_map(np.zeros((2, 1, 2)), GRID, np.zeros((1, 2)), np.array([4.85]))


def test_core_imports_no_pandas():
    # A new process: this test process imported pandas already.
    code = "import sys, radio_calibration.core; print('pandas' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


# --- checks ---


def test_check_cells_ok():
    check_cells([make_cell("a"), make_cell("b")])


def test_check_cells_empty():
    with pytest.raises(ValueError, match="minimum"):
        check_cells([])


def test_check_cells_duplicate_name():
    with pytest.raises(ValueError, match="'a'"):
        check_cells([make_cell("a"), make_cell("a", x=5.0)])


def test_check_cells_position_not_finite():
    with pytest.raises(ValueError, match="'b'"):
        check_cells([make_cell("a"), make_cell("b", x=float("nan"))])


def test_check_measurements_ok():
    rsrp = np.array([[-80.0, np.nan], [-90.0, -85.0]])

    check_measurements(np.zeros((2, 3)), rsrp, num_cells=2)


@pytest.mark.parametrize("positions", [np.zeros((2, 2)), np.array([[0.0, np.inf, 1.5]] * 2)])
def test_check_measurements_bad_positions(positions):
    with pytest.raises(ValueError, match="positions"):
        check_measurements(positions, np.zeros((2, 2)), num_cells=2)


def test_check_measurements_bad_rsrp_shape():
    with pytest.raises(ValueError, match=r"\(2, 2\)"):
        check_measurements(np.zeros((2, 3)), np.zeros((2, 3)), num_cells=2)


def test_check_measurements_rsrp_inf():
    with pytest.raises(ValueError, match="infinite"):
        check_measurements(np.zeros((2, 3)), np.array([[-80.0, -np.inf]] * 2), num_cells=2)


def test_check_positions_empty_is_valid():
    check_positions(np.zeros((0, 3)))
