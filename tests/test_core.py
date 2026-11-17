"""Tests for radio_calibration.core."""

import subprocess
import sys

import numpy as np
import pytest

from radio_calibration.core import (
    CalibrationSettings,
    Cell,
    SimulationSettings,
    check_cells,
    check_measurements,
    check_positions,
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
    "scene": {"name": "simple_street_canyon"},
    "radio": {"frequency_hz": 2.6e9, "subcarrier_spacing_hz": 30e3},
    "receiver": {"num_rows": 1, "num_cols": 1, "pattern": "dipole", "polarization": "V"},
    "solver": {"max_depth": 2, "samples_per_src": 10_000},
    "deterministic": True,
    "rx_batch_size": 4,
}
CALIBRATION_MAPPING = {
    "split": {"fractions": [0.5, 0.01, 0.49]},
    "network": {"encoding_octaves": 2, "hidden_layers": 1, "hidden_units": 32},
    "learning_rate": 1e-2,
    "weight_decay": 0.0,
    "num_steps": 2,
    "batch_size": 2,
    "validation_interval": 1,
    "scale_decay": 0.5,
}


def test_simulation_settings_from_mapping():
    settings = SimulationSettings.from_mapping(SIMULATION_MAPPING)

    assert settings.frequency_hz == 2.6e9
    assert settings.subcarrier_spacing_hz == 30e3
    assert settings.solver["max_depth"] == 2
    assert settings.receiver["pattern"] == "dipole"
    assert settings.deterministic is True
    assert settings.rx_batch_size == 4


def test_simulation_settings_ignores_scene():
    mapping = {key: value for key, value in SIMULATION_MAPPING.items() if key != "scene"}

    assert SimulationSettings.from_mapping(mapping) == SimulationSettings.from_mapping(
        SIMULATION_MAPPING
    )


@pytest.mark.parametrize("key", ["radio", "solver", "rx_batch_size"])
def test_simulation_settings_missing_key(key):
    mapping = {name: value for name, value in SIMULATION_MAPPING.items() if name != key}

    with pytest.raises(ValueError, match=key):
        SimulationSettings.from_mapping(mapping)


def test_simulation_settings_missing_nested_key():
    mapping = SIMULATION_MAPPING | {"radio": {"frequency_hz": 2.6e9}}

    with pytest.raises(ValueError, match="radio.subcarrier_spacing_hz"):
        SimulationSettings.from_mapping(mapping)


def test_simulation_settings_bad_batch_size():
    with pytest.raises(ValueError, match="rx_batch_size"):
        SimulationSettings.from_mapping(SIMULATION_MAPPING | {"rx_batch_size": 0})


def test_calibration_settings_from_mapping():
    settings = CalibrationSettings.from_mapping(CALIBRATION_MAPPING)

    assert settings.hidden_units == 32
    assert settings.encoding_octaves == 2
    assert settings.num_steps == 2
    assert settings.scale_decay == 0.5


def test_calibration_settings_missing_key():
    mapping = {key: value for key, value in CALIBRATION_MAPPING.items() if key != "num_steps"}

    with pytest.raises(ValueError, match="num_steps"):
        CalibrationSettings.from_mapping(mapping)


@pytest.mark.parametrize("name", ["num_steps", "batch_size", "validation_interval"])
def test_calibration_settings_bad_count(name):
    with pytest.raises(ValueError, match=name):
        CalibrationSettings.from_mapping(CALIBRATION_MAPPING | {name: 0})


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
