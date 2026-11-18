"""Tests for radio_calibration.api. They use a small scene, a small network, and CUDA."""

import math
import subprocess
import sys

import numpy as np
import pytest

from radio_calibration import api
from radio_calibration.api import CalibratedModel
from radio_calibration.simulation import (
    NeuralRadioMaterial,
    apply_neural_material,
    compute_rsrp_dbm,
    load_rt_scene,
    make_material_network,
)
from tests.test_calibration import CALIBRATION, POSITIONS, RSRP_DBM
from tests.test_simulation import CELLS, IN_BUILDING, NETWORK, SIMULATION

VALIDATION = (POSITIONS[:2], RSRP_DBM[:2])


@pytest.fixture
def scene():
    """A new scene for each test: a model replaces the materials of the scene."""
    return load_rt_scene("simple_street_canyon")


def fit(scene, positions=POSITIONS, rsrp_dbm=RSRP_DBM):
    return CalibratedModel.fit(
        scene, CELLS, positions, rsrp_dbm, VALIDATION, CALIBRATION, SIMULATION, seed=0
    )


def test_fit_returns_model_and_history(scene):
    model, history = fit(scene)

    assert list(history.columns) == [
        "step",
        "epoch",
        "loss",
        "scale",
        "validation_loss",
        "step_seconds",
    ]
    steps_per_epoch = math.ceil(len(POSITIONS) / CALIBRATION.batch_size)
    assert len(history) == CALIBRATION.num_epochs * steps_per_epoch
    assert model.material.scale > 0
    assert all(obj.radio_material is model.material for obj in scene.objects.values())


def test_fit_no_train_record(scene):
    with pytest.raises(ValueError, match="train record"):
        fit(scene, POSITIONS[:0], RSRP_DBM[:0])


def test_fit_bad_rsrp_shape(scene):
    with pytest.raises(ValueError, match="shape"):
        fit(scene, rsrp_dbm=np.zeros((len(POSITIONS), 3)))


def test_model_needs_scale(scene):
    material = apply_neural_material(scene, make_material_network(**NETWORK, seed=1))

    with pytest.raises(ValueError, match="scale"):
        CalibratedModel(scene, CELLS, material, SIMULATION)


def test_predict_equals_compute_rsrp_dbm(scene):
    model, _ = fit(scene)

    expected = compute_rsrp_dbm(scene, CELLS, POSITIONS, SIMULATION, model.material.scale)

    np.testing.assert_array_equal(model.predict_rsrp_dbm(POSITIONS), expected)


@pytest.mark.parametrize("positions", [np.zeros((4, 2)), np.array([[0.0, np.inf, 1.5]])])
def test_predict_bad_positions(scene, positions):
    model, _ = fit(scene)

    with pytest.raises(ValueError, match="positions"):
        model.predict_rsrp_dbm(positions)


def test_predict_reassigns_material(scene):
    model, _ = fit(scene)
    first = model.predict_rsrp_dbm(POSITIONS)
    # Other weights and another name: Sionna RT refuses two materials with one name.
    state = model.material.state()
    state["weights"] = state["weights"] * np.float16(0.5)
    other = CalibratedModel(
        scene, CELLS, NeuralRadioMaterial.from_state(state, "other"), SIMULATION
    )

    second = other.predict_rsrp_dbm(POSITIONS)

    assert not np.array_equal(second, first, equal_nan=True)
    np.testing.assert_array_equal(model.predict_rsrp_dbm(POSITIONS), first)


def test_errors_db_nan_rules(scene):
    model, _ = fit(scene)
    positions = np.vstack([POSITIONS, [IN_BUILDING]])
    predicted = model.predict_rsrp_dbm(positions)
    measured = predicted + 1.0
    measured[0, 1] = np.nan
    # The UE in the building has no path: the predicted RSRP is NaN.
    measured[-1] = -100.0

    errors = model.errors_db(positions, measured)

    assert np.isnan(errors[0, 1])
    assert np.isnan(errors[-1]).all()
    finite = np.isfinite(measured) & np.isfinite(predicted)
    np.testing.assert_allclose(errors[finite], 1.0)


def test_errors_db_bad_shape(scene):
    model, _ = fit(scene)

    with pytest.raises(ValueError, match="shape"):
        model.errors_db(POSITIONS, RSRP_DBM[:, :1])


def test_save_load_round_trip(scene, tmp_path):
    model, _ = fit(scene)
    path = tmp_path / "calibration.npz"
    model.save(path)

    loaded = CalibratedModel.load(scene, CELLS, path, SIMULATION)

    assert loaded.material.scale == model.material.scale
    np.testing.assert_array_equal(
        loaded.predict_rsrp_dbm(POSITIONS), model.predict_rsrp_dbm(POSITIONS)
    )


def test_save_bad_suffix(scene, tmp_path):
    model, _ = fit(scene)

    with pytest.raises(ValueError, match="npz"):
        model.save(tmp_path / "calibration.txt")


def test_integration_flow(tmp_path):
    # The sequence of another system, with names from radio_calibration.api only.
    cells = [api.Cell(**vars(cell)) for cell in CELLS]
    scene = api.load_rt_scene("simple_street_canyon")
    model, _ = api.CalibratedModel.fit(
        scene, cells, POSITIONS, RSRP_DBM, VALIDATION, CALIBRATION, SIMULATION, seed=0
    )
    model.save(tmp_path / "calibration.npz")

    loaded = api.CalibratedModel.load(
        api.load_rt_scene("simple_street_canyon"), cells, tmp_path / "calibration.npz", SIMULATION
    )
    errors = loaded.errors_db(POSITIONS, RSRP_DBM)
    statistics = api.error_statistics(
        RSRP_DBM, loaded.predict_rsrp_dbm(POSITIONS), quantiles=[0.5, 0.9]
    )

    assert errors.shape == RSRP_DBM.shape
    assert np.isfinite(statistics["mae_db"])


def test_api_does_not_import_lab():
    # A new process: this test process imported other modules already. The list has no
    # matplotlib: sionna.rt imports it.
    names = ["scenario", "utils", "preprocessing", "analysis", "hydra", "torch"]
    code = (
        "import sys, radio_calibration.api; "
        f"print([name for name in {names!r} if name in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
