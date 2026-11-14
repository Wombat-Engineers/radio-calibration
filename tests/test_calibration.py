"""Tests for src.calibration. The `calibrate` and file tests use a small scene and CUDA."""

import mitsuba as mi
import numpy as np
import pytest
from omegaconf import OmegaConf

from src.calibration import (
    absolute_log_error,
    calibrate,
    dbm_to_mw,
    load_calibration,
    save_calibration,
    scale_estimate,
    smape,
    update_scale,
)
from src.simulation import (
    apply_neural_material,
    compute_rsrp_mw,
    load_rt_scene,
    make_material_network,
)
from tests.test_simulation import CELLS, NETWORK, POINTS, SIMULATION, weights_of

CALIBRATION = OmegaConf.create(
    {
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "num_steps": 2,
        "batch_size": 2,
        "validation_interval": 1,
        "scale_decay": 0.5,
    }
)
POSITIONS = np.array([[x, 0.0, 1.5] for x in (-30.0, -10.0, 10.0, 30.0)])
# One record does not have the second cell.
RSRP_DBM = np.array([[-80.0, -90.0], [-85.0, np.nan], [-90.0, -85.0], [-95.0, -80.0]])


# --- loss ---


def test_dbm_to_mw():
    np.testing.assert_allclose(dbm_to_mw(np.array([0.0, 10.0, -30.0])), [1.0, 10.0, 1e-3])
    assert np.isnan(dbm_to_mw(np.array([np.nan]))).all()


def test_smape_equal():
    assert smape(np.array([2.0]), np.array([2.0])).tolist() == [0.0]


def test_smape_no_path():
    assert smape(np.array([2.0]), np.array([0.0])).tolist() == [1.0]


def test_smape_symmetric():
    first, second = np.array([1.0, 4.0]), np.array([3.0, 1.0])

    np.testing.assert_array_equal(smape(first, second), smape(second, first))
    np.testing.assert_allclose(smape(first, second), [0.5, 0.6])


def test_smape_drjit_equals_numpy():
    first, second = np.array([1.0, 4.0], dtype="f4"), np.array([3.0, 1.0], dtype="f4")

    np.testing.assert_allclose(
        smape(mi.Float(first), mi.Float(second)).numpy(), smape(first, second), rtol=1e-6
    )


def test_scale_estimate_exact():
    measured = np.array([1.0, 2.0, 5.0])

    assert scale_estimate(measured, 3.0 * measured) == pytest.approx(3.0)


def test_scale_estimate_strongest_dominates():
    measured_dbm = np.array([-60.0, -100.0, -120.0])
    # The simulation is 8 dB, 3 dB, and 2 dB above the measurement.
    simulated_dbm = measured_dbm + np.array([8.0, 3.0, 2.0])

    scale = scale_estimate(dbm_to_mw(measured_dbm), dbm_to_mw(simulated_dbm))

    assert 10.0 * np.log10(scale) == pytest.approx(8.0, abs=0.01)


@pytest.mark.parametrize(
    ("measured", "simulated"), [([0.0, 0.0], [1.0, 2.0]), ([1.0, 2.0], [0.0, 0.0]), ([], [])]
)
def test_scale_estimate_no_value(measured, simulated):
    with pytest.raises(ValueError, match="scale"):
        scale_estimate(np.array(measured), np.array(simulated))


def test_update_scale_first():
    assert update_scale(None, 2.0, decay=0.9) == 2.0


def test_update_scale_ema():
    assert update_scale(1.0, 2.0, decay=0.75) == pytest.approx(1.25)


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5])
def test_update_scale_bad_decay(decay):
    with pytest.raises(ValueError, match="decay"):
        update_scale(1.0, 2.0, decay)


def test_absolute_log_error_scale():
    measured = np.array([-100.0, -90.0])

    # A scale of 10 is 10 dB. The simulation is 10 dB above the measurement.
    np.testing.assert_allclose(absolute_log_error(measured, measured + 10.0, scale=10.0), 0.0)
    np.testing.assert_allclose(absolute_log_error(measured, measured, scale=10.0), 10.0)


def test_absolute_log_error_nan():
    error = absolute_log_error(np.array([np.nan, -90.0]), np.array([-90.0, np.nan]), scale=1.0)

    assert np.isnan(error).all()


# --- calibrate, save_calibration, load_calibration ---


@pytest.fixture
def neural_scene():
    """A new scene with a neural material for each test."""
    scene = load_rt_scene("simple_street_canyon")
    material = apply_neural_material(scene, make_material_network(**NETWORK, seed=1))
    return scene, material


def run_calibrate(neural_scene, rsrp_dbm=RSRP_DBM, **changes):
    scene, material = neural_scene
    calibration = OmegaConf.merge(CALIBRATION, changes)
    return calibrate(
        scene,
        CELLS,
        material.network,
        POSITIONS,
        rsrp_dbm,
        (POSITIONS[:2], rsrp_dbm[:2]),
        calibration,
        SIMULATION,
        seed=0,
    )


def test_calibrate_history_columns(neural_scene):
    result = run_calibrate(neural_scene)

    assert list(result.history.columns) == ["step", "loss", "scale", "validation_loss"]
    assert result.history["step"].tolist() == [0, 1]
    assert np.isfinite(result.history.to_numpy()).all()
    assert result.scale == result.history["scale"].iloc[-1]


def test_calibrate_validation_interval(neural_scene):
    result = run_calibrate(neural_scene, validation_interval=2)

    assert np.isnan(result.history["validation_loss"]).tolist() == [True, False]


def test_calibrate_changes_network(neural_scene):
    _, material = neural_scene
    before = weights_of(material.network)

    run_calibrate(neural_scene)

    assert (weights_of(material.network) != before).any()


def test_calibrate_recovers_scale(neural_scene):
    scene, _ = neural_scene
    factor = 4.0
    simulated_mw = compute_rsrp_mw(scene, CELLS, POSITIONS, SIMULATION).numpy()
    # The scale multiplies the measured power, thus the measurement is the simulation
    # divided by the factor.
    measured_dbm = 10.0 * np.log10(simulated_mw / factor)

    result = run_calibrate(neural_scene, measured_dbm, learning_rate=0.0, batch_size=4)

    assert result.scale == pytest.approx(factor, rel=0.1)
    assert result.history["loss"].iloc[0] < 0.1


def test_calibrate_cleans_scene(neural_scene):
    scene, _ = neural_scene

    run_calibrate(neural_scene)

    assert len(scene.transmitters) == 0
    assert len(scene.receivers) == 0


def test_calibrate_no_record(neural_scene):
    scene, material = neural_scene

    with pytest.raises(ValueError, match="train record"):
        calibrate(
            scene,
            CELLS,
            material.network,
            POSITIONS[:0],
            RSRP_DBM[:0],
            (POSITIONS, RSRP_DBM),
            CALIBRATION,
            SIMULATION,
            seed=0,
        )


def test_save_load_calibration_round_trip(neural_scene, tmp_path):
    _, material = neural_scene
    path = tmp_path / "calibration.npz"

    save_calibration(path, material, scale=2.5)
    loaded, scale = load_calibration(path, "loaded")

    assert scale == 2.5
    for value, expected in zip(loaded.values_at(POINTS), material.values_at(POINTS), strict=True):
        np.testing.assert_array_equal(value.numpy(), expected.numpy())


def test_save_calibration_bad_suffix(neural_scene, tmp_path):
    _, material = neural_scene

    with pytest.raises(ValueError, match="npz"):
        save_calibration(tmp_path / "calibration.bin", material, scale=1.0)


def test_load_calibration_missing_scale(neural_scene, tmp_path):
    _, material = neural_scene
    path = tmp_path / "calibration.npz"
    np.savez(path, **material.state())

    with pytest.raises(ValueError, match="scale"):
        load_calibration(path, "loaded")


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_load_calibration_bad_scale(neural_scene, tmp_path, scale):
    _, material = neural_scene
    path = tmp_path / "calibration.npz"
    np.savez(path, **material.state(), scale=np.array(scale))

    with pytest.raises(ValueError, match="scale"):
        load_calibration(path, "loaded")


def test_load_calibration_rejects_pickle(neural_scene, tmp_path):
    _, material = neural_scene
    path = tmp_path / "calibration.npz"
    np.savez(
        path, **(material.state() | {"weights": np.array([{"a": 1}], dtype=object)}), scale=1.0
    )

    with pytest.raises(ValueError):
        load_calibration(path, "loaded")
