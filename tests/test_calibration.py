"""Tests for radio_calibration.calibration.

The `calibrate` and file tests use a small scene and CUDA.
"""

from dataclasses import replace

import mitsuba as mi
import numpy as np
import pytest

from radio_calibration.calibration import (
    absolute_log_error,
    calibrate,
    dbm_to_mw,
    load_calibration,
    log_error,
    received_powers,
    save_calibration,
    scale_estimate,
    smape,
    update_scale,
)
from radio_calibration.core import CalibrationSettings
from radio_calibration.simulation import (
    apply_neural_material,
    compute_rsrp_mw,
    load_rt_scene,
    make_material_network,
)
from tests.test_simulation import CELLS, NETWORK, POINTS, SIMULATION, weights_of

CALIBRATION = CalibrationSettings(
    **NETWORK,
    learning_rate=1e-2,
    weight_decay=0.0,
    num_steps=2,
    batch_size=2,
    validation_interval=1,
    scale_decay=0.5,
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


def test_absolute_log_error_nan():
    error = absolute_log_error(np.array([np.nan, -90.0]), np.array([-90.0, np.nan]))

    assert np.isnan(error).all()


def test_log_error_sign():
    # The simulation is 5 dB below the measurement.
    np.testing.assert_allclose(log_error(np.array([-80.0]), np.array([-85.0])), 5.0)


def test_absolute_log_error_is_abs():
    rng = np.random.default_rng(0)
    measured, simulated = rng.uniform(-120.0, -60.0, size=(2, 20))

    np.testing.assert_array_equal(
        absolute_log_error(measured, simulated), np.abs(log_error(measured, simulated))
    )


def test_received_powers_no_path_is_zero():
    measured, simulated = received_powers(np.array([-80.0, -90.0]), np.array([-70.0, np.nan]))

    np.testing.assert_allclose(measured, dbm_to_mw(np.array([-80.0, -90.0])))
    np.testing.assert_allclose(simulated, [dbm_to_mw(-70.0), 0.0])


def test_received_powers_drops_missing_measurement():
    measured_dbm = np.array([[-80.0, np.nan], [np.nan, -90.0]])
    simulated_dbm = np.array([[-81.0, -60.0], [-50.0, -91.0]])

    measured, simulated = received_powers(measured_dbm, simulated_dbm)

    np.testing.assert_allclose(measured, dbm_to_mw(np.array([-80.0, -90.0])))
    np.testing.assert_allclose(simulated, dbm_to_mw(np.array([-81.0, -91.0])))


def test_received_powers_scale_counts_no_path():
    # A pair with no path stays in the sums of Eq. 43 with a simulated power of 0.
    measured, simulated = received_powers(
        10.0 * np.log10([1.0, 1.0]), np.array([10.0 * np.log10(2.0), np.nan])
    )

    assert scale_estimate(measured, simulated) == pytest.approx(1.0)


def test_received_powers_shape():
    with pytest.raises(ValueError, match="shapes"):
        received_powers(np.zeros(2), np.zeros(3))


# --- calibrate, save_calibration, load_calibration ---


@pytest.fixture
def neural_scene():
    """A new scene with a neural material for each test."""
    scene = load_rt_scene("simple_street_canyon")
    material = apply_neural_material(scene, make_material_network(**NETWORK, seed=1))
    return scene, material


def run_calibrate(neural_scene, rsrp_dbm=RSRP_DBM, **changes):
    scene, material = neural_scene
    calibration = replace(CALIBRATION, **changes)
    return calibrate(
        scene,
        CELLS,
        material,
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


def test_calibrate_sets_material_scale(neural_scene):
    _, material = neural_scene

    result = run_calibrate(neural_scene)

    assert material.scale == result.history["scale"].iloc[-1]


def test_calibrate_validation_interval(neural_scene):
    result = run_calibrate(neural_scene, validation_interval=2)

    assert np.isnan(result.history["validation_loss"]).tolist() == [True, False]


def test_calibrate_changes_network(neural_scene):
    _, material = neural_scene
    before = weights_of(material.network)

    run_calibrate(neural_scene)

    assert (weights_of(material.network) != before).any()


def recovers_scale_case(neural_scene, factor: float) -> np.ndarray:
    """Return a measurement that is the simulation divided by `factor`."""
    scene, _ = neural_scene
    simulated_mw = compute_rsrp_mw(scene, CELLS, POSITIONS, SIMULATION).numpy()
    return 10.0 * np.log10(simulated_mw / factor)


def test_calibrate_recovers_scale(neural_scene):
    _, material = neural_scene
    measured_dbm = recovers_scale_case(neural_scene, factor=4.0)

    result = run_calibrate(neural_scene, measured_dbm, learning_rate=0.0, batch_size=4)

    # The scale divides the simulated power. A scale applied two times gives 2 or 16.
    assert material.scale == pytest.approx(4.0, rel=0.1)
    assert result.history["loss"].iloc[0] < 0.1


def test_calibrate_resets_scale(neural_scene):
    _, material = neural_scene
    measured_dbm = recovers_scale_case(neural_scene, factor=4.0)
    material.scale = 100.0

    result = run_calibrate(neural_scene, measured_dbm, learning_rate=0.0, batch_size=4)

    # The first step does not start from the old scale of the material.
    assert result.history["scale"].iloc[0] == pytest.approx(4.0, rel=0.1)


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
            material,
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

    material.scale = 2.5

    save_calibration(path, material)
    loaded = load_calibration(path, "loaded")

    assert loaded.scale == 2.5
    for value, expected in zip(loaded.values_at(POINTS), material.values_at(POINTS), strict=True):
        np.testing.assert_array_equal(value.numpy(), expected.numpy())


def test_save_calibration_bad_suffix(neural_scene, tmp_path):
    _, material = neural_scene
    material.scale = 1.0

    with pytest.raises(ValueError, match="npz"):
        save_calibration(tmp_path / "calibration.bin", material)


def test_save_calibration_no_scale(neural_scene, tmp_path):
    _, material = neural_scene

    with pytest.raises(ValueError, match="scale"):
        save_calibration(tmp_path / "calibration.npz", material)


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
