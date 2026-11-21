"""Tests for radio_calibration.evaluation."""

import math
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pynvml
import pytest

import radio_calibration.evaluation.resources as resources
from radio_calibration.calibration import dbm_to_mw, smape
from radio_calibration.core import MapInputs, RadioMapGrid
from radio_calibration.evaluation import (
    ResourceMonitor,
    SpmModel,
    error_statistics,
    fit_offset,
    grouped_statistics,
    map_statistics,
    nearest_distance,
    spm_inputs,
)

QUANTILES = (0.5, 0.9)
NAN = np.nan
# Two records and two cells. The first record does not have the second cell. The second
# cell has no simulated path at the second record. Errors: +2 dB and -4 dB.
MEASURED = np.array([[-80.0, NAN], [-90.0, -70.0]])
SIMULATED = np.array([[-82.0, -60.0], [-86.0, NAN]])


def arrays_with_errors(errors) -> tuple[np.ndarray, np.ndarray]:
    """Make one column of measured RSRP of -80 dBm and the simulated RSRP with these errors."""
    errors = np.asarray(errors, dtype=float)
    measured = np.full((errors.size, 1), -80.0)
    return measured, measured - errors[:, np.newaxis]


# --- error_statistics ---


def test_error_statistics_values():
    statistics = error_statistics(MEASURED, SIMULATED, QUANTILES)

    assert statistics["pairs"] == 3
    assert statistics["no_path_fraction"] == pytest.approx(1 / 3)
    assert statistics["bias_db"] == pytest.approx(-1.0)
    assert statistics["mae_db"] == pytest.approx(3.0)


def test_error_statistics_known():
    statistics = error_statistics(*arrays_with_errors([1.0, -1.0, 3.0, -3.0]), QUANTILES)

    assert statistics["pairs"] == 4
    assert statistics["no_path_fraction"] == 0
    assert statistics["bias_db"] == pytest.approx(0.0)
    assert statistics["mae_db"] == pytest.approx(2.0)
    assert statistics["rmse_db"] == pytest.approx(math.sqrt(5.0))
    assert statistics["std_db"] == pytest.approx(math.sqrt(5.0))


def test_error_statistics_no_path():
    statistics = error_statistics(*arrays_with_errors([0.0, 0.0, 0.0, NAN]), QUANTILES)

    assert statistics["no_path_fraction"] == pytest.approx(0.25)
    assert statistics["mae_db"] == pytest.approx(0.0)
    # Three pairs with a loss of 0 and one pair with no path, which has a loss of 1.
    assert statistics["smape"] == pytest.approx(0.25)


def test_error_statistics_all_no_path():
    statistics = error_statistics(*arrays_with_errors([NAN, NAN]), QUANTILES)

    assert statistics["no_path_fraction"] == 1
    assert np.isnan(statistics[["bias_db", "mae_db", "q50_abs_db", "pearson"]]).all()
    assert statistics["smape"] == 1


def test_error_statistics_ignores_missing_measurement():
    measured, simulated = arrays_with_errors([1.0, 3.0])
    measured[1] = NAN

    statistics = error_statistics(measured, simulated, QUANTILES)

    assert statistics["pairs"] == 1
    assert statistics["mae_db"] == pytest.approx(1.0)


def test_error_statistics_smape_matches_loss():
    rng = np.random.default_rng(0)
    measured = rng.uniform(-120.0, -60.0, size=(50, 1))
    simulated = rng.uniform(-120.0, -60.0, size=(50, 1))

    expected = np.mean(smape(dbm_to_mw(measured), dbm_to_mw(simulated), 0.0))
    assert error_statistics(measured, simulated, QUANTILES)["smape"] == pytest.approx(expected)


def test_error_statistics_quantiles():
    values = np.arange(1.0, 11.0)
    statistics = error_statistics(*arrays_with_errors(values), QUANTILES)

    assert statistics["q50_abs_db"] == pytest.approx(np.quantile(values, 0.5))
    assert statistics["q90_abs_db"] == pytest.approx(np.quantile(values, 0.9))


def test_error_statistics_pearson():
    measured = np.array([[-90.0], [-80.0], [-70.0]])

    assert error_statistics(measured, measured - 5.0, QUANTILES)["pearson"] == pytest.approx(1.0)


def test_error_statistics_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        error_statistics(np.zeros((2, 2)), np.zeros((2, 3)), QUANTILES)


@pytest.mark.parametrize("errors", [[], [NAN, NAN]])
def test_error_statistics_no_pair(errors):
    measured, simulated = arrays_with_errors(errors)
    measured[:] = NAN

    with pytest.raises(ValueError, match="minimum"):
        error_statistics(measured, simulated, QUANTILES)


@pytest.mark.parametrize("quantile", [-0.1, 1.5])
def test_error_statistics_bad_quantile(quantile):
    with pytest.raises(ValueError, match="quantile"):
        error_statistics(*arrays_with_errors([1.0]), [quantile])


# --- grouped_statistics ---


def test_grouped_statistics_by_cell():
    groups = np.broadcast_to(["a", "b"], MEASURED.shape)

    table = grouped_statistics(MEASURED, SIMULATED, groups, QUANTILES)

    assert list(table.index) == ["a", "b"]
    assert table.loc["a", "pairs"] == 2
    assert table.loc["a", "bias_db"] == pytest.approx(-1.0)
    assert table.loc["a", "mae_db"] == pytest.approx(3.0)
    assert table.loc["b", "pairs"] == 1
    assert table.loc["b", "no_path_fraction"] == 1.0


def test_grouped_statistics_matches_error_statistics():
    groups = np.broadcast_to(["a", "b"], MEASURED.shape)

    table = grouped_statistics(MEASURED, SIMULATED, groups, QUANTILES)

    expected = error_statistics(MEASURED[:, :1], SIMULATED[:, :1], QUANTILES)
    pd.testing.assert_series_equal(table.loc["a"], expected, check_names=False)


def test_grouped_statistics_skips_empty_group():
    groups = np.array([["a", "c"], ["a", "a"]])

    table = grouped_statistics(MEASURED, SIMULATED, groups, QUANTILES)

    # Group "c" has only the element with no measurement.
    assert list(table.index) == ["a"]


def test_grouped_statistics_interval_order():
    bands = pd.qcut(np.array([30.0, 5.0]), 2)
    groups = np.broadcast_to(np.asarray(bands)[:, np.newaxis], MEASURED.shape)

    table = grouped_statistics(MEASURED, SIMULATED, groups, QUANTILES)

    assert list(table.index) == sorted(set(bands), key=lambda band: band.left)


def test_grouped_statistics_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        grouped_statistics(MEASURED, SIMULATED, np.array(["a", "b"]), QUANTILES)


# --- nearest_distance ---


# --- map_statistics ---


def test_map_statistics_known():
    truth = np.full((1, 2, 2), 1e-10)
    predicted = np.full((1, 2, 2), -97.0)

    statistics = map_statistics(truth, predicted, QUANTILES)

    # The truth is -100 dB, thus each error is -100 - (-97) = -3 dB.
    assert statistics["grid_cells"] == 4
    assert statistics["bias_db"] == pytest.approx(-3.0)
    assert statistics["mae_db"] == pytest.approx(3.0)
    assert statistics["rmse_db"] == pytest.approx(3.0)


def test_map_statistics_skips_zero_truth():
    truth = np.array([[[1e-10, 0.0]]])
    predicted = np.array([[[-100.0, -50.0]]])

    statistics = map_statistics(truth, predicted, QUANTILES)

    assert statistics["grid_cells"] == 1
    assert statistics["mae_db"] == pytest.approx(0.0)


def test_map_statistics_min_true_db():
    truth = np.array([[[1e-10, 1e-16]]])
    predicted = np.array([[[-100.0, -100.0]]])

    statistics = map_statistics(truth, predicted, QUANTILES, min_true_db=-150.0)

    # The truth of the second grid cell is -160 dB, below -150 dB.
    assert statistics["grid_cells"] == 1
    assert statistics["mae_db"] == pytest.approx(0.0)


def test_map_statistics_no_prediction():
    statistics = map_statistics(np.full((1, 1, 2), 1e-10), np.array([[[-100.0, NAN]]]), QUANTILES)

    assert statistics["no_prediction_fraction"] == pytest.approx(0.5)


def test_map_statistics_shape_mismatch():
    with pytest.raises(ValueError, match="shapes"):
        map_statistics(np.ones((1, 2, 2)), np.ones((1, 2, 3)), QUANTILES)


def test_map_statistics_no_truth():
    with pytest.raises(ValueError, match="minimum"):
        map_statistics(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)), QUANTILES)


# --- baselines ---


def test_fit_offset_known():
    measured = np.array([[-80.0, NAN], [-90.0, -70.0]])

    assert fit_offset(measured, measured - 4.0) == pytest.approx(4.0)


def test_fit_offset_ignores_no_path():
    measured = np.array([[-80.0, -90.0]])
    simulated = np.array([[-84.0, NAN]])

    assert fit_offset(measured, simulated) == pytest.approx(4.0)


def test_fit_offset_no_pair():
    with pytest.raises(ValueError, match="minimum"):
        fit_offset(np.array([[NAN]]), np.array([[-80.0]]))


SPM_COEFFICIENTS = np.array([20.0, 35.0, -5.0, -2.0, -8.0, 6.0])


def spm_data(height_m: np.ndarray):
    rng = np.random.default_rng(0)
    distance = rng.uniform(10.0, 1000.0, size=200)
    los = rng.random(200) < 0.3
    clutter = rng.uniform(0.0, 1.0, size=200)
    path_loss = SpmModel(SPM_COEFFICIENTS).path_loss_db(distance, height_m, los, clutter)
    return distance, height_m, los, clutter, path_loss


def test_spm_recovers_coefficients():
    height = np.random.default_rng(1).uniform(10.0, 40.0, size=200)

    model = SpmModel.fit(*spm_data(height))

    np.testing.assert_allclose(model.coefficients, SPM_COEFFICIENTS, atol=1e-9)


def test_spm_constant_height_no_error():
    distance, height, los, clutter, path_loss = spm_data(np.full(200, 25.0))

    model = SpmModel.fit(distance, height, los, clutter, path_loss)

    np.testing.assert_allclose(model.path_loss_db(distance, height, los, clutter), path_loss)


def test_spm_no_pair():
    with pytest.raises(ValueError, match="minimum"):
        SpmModel.fit(*(np.empty(0) for _ in range(5)))


def test_spm_inputs():
    grid = RadioMapGrid(0.0, 0.0, 1.5, 10.0, 2, 1)
    inputs = MapInputs(
        grid,
        np.ones((1, 1, 2)) * 1e-10,
        np.array([[[True, False]]]),
        np.zeros((1, 2)),
        np.array([[0.0, 0.5]]),
        np.array([[0.0, 0.0]]),
    )

    distance, height, los, clutter, inside = spm_inputs(
        inputs, [25.0], np.array([[3.0, 4.0, 1.5], [15.0, 5.0, 1.5], [-5.0, 0.0, 1.5]])
    )

    assert distance[:2, 0].tolist() == [5.0, pytest.approx(np.hypot(15.0, 5.0))]
    assert height[:, 0].tolist() == [25.0, 25.0, 25.0]
    assert los[:2, 0].tolist() == [True, False]
    assert clutter[:2, 0].tolist() == [0.0, 0.5]
    assert inside.tolist() == [True, True, False]


def test_nearest_distance():
    reference = np.array([[0.0, 0.0], [10.0, 0.0]])
    points = np.array([[1.0, 0.0, 5.0], [9.0, 1.0, 0.0]])

    np.testing.assert_allclose(nearest_distance(points, reference), [1.0, math.sqrt(2.0)])


def test_nearest_distance_empty():
    with pytest.raises(ValueError, match="minimum"):
        nearest_distance(np.zeros((2, 3)), np.empty((0, 3)))


# --- ResourceMonitor ---

MIB = 2**20


class FakeProcess:
    """A psutil.Process with a fixed memory and a fixed CPU use."""

    def __init__(self, rss_mib=100.0, cpu_percent=50.0):
        self.rss_mib = rss_mib
        self.cpu = cpu_percent

    def memory_info(self):
        return SimpleNamespace(rss=self.rss_mib * MIB)

    def cpu_percent(self, interval=None):
        return self.cpu


@pytest.fixture
def fake_nvml(monkeypatch):
    """Replace NVML with a GPU that uses 1234 MiB and 56 %."""
    calls = {"shutdown": 0}

    def shutdown():
        calls["shutdown"] += 1

    monkeypatch.setattr(resources.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(resources.pynvml, "nvmlShutdown", shutdown)
    monkeypatch.setattr(resources.pynvml, "nvmlDeviceGetHandleByIndex", lambda index: "gpu")
    monkeypatch.setattr(
        resources.pynvml,
        "nvmlDeviceGetMemoryInfo",
        lambda handle: SimpleNamespace(used=1234 * MIB),
    )
    monkeypatch.setattr(
        resources.pynvml, "nvmlDeviceGetUtilizationRates", lambda handle: SimpleNamespace(gpu=56)
    )
    return calls


def test_resource_monitor_samples():
    with ResourceMonitor(interval_s=0.05) as monitor:
        time.sleep(0.3)
    samples = monitor.samples

    assert list(samples.columns) == list(resources.SAMPLE_COLUMNS)
    assert len(samples) >= 2
    assert samples["seconds"].is_monotonic_increasing
    assert samples["seconds"].iloc[-1] >= 0.3
    assert (samples["rss_mib"] > 0).all()
    assert math.isnan(samples["cpu_percent"].iloc[0])


def test_resource_monitor_reads_nvml(fake_nvml):
    with ResourceMonitor(interval_s=0.05):
        pass
    with ResourceMonitor(interval_s=0.05) as monitor:
        pass

    assert (monitor.samples["gpu_memory_mib"] == 1234.0).all()
    assert (monitor.samples["gpu_utilization_percent"] == 56.0).all()
    assert fake_nvml["shutdown"] == 2


def test_resource_monitor_no_nvml(monkeypatch):
    def fail():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)

    monkeypatch.setattr(resources.pynvml, "nvmlInit", fail)
    with ResourceMonitor(interval_s=0.05) as monitor:
        pass

    assert monitor.samples["gpu_memory_mib"].isna().all()
    assert monitor.samples["gpu_utilization_percent"].isna().all()
    assert (monitor.samples["rss_mib"] > 0).all()


@pytest.mark.parametrize("interval_s", [0.0, -1.0])
def test_resource_monitor_bad_interval(interval_s):
    with pytest.raises(ValueError, match="interval_s"):
        ResourceMonitor(interval_s=interval_s)


def test_resource_monitor_samples_running():
    with ResourceMonitor(interval_s=0.05) as monitor, pytest.raises(RuntimeError, match="after"):
        monitor.samples  # noqa: B018


def test_resource_monitor_samples_not_run():
    with pytest.raises(RuntimeError, match="after"):
        ResourceMonitor().samples  # noqa: B018


def test_resource_monitor_exception_in_block():
    monitor = ResourceMonitor(interval_s=0.05)
    with pytest.raises(KeyError), monitor:
        raise KeyError("stop")

    assert monitor._thread is None
    assert len(monitor.samples) == 2


def test_resource_monitor_summary(monkeypatch, fake_nvml):
    monkeypatch.setattr(resources.psutil, "Process", lambda: FakeProcess(100.0, 50.0))
    with ResourceMonitor(interval_s=0.05) as monitor:
        time.sleep(0.15)
    summary = monitor.summary()

    assert summary["seconds"] == monitor.samples["seconds"].iloc[-1]
    assert summary["peak_gpu_memory_mib"] == 1234.0
    assert summary["mean_gpu_utilization_percent"] == 56.0
    assert summary["peak_rss_mib"] == 100.0
    # The first sample has no CPU use.
    assert summary["mean_cpu_percent"] == 50.0
