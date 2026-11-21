"""Tests for radio_calibration.api. They use small maps and the CPU."""

import subprocess
import sys

import numpy as np
import pytest
import torch

from radio_calibration import api
from radio_calibration.api import RadioMapModel
from radio_calibration.calibration import (
    HISTORY_COLUMNS,
    Normalization,
    make_network,
)
from tests.test_calibration import CALIBRATION, CPU, FLOOR_DB, GRID, INPUTS, LABELS, records
from tests.test_core import make_cell

CELLS = [
    make_cell("a", x=10.0, y=10.0, z=20.0, num_rows=2, num_cols=2, power_rs_dbm=0.0),
    make_cell("b", x=50.0, y=40.0, z=20.0, num_rows=2, num_cols=2, power_rs_dbm=3.0),
]


@pytest.fixture(scope="module")
def pretrained():
    return RadioMapModel.pretrain(INPUTS, LABELS, CALIBRATION, 0, CPU, FLOOR_DB)


def test_pretrain_returns_model_and_history(pretrained):
    model, history = pretrained

    assert isinstance(model, RadioMapModel)
    assert list(history.columns) == list(HISTORY_COLUMNS)


def test_predict_path_gain_shape(pretrained):
    model, _ = pretrained

    assert model.predict_path_gain_db(INPUTS[0]).shape == (2, GRID.ny, GRID.nx)


def test_predict_rsrp_equals_power_plus_pg(pretrained):
    model, _ = pretrained
    positions = np.array([[15.0, 25.0, 1.5]])
    row, column, _ = GRID.cell_index(positions)

    rsrp = model.predict_rsrp_dbm(INPUTS[0], positions, CELLS)
    path_gain_db = model.predict_path_gain_db(INPUTS[0])[:, row[0], column[0]]

    np.testing.assert_allclose(rsrp[0], [0.0, 3.0] + path_gain_db)


def test_predict_outside_grid_nan(pretrained):
    model, _ = pretrained

    rsrp = model.predict_rsrp_dbm(INPUTS[0], np.array([[-50.0, 0.0, 1.5]]), CELLS)

    assert np.isnan(rsrp).all()


def test_predict_wrong_cell_count(pretrained):
    model, _ = pretrained

    with pytest.raises(ValueError, match="cells"):
        model.predict_rsrp_dbm(INPUTS[0], np.zeros((1, 3)), CELLS[:1])


def test_fine_tune_bad_rsrp_shape(pretrained):
    model, _ = pretrained
    positions, rsrp = records(10, 1)

    with pytest.raises(ValueError, match="shape"):
        model.fine_tune(
            INPUTS[0], positions, rsrp[:, :1], records(5, 2), CELLS, CALIBRATION, seed=0
        )


def test_errors_db_nan_rules(pretrained):
    model, _ = pretrained
    positions = np.array([[15.0, 25.0, 1.5], [-50.0, 0.0, 1.5]])
    measured = np.array([[-80.0, np.nan], [-80.0, -80.0]])

    errors = model.errors_db(INPUTS[0], positions, measured, CELLS)

    assert np.isfinite(errors[0, 0])
    assert np.isnan(errors[0, 1]) and np.isnan(errors[1]).all()


def test_save_load_round_trip(pretrained, tmp_path):
    model, _ = pretrained
    path = tmp_path / "model.pt"

    model.save(path)
    loaded = RadioMapModel.load(path, CPU)

    np.testing.assert_allclose(
        loaded.predict_path_gain_db(INPUTS[1]), model.predict_path_gain_db(INPUTS[1]), rtol=1e-6
    )
    assert loaded.channels == model.channels


def test_save_bad_suffix(pretrained, tmp_path):
    model, _ = pretrained

    with pytest.raises(ValueError, match=".pt"):
        model.save(tmp_path / "model.npz")


def test_integration_flow(tmp_path):
    """Pretrain, fine-tune, save, load, and predict, as another system does."""
    model, _ = RadioMapModel.pretrain(INPUTS, LABELS, CALIBRATION, 0, CPU, FLOOR_DB)
    before = model.predict_path_gain_db(INPUTS[0])
    positions, rsrp = records(40, 1)

    history = model.fine_tune(INPUTS[0], positions, rsrp, records(10, 2), CELLS, CALIBRATION, 0)
    model.save(tmp_path / "model.pt")
    loaded = RadioMapModel.load(tmp_path / "model.pt", CPU)
    predicted = loaded.predict_rsrp_dbm(INPUTS[0], positions, CELLS)

    assert (history["stage"] == "fine_tuning").all()
    assert not np.allclose(loaded.predict_path_gain_db(INPUTS[0]), before)
    assert predicted.shape == rsrp.shape


def test_untrained_model_gives_mean():
    torch.manual_seed(0)
    normalization = Normalization.fit(INPUTS, FLOOR_DB)
    model = RadioMapModel(make_network(6, CALIBRATION), normalization, CPU)

    # A new model gives the mean input path gain everywhere.
    np.testing.assert_allclose(
        model.predict_path_gain_db(INPUTS[0]), normalization.path_gain_mean, rtol=1e-5
    )


def test_api_exports():
    assert {"RadioMapModel", "compute_map_inputs", "map_statistics"} <= set(api.__all__)


def test_api_does_not_import_lab():
    names = ["scenario", "utils", "preprocessing", "experiments", "hydra"]
    code = (
        "import sys, radio_calibration.api; "
        f"print([name for name in {names!r} if name in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
