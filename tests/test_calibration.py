"""Tests for radio_calibration.calibration. They use small maps and the CPU."""

import numpy as np
import pandas as pd
import pytest
import torch

from radio_calibration.calibration import (
    FEATURE_NAMES,
    HISTORY_COLUMNS,
    Normalization,
    UNet,
    absolute_log_error,
    base_path_gain_db,
    dbm_to_mw,
    feature_tensor,
    fine_tune,
    load_model,
    log_error,
    make_network,
    predict_db,
    predict_maps_db,
    pretrain,
    save_model,
    sensitivity_floor_db,
    smape,
)
from radio_calibration.core import CalibrationSettings, MapInputs, RadioMapGrid
from tests.test_core import CALIBRATION_MAPPING

CPU = torch.device("cpu")
CALIBRATION = CalibrationSettings.from_mapping(CALIBRATION_MAPPING)
# 6 rows and 7 columns: the features need padding to 8 x 8 for depth 2.
GRID = RadioMapGrid(x_min=0.0, y_min=0.0, z=1.5, cell_size=10.0, nx=7, ny=6)
# The path gains of the test maps are from -120 dB to -70 dB.
FLOOR_DB = -130.0


def make_inputs(seed: int, num_cells: int = 2) -> MapInputs:
    rng = np.random.default_rng(seed)
    path_gain = 10.0 ** rng.uniform(-12, -7, size=(num_cells, GRID.ny, GRID.nx))
    path_gain[:, 0, 0] = 0.0
    return MapInputs(
        grid=GRID,
        path_gain=path_gain,
        los=rng.random((num_cells, GRID.ny, GRID.nx)) < 0.5,
        max_height=rng.uniform(0, 30, size=(GRID.ny, GRID.nx)),
        building_fraction=rng.uniform(0, 1, size=(GRID.ny, GRID.nx)),
        cell_xy=rng.uniform(0, 60, size=(num_cells, 2)),
    )


INPUTS = [make_inputs(seed) for seed in range(5)]
# The labels are the inputs plus 3 dB, thus a residual network can learn them.
LABELS = [inputs.path_gain * 2.0 for inputs in INPUTS]


# --- loss ---


def test_dbm_to_mw():
    np.testing.assert_allclose(dbm_to_mw(np.array([0.0, -30.0, np.nan])), [1.0, 1e-3, np.nan])


def test_smape_equal():
    assert smape(np.array([2.0]), np.array([2.0]), 0.0).tolist() == [0.0]


def test_smape_value():
    # |1 - 3| / (1 + 3) = 0.5
    assert smape(np.array([1.0]), np.array([3.0]), 0.0).tolist() == [0.5]


def test_smape_zero_label_epsilon():
    assert smape(np.array([0.0]), np.array([0.0]), 1e-30).tolist() == [0.0]


def test_smape_numpy_equals_torch():
    measured, simulated = np.array([1e-10, 3e-12]), np.array([2e-10, 1e-12])

    expected = smape(measured, simulated, 1e-30)
    actual = smape(torch.tensor(measured), torch.tensor(simulated), 1e-30).numpy()

    np.testing.assert_allclose(actual, expected)


def test_log_error_sign():
    assert log_error(np.array([-80.0]), np.array([-90.0])).tolist() == [10.0]


def test_absolute_log_error_nan():
    error = absolute_log_error(np.array([-80.0, np.nan]), np.array([-90.0, -70.0]))

    assert error[0] == 10.0
    assert np.isnan(error[1])


# --- features ---


def test_normalization_fit():
    normalization = Normalization.fit(INPUTS, FLOOR_DB)

    assert normalization.floor_db == FLOOR_DB
    assert normalization.path_gain_std > 0
    assert normalization.height_std > 0


def test_sensitivity_floor_db():
    # -140 dBm - 4.85 dBm - 10 dB margin.
    assert sensitivity_floor_db(-140.0, [4.85, 0.0]) == pytest.approx(-154.85)


def test_normalization_no_map():
    with pytest.raises(ValueError, match="minimum"):
        Normalization.fit([], FLOOR_DB)


def test_normalization_state_round_trip():
    normalization = Normalization.fit(INPUTS, FLOOR_DB)

    assert Normalization.from_state(normalization.state()) == normalization


def test_normalization_state_missing_key():
    state = Normalization.fit(INPUTS, FLOOR_DB).state()
    del state["floor_db"]

    with pytest.raises(ValueError, match="floor_db"):
        Normalization.from_state(state)


def test_base_path_gain_db():
    inputs = make_inputs(0)
    base = base_path_gain_db(inputs, floor_db=-100.0)
    expected = 10 * np.log10(inputs.path_gain[0, 1, 1])

    # A path gain of 0 and a path gain below the floor become the floor.
    assert base[0, 0, 0] == -100.0
    assert base[0, 1, 1] == pytest.approx(max(expected, -100.0))
    assert base.min() == -100.0


def test_feature_padding_and_channels():
    features = feature_tensor(INPUTS[0], Normalization.fit(INPUTS, FLOOR_DB), multiple=4)

    assert features.shape == (2, len(FEATURE_NAMES), 8, 8)
    assert features.dtype == np.float32
    # The added rows and columns are 0.
    assert (features[:, :, 6:, :] == 0).all() and (features[:, :, :, 7:] == 0).all()


def test_feature_floor_and_mask():
    features = feature_tensor(INPUTS[0], Normalization.fit(INPUTS, FLOOR_DB), multiple=4)
    has_path = features[:, FEATURE_NAMES.index("has_path")]

    assert has_path[0, 0, 0] == 0.0
    assert has_path[0, 1, 1] == 1.0


def test_feature_distance_scaled():
    inputs = make_inputs(0)
    distance = feature_tensor(inputs, Normalization.fit(INPUTS, FLOOR_DB), multiple=1)[
        :, FEATURE_NAMES.index("distance")
    ]
    center = GRID.centers()[2, 3, :2]
    expected = np.linalg.norm(center - inputs.cell_xy[0]) / GRID.diagonal

    assert distance[0, 2, 3] == pytest.approx(expected, rel=1e-6)
    assert (distance <= 1).all()


def test_feature_channel_subset():
    channels = tuple(name for name in FEATURE_NAMES if name != "los")

    features = feature_tensor(INPUTS[0], Normalization.fit(INPUTS, FLOOR_DB), 4, channels)

    assert features.shape[1] == len(FEATURE_NAMES) - 1


def test_feature_unknown_channel():
    with pytest.raises(ValueError, match="channels"):
        feature_tensor(INPUTS[0], Normalization.fit(INPUTS, FLOOR_DB), 4, ["nope"])


# --- U-Net ---


def test_unet_output_shape():
    network = UNet(6, 8, 2, 4)

    assert network(torch.zeros(3, 6, 8, 12)).shape == (3, 8, 12)


def test_unet_bad_size():
    with pytest.raises(ValueError, match="multiples of 4"):
        UNet(6, 8, 2, 4)(torch.zeros(1, 6, 6, 8))


def test_unet_initial_output_zero():
    output = UNet(6, 8, 2, 4)(torch.randn(2, 6, 8, 8))

    assert (output == 0).all()


def test_unet_output_independent_of_batch():
    torch.manual_seed(0)
    network = UNet(6, 8, 2, 4)
    torch.nn.init.normal_(network.output.weight)
    first, second = torch.randn(1, 6, 8, 8), torch.randn(1, 6, 8, 8) * 10

    with torch.no_grad():
        alone = network(first)
        together = network(torch.cat([first, second]))[:1]

    torch.testing.assert_close(alone, together)


def test_unet_bad_groups():
    with pytest.raises(ValueError, match="groups"):
        UNet(6, 12, 2, 8)


def test_unet_parameter_count():
    network = UNet(len(FEATURE_NAMES), 32, 3, 8)

    assert 1.8e6 < sum(p.numel() for p in network.parameters()) < 2.0e6


# --- training ---


def run_pretrain(**changes):
    return pretrain(INPUTS, LABELS, CALIBRATION, seed=0, device=CPU, floor_db=FLOOR_DB, **changes)


def test_predict_db_untrained_equals_base():
    normalization = Normalization.fit(INPUTS, FLOOR_DB)
    network = make_network(len(FEATURE_NAMES), CALIBRATION)
    features = torch.as_tensor(feature_tensor(INPUTS[0], normalization, 4))
    base = torch.randn(2, 8, 8)

    predicted = predict_db(network, features, base, normalization, residual=True)

    torch.testing.assert_close(predicted, base)


def test_predict_db_direct():
    normalization = Normalization.fit(INPUTS, FLOOR_DB)
    network = make_network(len(FEATURE_NAMES), CALIBRATION)
    features = torch.as_tensor(feature_tensor(INPUTS[0], normalization, 4))

    predicted = predict_db(network, features, torch.randn(2, 8, 8), normalization, residual=False)

    # A new network gives 0, thus the direct output is the mean path gain.
    assert torch.allclose(
        predicted, torch.tensor(normalization.path_gain_mean, dtype=torch.float32)
    )


def test_pretrain_history_columns():
    result = run_pretrain()

    assert list(result.history.columns) == list(HISTORY_COLUMNS)
    assert len(result.history) == CALIBRATION.pretraining.num_epochs
    assert (result.history["stage"] == "pretraining").all()
    assert result.history["validation_loss"].notna().all()


def test_pretrain_loss_decreases():
    calibration = CalibrationSettings.from_mapping(
        CALIBRATION_MAPPING
        | {"pretraining": {"learning_rate": 1e-2, "batch_size": 4, "num_epochs": 15}}
    )

    history = pretrain(INPUTS, LABELS, calibration, seed=0, device=CPU, floor_db=FLOOR_DB).history

    assert history["loss"].iloc[-1] < history["loss"].iloc[0]


def test_pretrain_keeps_best_checkpoint():
    result = run_pretrain()
    # The validation part is the first map of the random order of seed 0 (1 of 5 maps).
    index = np.random.default_rng(0).permutation(len(INPUTS))[0]

    predicted = predict_maps_db(result.network, result.normalization, INPUTS[index], CPU)
    label = LABELS[index]
    # The loss is the MAE in dB over the grid cells with a target more than 0.
    covered = label > 0
    loss = np.abs(10 * np.log10(label[covered]) - predicted[covered]).mean()

    # The kept parameters give the smallest validation loss of the history.
    assert loss == pytest.approx(result.history["validation_loss"].min(), rel=1e-3)


def test_pretrain_ignores_zero_labels():
    zero = [np.zeros_like(label) for label in LABELS]

    history = pretrain(INPUTS, zero, CALIBRATION, seed=0, device=CPU, floor_db=FLOOR_DB).history

    # No grid cell has a target, thus the loss is 0.
    assert (history["loss"] == 0).all()
    assert (history["validation_loss"] == 0).all()


def test_pretrain_label_shape():
    with pytest.raises(ValueError, match="shape"):
        pretrain(INPUTS, [label[:1] for label in LABELS], CALIBRATION, 0, CPU, floor_db=FLOOR_DB)


def test_pretrain_needs_two_maps():
    with pytest.raises(ValueError, match="two"):
        pretrain(INPUTS[:1], LABELS[:1], CALIBRATION, seed=0, device=CPU, floor_db=FLOOR_DB)


def records(num: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Records in the grid, with the RSRP 3 dB above the input map plus a power of 0 dBm."""
    rng = np.random.default_rng(seed)
    positions = np.column_stack(
        [rng.uniform(0, 70, num), rng.uniform(0, 60, num), np.full(num, 1.5)]
    )
    row, column, _ = GRID.cell_index(positions)
    with np.errstate(divide="ignore"):
        rsrp = 10 * np.log10(INPUTS[0].path_gain[:, row, column].T * 2.0)
    return positions, np.where(np.isfinite(rsrp), rsrp, np.nan)


def run_fine_tune(network=None, positions_rsrp=None):
    result = run_pretrain()
    network = result.network if network is None else network
    positions, rsrp = records(40, 1) if positions_rsrp is None else positions_rsrp
    history = fine_tune(
        network,
        result.normalization,
        INPUTS[0],
        positions,
        rsrp,
        records(10, 2),
        np.zeros(2),
        CALIBRATION,
        seed=0,
        device=CPU,
    )
    return network, history


def test_fine_tune_history():
    _, history = run_fine_tune()

    assert list(history.columns) == list(HISTORY_COLUMNS)
    assert (history["stage"] == "fine_tuning").all()
    assert len(history) == CALIBRATION.fine_tuning.num_epochs


def test_fine_tune_changes_all_layers():
    network = run_pretrain().network
    before = {name: value.clone() for name, value in network.state_dict().items()}
    run_fine_tune(network)

    changed = {
        name for name, value in network.state_dict().items() if not torch.equal(value, before[name])
    }
    assert "encoder.0.0.weight" in changed
    assert "output.weight" in changed


def test_fine_tune_no_record():
    outside = np.array([[1000.0, 1000.0, 1.5]])

    with pytest.raises(ValueError, match="minimum"):
        run_fine_tune(positions_rsrp=(outside, np.array([[-80.0, -80.0]])))


def test_fine_tune_drops_outside_records():
    positions, rsrp = records(20, 1)
    positions = np.vstack([positions, [[1000.0, 1000.0, 1.5]]])
    rsrp = np.vstack([rsrp, [[-80.0, -80.0]]])

    _, history = run_fine_tune(positions_rsrp=(positions, rsrp))

    assert history["loss"].notna().all()


def test_predict_maps_db_shape():
    result = run_pretrain()

    predicted = predict_maps_db(result.network, result.normalization, INPUTS[0], CPU)

    assert predicted.shape == (2, GRID.ny, GRID.nx)


# --- files ---


def test_save_load_round_trip(tmp_path):
    result = run_pretrain()
    path = tmp_path / "model.pt"
    channels = FEATURE_NAMES[:-1]

    save_model(path, result.network, result.normalization, channels, residual=False)
    network, normalization, loaded_channels, residual = load_model(path, CPU)

    assert normalization == result.normalization
    assert loaded_channels == channels
    assert residual is False
    for name, value in result.network.state_dict().items():
        torch.testing.assert_close(network.state_dict()[name], value)


def test_save_bad_suffix(tmp_path):
    result = run_pretrain()

    with pytest.raises(ValueError, match=".pt"):
        save_model(
            tmp_path / "model.npz", result.network, result.normalization, FEATURE_NAMES, True
        )


def test_load_rejects_missing_key(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {}}, path)

    with pytest.raises(ValueError, match="architecture"):
        load_model(path, CPU)


def test_history_is_frame():
    assert isinstance(run_pretrain().history, pd.DataFrame)
