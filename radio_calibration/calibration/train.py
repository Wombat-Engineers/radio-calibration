"""Pretraining and fine-tuning of the U-Net, and the prediction of corrected maps.

The loss is the mean absolute error (MAE) in dB. The pretraining compares the predicted path
gain with the path gain of simulated scene variants. The fine-tuning compares the predicted
RSRP with the measured RSRP of each received (record, cell) pair. The user chose the MAE in dB
on 2026-11-21: with the SMAPE of the linear values, the pretraining stopped on a plateau for
some seeds, because the SMAPE is almost constant for an error of more than about 20 dB.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from radio_calibration.calibration.features import (
    FEATURE_NAMES,
    Normalization,
    base_path_gain_db,
    feature_tensor,
    pad,
    padded_shape,
)
from radio_calibration.calibration.unet import UNet
from radio_calibration.core import CalibrationSettings, MapInputs, TrainingSettings

HISTORY_COLUMNS = ("stage", "epoch", "loss", "validation_loss", "step_seconds")


@dataclass(frozen=True)
class PretrainResult:
    """Result of the pretraining.

    Attributes:
        network: The network with the parameters of the smallest validation loss.
        normalization: The statistics of the pretraining inputs.
        history: One row for each epoch, with the columns `HISTORY_COLUMNS`.
    """

    network: UNet
    normalization: Normalization
    history: pd.DataFrame


def make_network(num_channels: int, calibration: CalibrationSettings) -> UNet:
    """Make a U-Net with the structure of the settings and new parameters."""
    return UNet(num_channels, calibration.base_channels, calibration.depth, calibration.groups)


def predict_db(
    network: UNet,
    features: torch.Tensor,
    base_db: torch.Tensor,
    normalization: Normalization,
    residual: bool,
) -> torch.Tensor:
    """Calculate the predicted path gain in dB.

    Args:
        network: The network.
        features: Input channels with shape `[batch, channels, rows, columns]`.
        base_db: Simulated path gain in dB with the floor, with shape `[batch, rows, columns]`.
        normalization: The statistics of the pretraining inputs.
        residual: If True, the prediction is `base_db` plus the network output. If False,
            the network output is the normalized path gain in dB.

    Returns:
        The path gain in dB, with shape `[batch, rows, columns]`.
    """
    output = network(features)
    if residual:
        return base_db + output
    return normalization.path_gain_mean + normalization.path_gain_std * output


def _tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array, dtype=np.float32), device=device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _batches(count: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    order = torch.randperm(count, generator=generator)
    return list(torch.split(order, batch_size))


def _copy_state(network: UNet) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in network.state_dict().items()}


def _fit(
    network: UNet,
    stage: str,
    training: TrainingSettings,
    num_samples: int,
    train_loss,
    validation_loss,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """Run the epochs of one stage and keep the parameters of the smallest validation loss.

    `train_loss(indices)` gives the loss of a batch. `validation_loss()` gives the
    validation loss, or None if there is no validation data. Then the last parameters stay.
    """
    optimizer = torch.optim.Adam(network.parameters(), lr=training.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state = np.inf, None
    rows = []
    for epoch in range(1, training.num_epochs + 1):
        network.train()
        losses, seconds = [], []
        for indices in _batches(num_samples, training.batch_size, generator):
            _synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad()
            loss = train_loss(indices.to(device))
            loss.backward()
            optimizer.step()
            _synchronize(device)
            seconds.append(time.perf_counter() - start)
            losses.append(float(loss.detach()))
        network.eval()
        with torch.no_grad():
            validation = validation_loss()
        validation = float("nan") if validation is None else float(validation)
        if np.isfinite(validation) and validation < best_loss:
            best_loss, best_state = validation, _copy_state(network)
        rows.append((stage, epoch, float(np.mean(losses)), validation, float(np.mean(seconds))))
    if best_state is not None:
        network.load_state_dict(best_state)
    network.eval()
    return pd.DataFrame(rows, columns=list(HISTORY_COLUMNS))


def pretrain(
    inputs: Sequence[MapInputs],
    labels: Sequence[np.ndarray],
    calibration: CalibrationSettings,
    seed: int,
    device: torch.device,
    floor_db: float,
    channels: Sequence[str] = FEATURE_NAMES,
    residual: bool = False,
) -> PretrainResult:
    """Train a new U-Net to change simulated path gain maps into the target maps.

    The function keeps `calibration.validation_fraction` of the maps for validation. All
    cells of a map are in the same part. The loss is the MAE in dB over the grid cells with a
    target path gain more than 0: a target of 0 shows a grid cell that no ray reaches, not a
    measurement. A target below the floor becomes the floor, because no UE can measure a
    smaller value.

    Args:
        inputs: The input maps of each scene variant.
        labels: The target linear path gain of each variant, with the shape of
            `inputs[i].path_gain`.
        calibration: The settings.
        seed: Seed of the split, of the parameters, and of the batch order.
        device: The PyTorch device.
        floor_db: The floor of the path gain in dB, from `sensitivity_floor_db`.
        channels: The input channels, as for `feature_tensor`.
        residual: The output type, as for `predict_db`.

    Returns:
        The result.

    Raises:
        ValueError: If the number of labels is not the number of inputs, if a label shape
            is not the shape of its input, or if there are less than two maps.
    """
    if len(inputs) != len(labels):
        raise ValueError(f"There are {len(inputs)} inputs, but {len(labels)} labels.")
    if len(inputs) < 2:
        raise ValueError("The pretraining needs a minimum of two maps.")
    for index, (map_inputs, label) in enumerate(zip(inputs, labels, strict=True)):
        if np.shape(label) != map_inputs.path_gain.shape:
            raise ValueError(
                f"Label {index} has the shape {np.shape(label)}, but its input has "
                f"{map_inputs.path_gain.shape}."
            )
    normalization = Normalization.fit(inputs, floor_db)
    multiple = 2**calibration.depth
    parts = {"features": [], "base": [], "label": [], "mask": []}
    for map_inputs, label in zip(inputs, labels, strict=True):
        shape = padded_shape(map_inputs.grid, multiple)
        parts["features"].append(feature_tensor(map_inputs, normalization, multiple, channels))
        parts["base"].append(pad(base_path_gain_db(map_inputs, normalization.floor_db), shape))
        label = np.asarray(label, dtype=float)
        # A grid cell with no target gets the floor here. The mask removes it from the loss.
        label_db = np.maximum(_finite_db(label), floor_db)
        parts["label"].append(pad(label_db, shape))
        parts["mask"].append(pad((label > 0).astype(float), shape))
    num_validation = min(
        max(1, round(calibration.validation_fraction * len(inputs))), len(inputs) - 1
    )
    order = np.random.default_rng(seed).permutation(len(inputs))
    split = {"validation": order[:num_validation], "train": order[num_validation:]}
    data = {
        part: {
            name: _tensor(np.concatenate([values[i] for i in maps]), device)
            for name, values in parts.items()
        }
        for part, maps in split.items()
    }
    torch.manual_seed(seed)
    network = make_network(len(channels), calibration).to(device)

    def loss_of(part: dict[str, torch.Tensor], indices) -> torch.Tensor:
        predicted = predict_db(
            network, part["features"][indices], part["base"][indices], normalization, residual
        )
        error = torch.abs(part["label"][indices] - predicted)
        mask = part["mask"][indices]
        return (error * mask).sum() / mask.sum().clamp(min=1.0)

    validation = data["validation"]
    history = _fit(
        network,
        "pretraining",
        calibration.pretraining,
        len(data["train"]["features"]),
        lambda indices: loss_of(data["train"], indices),
        lambda: loss_of(validation, slice(None)),
        seed,
        device,
    )
    return PretrainResult(network, normalization, history)


def _finite_db(path_gain: np.ndarray) -> np.ndarray:
    """Return the path gain in dB, with -inf where it is 0."""
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(path_gain)


def _pairs(map_inputs: MapInputs, positions: np.ndarray, rsrp_dbm: np.ndarray) -> np.ndarray:
    """Return `[cell, row, column, measured_dbm]` of each received pair in the grid."""
    row, column, inside = map_inputs.grid.cell_index(positions)
    record, cell = np.nonzero(np.isfinite(rsrp_dbm) & inside[:, None])
    return np.column_stack([cell, row[record], column[record], rsrp_dbm[record, cell]])


def fine_tune(
    network: UNet,
    normalization: Normalization,
    map_inputs: MapInputs,
    positions: np.ndarray,
    rsrp_dbm: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray],
    power_rs_dbm: np.ndarray,
    calibration: CalibrationSettings,
    seed: int,
    device: torch.device,
    channels: Sequence[str] = FEATURE_NAMES,
    residual: bool = False,
) -> pd.DataFrame:
    """Change all parameters of the network to agree with the measured RSRP.

    The predicted RSRP of a pair is `power_rs_dbm` of the cell plus the predicted path gain
    of the grid cell that holds the record. The loss is the MAE in dB over the received pairs.
    A record outside the grid has no pair.

    Args:
        network: The network. The function changes it in place.
        normalization: The statistics of the pretraining inputs.
        map_inputs: The input maps of the measured cells.
        positions: Train positions in meters, with shape `[n, 3]`.
        rsrp_dbm: Measured train RSRP in dBm, with shape `[n, n_cells]`. NaN shows that the
            record does not have the cell.
        validation: Validation positions and RSRP, with the same format.
        power_rs_dbm: Reference-signal power of each cell, in dBm, with shape `[n_cells]`.
        calibration: The settings.
        seed: Seed of the batch order.
        device: The PyTorch device.
        channels: The input channels of the network.
        residual: The output type of the network.

    Returns:
        One row for each epoch, with the columns `HISTORY_COLUMNS`.

    Raises:
        ValueError: If no train record has a received pair in the grid.
    """
    train_pairs = _tensor(_pairs(map_inputs, positions, rsrp_dbm), device)
    validation_pairs = _tensor(_pairs(map_inputs, *validation), device)
    if len(train_pairs) == 0:
        raise ValueError("The fine-tuning needs a minimum of one received pair in the grid.")
    multiple = 2**calibration.depth
    shape = padded_shape(map_inputs.grid, multiple)
    features = _tensor(feature_tensor(map_inputs, normalization, multiple, channels), device)
    base = _tensor(pad(base_path_gain_db(map_inputs, normalization.floor_db), shape), device)
    power = _tensor(power_rs_dbm, device)

    def loss_of(pairs: torch.Tensor) -> torch.Tensor:
        predicted = predict_db(network, features, base, normalization, residual)
        cell, row, column = (pairs[:, axis].long() for axis in range(3))
        return torch.abs(pairs[:, 3] - (power[cell] + predicted[cell, row, column])).mean()

    return _fit(
        network,
        "fine_tuning",
        calibration.fine_tuning,
        len(train_pairs),
        lambda indices: loss_of(train_pairs[indices]),
        lambda: loss_of(validation_pairs) if len(validation_pairs) else None,
        seed,
        device,
    )


def predict_maps_db(
    network: UNet,
    normalization: Normalization,
    map_inputs: MapInputs,
    device: torch.device,
    channels: Sequence[str] = FEATURE_NAMES,
    residual: bool = False,
) -> np.ndarray:
    """Calculate the corrected path gain map of each cell, in dB.

    Returns:
        The path gain in dB, with shape `[n_cells, ny, nx]`.
    """
    multiple = 2**network.depth
    shape = padded_shape(map_inputs.grid, multiple)
    features = _tensor(feature_tensor(map_inputs, normalization, multiple, channels), device)
    base = _tensor(pad(base_path_gain_db(map_inputs, normalization.floor_db), shape), device)
    network.eval()
    with torch.no_grad():
        predicted = predict_db(network, features, base, normalization, residual)
    grid = map_inputs.grid
    return predicted[:, : grid.ny, : grid.nx].cpu().numpy().astype(float)
