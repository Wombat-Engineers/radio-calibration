"""Pretraining, fine-tuning, and files of the models of one scenario."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from experiments.data import ScenarioData, floor_db
from radio_calibration.api import RadioMapModel
from radio_calibration.calibration import FEATURE_NAMES, Normalization, make_network
from utils import calibration_settings


def channels_without(drop: Sequence[str]) -> tuple[str, ...]:
    """Return the input channels of `FEATURE_NAMES` without the channels of `drop`.

    Raises:
        ValueError: If a name of `drop` is not in `FEATURE_NAMES`.
    """
    unknown = sorted(set(drop) - set(FEATURE_NAMES))
    if unknown:
        raise ValueError(f"The channels {unknown} are not in {FEATURE_NAMES}.")
    return tuple(name for name in FEATURE_NAMES if name not in drop)


def model_path(data: ScenarioData, name: str) -> Path:
    """Return the path of the model file `lab/outputs/<scenario>/models/<name>.pt`."""
    return data.dirs.outputs / "models" / f"{name}.pt"


def history_path(data: ScenarioData, name: str) -> Path:
    """Return the path of the history file `lab/outputs/<scenario>/histories/<name>.csv`."""
    return data.dirs.outputs / "histories" / f"{name}.csv"


def pretrain_model(
    cfg: DictConfig,
    data: ScenarioData,
    device: torch.device,
    channels: Sequence[str] = FEATURE_NAMES,
) -> tuple[RadioMapModel, pd.DataFrame]:
    """Pretrain a new model on the layouts of `maps.npz`."""
    return RadioMapModel.pretrain(
        data.maps.variants,
        data.maps.variant_labels,
        calibration_settings(cfg),
        cfg.seed,
        device,
        floor_db(cfg, data),
        channels,
    )


def new_model(
    cfg: DictConfig,
    data: ScenarioData,
    device: torch.device,
    channels: Sequence[str] = FEATURE_NAMES,
) -> RadioMapModel:
    """Make a model with new parameters and no pretraining.

    The normalization comes from the pretraining inputs, as for a pretrained model, thus the
    two models get the same input values.
    """
    calibration = calibration_settings(cfg)
    torch.manual_seed(cfg.seed)
    network = make_network(len(channels), calibration)
    normalization = Normalization.fit(data.maps.variants, floor_db(cfg, data))
    return RadioMapModel(network, normalization, device, channels)


def fine_tune_model(
    cfg: DictConfig,
    data: ScenarioData,
    model: RadioMapModel,
    train: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Fine-tune all parameters of the model on the train records of the scenario.

    Args:
        cfg: The configuration.
        data: The data of the scenario.
        model: The model. The function changes it in place.
        train: Positions and RSRP to use in place of the full train part.

    Returns:
        The history of the fine-tuning.
    """
    positions, rsrp = data.arrays["train"] if train is None else train
    return model.fine_tune(
        data.maps.target,
        positions,
        rsrp,
        data.arrays["validation"],
        data.cells,
        calibration_settings(cfg),
        cfg.seed,
    )


def save_model(data: ScenarioData, name: str, model: RadioMapModel, history: pd.DataFrame) -> None:
    """Write the model and its history to the outputs of the scenario."""
    model.save(model_path(data, name))
    path = history_path(data, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index=False)


def load_model(data: ScenarioData, name: str, device: torch.device) -> RadioMapModel:
    """Read a model of `save_model`."""
    return RadioMapModel.load(model_path(data, name), device)
