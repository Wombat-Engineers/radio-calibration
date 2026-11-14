"""Gradient-based calibration of a neural material (arXiv:2311.18558, Algorithm 1)."""

from collections.abc import Sequence
from dataclasses import dataclass

import drjit as dr
import mitsuba as mi
import numpy as np
import pandas as pd
from drjit import nn
from omegaconf import DictConfig

from src.calibration.loss import dbm_to_mw, scale_estimate, smape, update_scale
from src.core import Cell
from src.simulation import compute_rsrp, compute_rsrp_mw


@dataclass(frozen=True)
class CalibrationResult:
    """Result of `calibrate`.

    Attributes:
        scale: The linear power scale after the last step. It multiplies the measured
            power.
        history: One row for each step, with the columns `step`, `loss`, `scale`, and
            `validation_loss`. `validation_loss` is NaN at a step with no validation.
    """

    scale: float
    history: pd.DataFrame


def _mean_loss(
    scene,
    cells: Sequence[Cell],
    positions: np.ndarray,
    rsrp_dbm: np.ndarray,
    scale: float,
    simulation: DictConfig,
) -> float:
    """Return the mean loss of the received (record, cell) pairs, without gradients."""
    received = np.isfinite(rsrp_dbm)
    if not received.any():
        return float("nan")
    # No path gives NaN in dBm. It is a power of 0.
    simulated = np.nan_to_num(dbm_to_mw(compute_rsrp(scene, cells, positions, simulation)))
    measured = scale * dbm_to_mw(rsrp_dbm[received])
    return float(np.mean(smape(measured, simulated[received])))


def calibrate(
    scene,
    cells: Sequence[Cell],
    network: nn.Module,
    positions: np.ndarray,
    rsrp_dbm: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray],
    calibration: DictConfig,
    simulation: DictConfig,
    seed: int,
) -> CalibrationResult:
    """Calibrate the neural material of a scene against measured RSRP.

    The function does Algorithm 1 of arXiv:2311.18558 with the AdamW optimizer. The loss
    is the power term of Eq. 38 of the paper: the measurements have no delay spread. The
    function changes the weights of `network`.

    Args:
        scene: The Sionna RT scene. All objects must have a material that uses `network`.
            The scene must have no transmitter and no receiver.
        cells: The cells.
        network: The network of the material, from `make_material_network`.
        positions: Positions of the train records in meters, with shape `[n, 3]`.
        rsrp_dbm: Measured RSRP of the train records in dBm, with shape
            `[n, len(cells)]`. NaN shows that the record does not have the cell.
        validation: Positions and measured RSRP of the validation records, with the same
            format.
        calibration: The configuration from `configs/calibration.yaml`. The function uses
            `learning_rate`, `weight_decay`, `num_steps`, `batch_size`,
            `validation_interval`, and `scale_decay`.
        simulation: The configuration from `configs/simulation.yaml`.
        seed: Seed of the random order of the train records.

    Returns:
        The power scale and the history of the steps.

    Raises:
        ValueError: If there is no train record, or if all simulated powers of a batch
            are 0.
    """
    if len(positions) == 0:
        raise ValueError("The calibration needs a minimum of one train record.")
    rng = np.random.default_rng(seed)
    optimizer = dr.opt.AdamW(lr=calibration.learning_rate, weight_decay=calibration.weight_decay)
    optimizer.update(network)
    measured_mw = dbm_to_mw(rsrp_dbm)
    batch_size = min(calibration.batch_size, len(positions))
    scale = None
    rows = []
    for step in range(calibration.num_steps):
        network.update(optimizer)
        batch = rng.choice(len(positions), size=batch_size, replace=False)
        simulated = compute_rsrp_mw(scene, cells, positions[batch], simulation)
        # Only the received (record, cell) pairs have a measurement.
        received = np.flatnonzero(np.isfinite(measured_mw[batch]))
        simulated = dr.gather(mi.Float, simulated.array, mi.UInt32(received))
        measured = np.ravel(measured_mw[batch])[received]
        # The scale gets no gradient: it has a closed-form solution (Eq. 43).
        estimate = scale_estimate(measured, simulated.numpy())
        scale = update_scale(scale, estimate, calibration.scale_decay)
        loss = dr.mean(smape(mi.Float(scale * measured), simulated))
        dr.backward(loss)
        optimizer.step()
        loss_value = float(np.ravel(loss.numpy())[0])
        # Release the graph of this step before the validation: the GPU memory is small.
        del simulated, loss
        validation_loss = float("nan")
        if (step + 1) % calibration.validation_interval == 0:
            network.update(optimizer)
            validation_loss = _mean_loss(scene, cells, *validation, scale, simulation)
        rows.append(
            {
                "step": step,
                "loss": loss_value,
                "scale": scale,
                "validation_loss": validation_loss,
            }
        )
    network.update(optimizer)
    return CalibrationResult(float(scale), pd.DataFrame(rows))
