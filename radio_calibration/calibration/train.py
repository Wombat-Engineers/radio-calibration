"""Gradient-based calibration of a neural material (arXiv:2311.18558, Algorithm 1)."""

from collections.abc import Sequence
from dataclasses import dataclass

import drjit as dr
import mitsuba as mi
import numpy as np
import pandas as pd

from radio_calibration.calibration.loss import (
    dbm_to_mw,
    received_powers,
    scale_estimate,
    smape,
    update_scale,
)
from radio_calibration.core import CalibrationSettings, Cell, SimulationSettings
from radio_calibration.simulation import NeuralRadioMaterial, compute_rsrp_dbm, compute_rsrp_mw


@dataclass(frozen=True)
class CalibrationResult:
    """Result of `calibrate`. The power scale is `NeuralRadioMaterial.scale`.

    Attributes:
        history: One row for each step, with the columns `step`, `loss`, `scale`, and
            `validation_loss`. `validation_loss` is NaN at a step with no validation.
    """

    history: pd.DataFrame


def _mean_loss(
    scene,
    cells: Sequence[Cell],
    positions: np.ndarray,
    rsrp_dbm: np.ndarray,
    scale: float,
    simulation: SimulationSettings,
) -> float:
    """Return the mean loss of the received (record, cell) pairs, without gradients."""
    if not np.isfinite(rsrp_dbm).any():
        return float("nan")
    measured, simulated = received_powers(
        rsrp_dbm, compute_rsrp_dbm(scene, cells, positions, simulation, scale)
    )
    return float(np.mean(smape(measured, simulated)))


def calibrate(
    scene,
    cells: Sequence[Cell],
    material: NeuralRadioMaterial,
    positions: np.ndarray,
    rsrp_dbm: np.ndarray,
    validation: tuple[np.ndarray, np.ndarray],
    calibration: CalibrationSettings,
    simulation: SimulationSettings,
    seed: int,
) -> CalibrationResult:
    """Calibrate the neural material of a scene against measured RSRP.

    The function does Algorithm 1 of arXiv:2311.18558 with the AdamW optimizer. The loss
    is the power term of Eq. 38 of the paper: the measurements have no delay spread. The
    function changes the weights of `material.network` and sets `material.scale`. The
    function starts with no scale, also if `material` has a scale. If the function stops
    with an error, the weights and the scale are those of the last completed step.

    Args:
        scene: The Sionna RT scene. All objects must have `material`. The scene must have
            no transmitter and no receiver.
        cells: The cells.
        material: The neural material of the scene.
        positions: Positions of the train records in meters, with shape `[n, 3]`.
        rsrp_dbm: Measured RSRP of the train records in dBm, with shape
            `[n, len(cells)]`. NaN shows that the record does not have the cell.
        validation: Positions and measured RSRP of the validation records, with the same
            format.
        calibration: The settings of the calibration. The function ignores the network
            structure: `material.network` has it.
        simulation: The settings of the simulation.
        seed: Seed of the random order of the train records.

    Returns:
        The history of the steps.

    Raises:
        ValueError: If there is no train record, or if all simulated powers of a batch
            are 0.
    """
    if len(positions) == 0:
        raise ValueError("The calibration needs a minimum of one train record.")
    network = material.network
    rng = np.random.default_rng(seed)
    optimizer = dr.opt.AdamW(lr=calibration.learning_rate, weight_decay=calibration.weight_decay)
    optimizer.update(network)
    measured_mw = dbm_to_mw(rsrp_dbm)
    batch_size = min(calibration.batch_size, len(positions))
    material.scale = None
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
        scale = update_scale(
            material.scale,
            scale_estimate(measured, simulated.numpy()),
            calibration.scale_decay,
        )
        # The only place where the scale multiplies the measured power: the scale comes from
        # this path solver call, thus `compute_rsrp_mw` cannot get it as an argument. A
        # second call with the scale costs one more ray tracing of the batch.
        # smape(scale * P, P_hat) is equal to smape(P, P_hat / scale).
        loss = dr.mean(smape(mi.Float(scale * measured), simulated))
        dr.backward(loss)
        optimizer.step()
        material.scale = scale
        loss_value = float(np.ravel(loss.numpy())[0])
        # Release the graph of this step before the validation: the GPU memory is small.
        del simulated, loss
        validation_loss = float("nan")
        if (step + 1) % calibration.validation_interval == 0:
            network.update(optimizer)
            validation_loss = _mean_loss(scene, cells, *validation, material.scale, simulation)
        rows.append(
            {
                "step": step,
                "loss": loss_value,
                "scale": material.scale,
                "validation_loss": validation_loss,
            }
        )
    network.update(optimizer)
    return CalibrationResult(pd.DataFrame(rows))
