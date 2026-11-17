"""Error of each (record, cell) pair and the comparison of two models.

The functions use the RSRP that the caller simulates with
`radio_calibration.api.CalibratedModel.predict_rsrp_dbm` or
`radio_calibration.simulation.compute_rsrp_dbm`, with the power scale of the model.
They do no ray tracing.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from radio_calibration.calibration.loss import log_error
from radio_calibration.core import Cell

PAIR_COLUMNS: tuple[str, ...] = (
    "record",
    "cell",
    "x",
    "y",
    "z",
    "rsrp_dbm",
    "simulated_dbm",
    "error_db",
    "distance_m",
)


class BootstrapInterval(NamedTuple):
    """Estimate of a statistic and its bootstrap confidence interval."""

    estimate: float
    lower: float
    upper: float


def pair_errors(
    part: pd.DataFrame, cells: Sequence[Cell], simulated_dbm: np.ndarray
) -> pd.DataFrame:
    """Calculate the error of each received (record, cell) pair of a part.

    Args:
        part: MDT records in the long format of `preprocessing.mdt_to_long`.
        cells: The cells. Each `cell` value of `part` must be the name of one of them.
        simulated_dbm: Simulated RSRP in dBm, with shape `[n, len(cells)]`. `n` is the
            number of records of `part`. The rows are in the order of `record`, as
            `preprocessing.mdt_to_arrays` gives the positions. NaN shows that no
            path connects the cell and the record. The values must have the power scale of
            the model.

    Returns:
        One row for each row of `part`, with the columns `PAIR_COLUMNS`.
        `simulated_dbm` has the power scale of the model. `error_db` is the
        signed error of `radio_calibration.calibration.log_error`. It is NaN if the pair has no
        simulated path. `distance_m` is the distance from the cell to the reported
        position, in meters.

    Raises:
        ValueError: If a `cell` value is not the name of a cell, or if the shape of
            `simulated_dbm` does not agree with the records and the cells.
    """
    names = [cell.name for cell in cells]
    unknown = sorted(set(part["cell"]) - set(names))
    if unknown:
        raise ValueError(f"The cells {unknown} are not in `cells`.")
    records, row = np.unique(part["record"].to_numpy(), return_inverse=True)
    simulated_dbm = np.asarray(simulated_dbm, dtype=float)
    expected = (len(records), len(cells))
    if simulated_dbm.shape != expected:
        raise ValueError(
            f"simulated_dbm must have the shape {expected}, but it is {simulated_dbm.shape}."
        )
    column = part["cell"].map({name: index for index, name in enumerate(names)})
    column = column.to_numpy(dtype=int)
    simulated = simulated_dbm[row, column]
    measured = part["rsrp_dbm"].to_numpy(dtype=float)
    position = part[["x", "y", "z"]].to_numpy(dtype=float)
    cell_position = np.array([[cell.x, cell.y, cell.z] for cell in cells]).reshape(-1, 3)
    return pd.DataFrame(
        {
            "record": part["record"].to_numpy(),
            "cell": part["cell"].to_numpy(),
            "x": position[:, 0],
            "y": position[:, 1],
            "z": position[:, 2],
            "rsrp_dbm": measured,
            "simulated_dbm": simulated,
            "error_db": log_error(measured, simulated),
            "distance_m": np.linalg.norm(position - cell_position[column], axis=1),
        },
        columns=list(PAIR_COLUMNS),
    )


def paired_bootstrap(
    first: pd.DataFrame, second: pd.DataFrame, num_samples: int, confidence: float, seed: int
) -> BootstrapInterval:
    """Calculate the difference of the mean absolute error of two models, with an interval.

    The statistic is the mean absolute error of `second` minus that of `first`. It uses
    the pairs that have a simulated path in the two models. A negative value shows that
    `second` has the smaller error.

    The bootstrap samples the records with replacement, not the pairs: the pairs of one
    record share the error of the reported position. The interval is the percentile
    interval of the bootstrap samples.

    Args:
        first: A frame from `pair_errors` for the first model.
        second: A frame from `pair_errors` for the second model, with the same `record`
            and `cell` rows in the same order.
        num_samples: Number of bootstrap samples.
        confidence: Probability of the interval, more than 0 and less than 1.
        seed: Seed of the bootstrap samples.

    Returns:
        The difference in dB and its interval.

    Raises:
        ValueError: If the two frames do not have the same `record` and `cell` rows, if no
            pair has a path in the two models, if `num_samples` is less than 1, or if
            `confidence` is not more than 0 and less than 1.
    """
    keys = ["record", "cell"]
    if not first[keys].reset_index(drop=True).equals(second[keys].reset_index(drop=True)):
        raise ValueError("The two frames must have the same record and cell rows.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be 1 or more, but it is {num_samples}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be more than 0 and less than 1, but it is {confidence}.")
    first_error = first["error_db"].to_numpy(dtype=float)
    second_error = second["error_db"].to_numpy(dtype=float)
    both = np.isfinite(first_error) & np.isfinite(second_error)
    if not both.any():
        raise ValueError("No pair has a simulated path in the two models.")
    difference = np.abs(second_error[both]) - np.abs(first_error[both])
    _, record = np.unique(first["record"].to_numpy()[both], return_inverse=True)
    sums = np.bincount(record, weights=difference)
    counts = np.bincount(record).astype(float)
    num_records = sums.size
    weights = np.random.default_rng(seed).multinomial(
        num_records, np.full(num_records, 1.0 / num_records), size=num_samples
    )
    samples = (weights @ sums) / (weights @ counts)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return BootstrapInterval(float(sums.sum() / counts.sum()), float(lower), float(upper))
