"""Error statistics of a calibrated model against the measured RSRP.

The functions use the RSRP that the caller simulates with `src.simulation.compute_rsrp`.
They do no ray tracing.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from src.calibration.loss import log_error, smape
from src.core import Cell

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
    part: pd.DataFrame, cells: Sequence[Cell], simulated_dbm: np.ndarray, scale: float
) -> pd.DataFrame:
    """Calculate the error of each received (record, cell) pair of a part.

    Args:
        part: MDT records in the long format of `src.data.mdt_to_long`.
        cells: The cells. Each `cell` value of `part` must be the name of one of them.
        simulated_dbm: Simulated RSRP in dBm, with shape `[n, len(cells)]`. `n` is the
            number of records of `part`. The rows are in the order of `record`, as
            `src.data.mdt_to_arrays` gives the positions. NaN shows that no path connects
            the cell and the record.
        scale: The linear power scale of the model. It multiplies the measured power.

    Returns:
        One row for each row of `part`, with the columns `PAIR_COLUMNS`. `error_db` is the
        signed error of `src.calibration.log_error`. It is NaN if the pair has no
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
            "error_db": log_error(measured, simulated, scale),
            "distance_m": np.linalg.norm(position - cell_position[column], axis=1),
        },
        columns=list(PAIR_COLUMNS),
    )


def _quantile_name(quantile: float) -> str:
    return f"q{100.0 * quantile:g}_abs_db"


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Return the Pearson correlation, or NaN if a sample has less than two values or no spread."""
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def error_statistics(pairs: pd.DataFrame, quantiles: Sequence[float]) -> pd.Series:
    """Calculate the error statistics of a set of received pairs.

    The statistics in dB use only the pairs with a simulated path. `smape` uses all pairs:
    the symmetric absolute percentage error of the calibration loss (arXiv:2311.18558,
    Eq. 39). The ratio of the scaled measured power to the simulated power is
    `10 ** (error_db / 10)`, thus `smape` needs no scale. A pair with no path has a
    `smape` of 1, as in the calibration.

    Args:
        pairs: A frame from `pair_errors`.
        quantiles: Quantiles of the absolute error, each from 0 to 1.

    Returns:
        A series with these values:

        - `pairs`: number of pairs.
        - `no_path_fraction`: fraction of the pairs with no simulated path.
        - `bias_db`: mean signed error.
        - `mae_db`: mean absolute error. It is the mean ALE of the paper (Eq. 44).
        - `rmse_db`: root mean square error.
        - `std_db`: standard deviation of the signed error.
        - `q<percent>_abs_db`: one quantile of the absolute error for each value of
          `quantiles`.
        - `pearson`: Pearson correlation of the measured and the simulated RSRP in dBm.
        - `smape`: mean symmetric absolute percentage error.

        A statistic in dB is NaN if no pair has a path.

    Raises:
        ValueError: If `pairs` has no row, or if a quantile is not from 0 to 1.
    """
    if len(pairs) == 0:
        raise ValueError("The statistics need a minimum of one pair.")
    if any(not 0.0 <= quantile <= 1.0 for quantile in quantiles):
        raise ValueError(f"Each quantile must be from 0 to 1, but they are {list(quantiles)}.")
    error = pairs["error_db"].to_numpy(dtype=float)
    has_path = np.isfinite(error)
    found = error[has_path]
    absolute = np.abs(found)
    loss = np.ones_like(error)
    loss[has_path] = smape(10.0 ** (found / 10.0), 1.0)
    statistics = {"pairs": float(len(pairs)), "no_path_fraction": float(np.mean(~has_path))}
    if found.size:
        statistics |= {
            "bias_db": float(np.mean(found)),
            "mae_db": float(np.mean(absolute)),
            "rmse_db": float(np.sqrt(np.mean(found**2))),
            "std_db": float(np.std(found)),
        }
        statistics |= {
            _quantile_name(quantile): float(np.quantile(absolute, quantile))
            for quantile in quantiles
        }
    else:
        names = ["bias_db", "mae_db", "rmse_db", "std_db"] + [_quantile_name(q) for q in quantiles]
        statistics |= dict.fromkeys(names, float("nan"))
    statistics["pearson"] = _pearson(
        pairs["rsrp_dbm"].to_numpy(dtype=float)[has_path],
        pairs["simulated_dbm"].to_numpy(dtype=float)[has_path],
    )
    statistics["smape"] = float(np.mean(loss))
    return pd.Series(statistics)


def grouped_statistics(
    pairs: pd.DataFrame, key: pd.Series, quantiles: Sequence[float]
) -> pd.DataFrame:
    """Calculate the error statistics of each group of pairs.

    Args:
        pairs: A frame from `pair_errors`.
        key: The group of each pair, with the index of `pairs`. For example the `cell`
            column, or `pandas.qcut` of the `distance_m` column.
        quantiles: Quantiles of the absolute error, as for `error_statistics`.

    Returns:
        One row of `error_statistics` for each group that has pairs, in the order of the
        groups. The index has the groups.
    """
    rows = {
        group: error_statistics(members, quantiles)
        for group, members in pairs.groupby(key, observed=True, sort=True)
    }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = key.name
    return frame


def nearest_distance(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Calculate the horizontal distance from each point to the nearest reference point.

    The function uses only the `x` and `y` coordinates. A test record near a train record
    can have a small error only because the two records are near (arXiv:2411.17752).

    Args:
        points: Positions in meters, with shape `[n, 2]` or `[n, 3]`.
        reference: Reference positions in meters, with shape `[m, 2]` or `[m, 3]`.

    Returns:
        The distance in meters, with shape `[n]`.

    Raises:
        ValueError: If `reference` has no row.
    """
    reference = np.asarray(reference, dtype=float)
    if len(reference) == 0:
        raise ValueError("The reference must have a minimum of one position.")
    distance, _ = KDTree(reference[:, :2]).query(np.asarray(points, dtype=float)[:, :2])
    return np.asarray(distance, dtype=float)


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
