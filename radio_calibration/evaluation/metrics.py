"""Error statistics of a calibrated model against the measured RSRP.

The functions use the arrays of the fixed data format: one row for each record and one
column for each cell. NaN in the measured RSRP shows that the record does not have the
cell. NaN in the simulated RSRP shows that no path connects the cell and the position.
The functions do no ray tracing.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from radio_calibration.calibration.loss import log_error, smape


def _quantile_name(quantile: float) -> str:
    return f"q{100.0 * quantile:g}_abs_db"


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Return the Pearson correlation, or NaN if a sample has less than two values or no spread."""
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def error_statistics(
    measured_dbm: np.ndarray, simulated_dbm: np.ndarray, quantiles: Sequence[float]
) -> pd.Series:
    """Calculate the error statistics of the received (record, cell) pairs.

    A received pair is an element with a finite `measured_dbm`. The statistics in dB use
    only the received pairs with a simulated path. `smape` uses all received pairs: the
    symmetric absolute percentage error of the calibration loss (arXiv:2311.18558,
    Eq. 39). The ratio of the measured power to the scaled simulated power is
    `10 ** (error_db / 10)`, thus `smape` needs no scale. A pair with no path has a
    `smape` of 1, as in the calibration.

    Args:
        measured_dbm: Measured RSRP in dBm. NaN shows that the record does not have the
            cell.
        simulated_dbm: Simulated RSRP in dBm with the power scale of the model, with the
            same shape. NaN shows that no path connects the cell and the position.
        quantiles: Quantiles of the absolute error, each from 0 to 1.

    Returns:
        A series with these values:

        - `pairs`: number of received pairs.
        - `no_path_fraction`: fraction of the received pairs with no simulated path.
        - `bias_db`: mean signed error.
        - `mae_db`: mean absolute error. It is the mean ALE of the paper (Eq. 44).
        - `rmse_db`: root mean square error.
        - `std_db`: standard deviation of the signed error.
        - `q<percent>_abs_db`: one quantile of the absolute error for each value of
          `quantiles`.
        - `pearson`: Pearson correlation of the measured and the simulated RSRP in dBm.
        - `smape`: mean symmetric absolute percentage error.

        A statistic in dB is NaN if no received pair has a path.

    Raises:
        ValueError: If the two arrays do not have the same shape, if no element has a
            measurement, or if a quantile is not from 0 to 1.
    """
    measured_dbm = np.asarray(measured_dbm, dtype=float)
    simulated_dbm = np.asarray(simulated_dbm, dtype=float)
    if measured_dbm.shape != simulated_dbm.shape:
        raise ValueError(
            f"The shapes must be equal, but they are {measured_dbm.shape} and "
            f"{simulated_dbm.shape}."
        )
    received = np.isfinite(measured_dbm)
    if not received.any():
        raise ValueError("The statistics need a minimum of one pair.")
    if any(not 0.0 <= quantile <= 1.0 for quantile in quantiles):
        raise ValueError(f"Each quantile must be from 0 to 1, but they are {list(quantiles)}.")
    measured = measured_dbm[received]
    simulated = simulated_dbm[received]
    error = log_error(measured, simulated)
    has_path = np.isfinite(error)
    found = error[has_path]
    absolute = np.abs(found)
    loss = np.ones_like(error)
    loss[has_path] = smape(10.0 ** (found / 10.0), 1.0)
    statistics = {"pairs": float(error.size), "no_path_fraction": float(np.mean(~has_path))}
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
    statistics["pearson"] = _pearson(measured[has_path], simulated[has_path])
    statistics["smape"] = float(np.mean(loss))
    return pd.Series(statistics)


def grouped_statistics(
    measured_dbm: np.ndarray,
    simulated_dbm: np.ndarray,
    groups: np.ndarray,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    """Calculate the error statistics of each group of received pairs.

    Args:
        measured_dbm: Measured RSRP in dBm, as for `error_statistics`.
        simulated_dbm: Simulated RSRP in dBm, as for `error_statistics`.
        groups: The group of each element, with the same shape. For example the name of
            the cell of each column, or the distance band of each record.
        quantiles: Quantiles of the absolute error, as for `error_statistics`.

    Returns:
        One row of `error_statistics` for each group that has a received pair, in the
        sorted order of the groups. The index has the groups.

    Raises:
        ValueError: If the three arrays do not have the same shape, or if a quantile is
            not from 0 to 1.
    """
    measured_dbm = np.asarray(measured_dbm, dtype=float)
    simulated_dbm = np.asarray(simulated_dbm, dtype=float)
    groups = np.asarray(groups)
    if not measured_dbm.shape == simulated_dbm.shape == groups.shape:
        raise ValueError(
            f"The shapes must be equal, but they are {measured_dbm.shape}, "
            f"{simulated_dbm.shape}, and {groups.shape}."
        )
    received = np.isfinite(measured_dbm)
    pairs = pd.DataFrame(
        {
            "group": groups[received],
            "measured": measured_dbm[received],
            "simulated": simulated_dbm[received],
        }
    )
    rows = {
        group: error_statistics(
            members["measured"].to_numpy(), members["simulated"].to_numpy(), quantiles
        )
        for group, members in pairs.groupby("group", observed=True, sort=True)
    }
    return pd.DataFrame.from_dict(rows, orient="index")


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
