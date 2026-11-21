"""Error statistics of a model against the measured RSRP and against a true path gain map.

The functions use the arrays of the fixed data format: one row for each record and one
column for each cell. NaN in the measured RSRP shows that the record does not have the
cell. NaN in the simulated RSRP shows that the model gives no RSRP for the pair.
The functions do no ray tracing.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def log_error(measured_dbm: np.ndarray, simulated_dbm: np.ndarray) -> np.ndarray:
    """Calculate the signed error between measured and simulated powers in decibels.

    A positive error shows that the simulated power is less than the measured power.

    Args:
        measured_dbm: Measured powers in dBm.
        simulated_dbm: Simulated powers in dBm, with the same shape.

    Returns:
        The error in dB, with the shape of the inputs. A value is NaN if the measured
        power or the simulated power is NaN.
    """
    return np.asarray(measured_dbm) - np.asarray(simulated_dbm)


def smape(measured: np.ndarray, simulated: np.ndarray, epsilon: float) -> np.ndarray:
    """Calculate the symmetric absolute percentage error of each pair of linear values.

    The error is `|m - s| / (m + s + epsilon)`. It is from 0 to 1.

    Args:
        measured: Measured linear values, not negative.
        simulated: Simulated linear values, not negative, with the same shape.
        epsilon: A value of 0 or more. A value more than 0 gives an error of 0, not 0/0,
            when the two values are 0.

    Returns:
        One error for each pair, with the shape of the inputs.
    """
    return np.abs(measured - simulated) / (measured + simulated + epsilon)


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
    only the received pairs with a simulated value. `smape` uses all received pairs: it is
    the symmetric absolute percentage error of the linear powers. The ratio of the two
    powers is `10 ** (error_db / 10)`, thus `smape` does not need the absolute powers. A pair
    with no simulated value has a `smape` of 1.

    Args:
        measured_dbm: Measured RSRP in dBm. NaN shows that the record does not have the
            cell.
        simulated_dbm: Simulated RSRP in dBm, with the same shape. NaN shows that the
            model gives no RSRP for the pair.
        quantiles: Quantiles of the absolute error, each from 0 to 1.

    Returns:
        A series with these values:

        - `pairs`: number of received pairs.
        - `no_path_fraction`: fraction of the received pairs with no simulated value.
        - `bias_db`: mean signed error.
        - `mae_db`: mean absolute error.
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
    loss[has_path] = smape(10.0 ** (found / 10.0), 1.0, epsilon=0.0)
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


def map_statistics(
    true_path_gain: np.ndarray,
    predicted_path_gain_db: np.ndarray,
    quantiles: Sequence[float],
    min_true_db: float = -np.inf,
) -> pd.Series:
    """Calculate the error statistics of a path gain map against the true map.

    The statistics use only the grid cells where the true path gain is more than 0 and at or
    above `min_true_db`. The error is the true path gain minus the predicted path gain, in dB.

    Args:
        true_path_gain: True linear path gain, with shape `[n_cells, ny, nx]`.
        predicted_path_gain_db: Predicted path gain in dB, with the same shape. NaN shows a
            grid cell with no prediction.
        quantiles: Quantiles of the absolute error, each from 0 to 1.
        min_true_db: Smallest true path gain of the statistics, in dB. Use the smallest path
            gain that a UE can measure to compare only the grid cells that matter for a
            measurement.

    Returns:
        A series with `grid_cells`, `no_prediction_fraction`, `bias_db`, `mae_db`,
        `rmse_db`, `std_db`, one `q<percent>_abs_db` for each quantile, and `pearson`.

    Raises:
        ValueError: If the two arrays do not have the same shape, if no true path gain is in
            the range, or if a quantile is not from 0 to 1.
    """
    true_path_gain = np.asarray(true_path_gain, dtype=float)
    predicted = np.asarray(predicted_path_gain_db, dtype=float)
    if true_path_gain.shape != predicted.shape:
        raise ValueError(
            f"The shapes must be equal, but they are {true_path_gain.shape} and {predicted.shape}."
        )
    with np.errstate(divide="ignore"):
        covered = (true_path_gain > 0) & (10.0 * np.log10(true_path_gain) >= min_true_db)
    if not covered.any():
        raise ValueError("The true map must have a minimum of one path gain in the range.")
    if any(not 0.0 <= quantile <= 1.0 for quantile in quantiles):
        raise ValueError(f"Each quantile must be from 0 to 1, but they are {list(quantiles)}.")
    true_db = 10.0 * np.log10(true_path_gain[covered])
    predicted = predicted[covered]
    has_value = np.isfinite(predicted)
    error = true_db[has_value] - predicted[has_value]
    absolute = np.abs(error)
    statistics = {
        "grid_cells": float(covered.sum()),
        "no_prediction_fraction": float(np.mean(~has_value)),
        "bias_db": float(np.mean(error)) if error.size else float("nan"),
        "mae_db": float(np.mean(absolute)) if error.size else float("nan"),
        "rmse_db": float(np.sqrt(np.mean(error**2))) if error.size else float("nan"),
        "std_db": float(np.std(error)) if error.size else float("nan"),
    }
    statistics |= {
        _quantile_name(quantile): float(np.quantile(absolute, quantile))
        if error.size
        else float("nan")
        for quantile in quantiles
    }
    statistics["pearson"] = _pearson(true_db[has_value], predicted[has_value])
    return pd.Series(statistics)
