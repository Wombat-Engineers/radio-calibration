"""Loss, power scale, and error metric of the calibration (arXiv:2311.18558)."""

import numpy as np


def dbm_to_mw(power_dbm: np.ndarray) -> np.ndarray:
    """Change a power from dBm to milliwatts. NaN stays NaN."""
    return 10.0 ** (np.asarray(power_dbm, dtype=float) / 10.0)


def received_powers(
    measured_dbm: np.ndarray, simulated_dbm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the measured and the simulated powers of the received (record, cell) pairs.

    A pair is received if its measured power is not NaN. A simulated NaN shows that no path
    connects the cell and the record. It becomes a power of 0, not a missing pair: the pair
    has a measurement, and the model gives no power for it.

    Args:
        measured_dbm: Measured powers in dBm. NaN shows a pair with no measurement.
        simulated_dbm: Simulated powers in dBm, with the same shape. NaN shows no path.

    Returns:
        The measured and the simulated powers in milliwatts, each with one value for each
        received pair, in row-major order.

    Raises:
        ValueError: If the two arrays do not have the same shape.
    """
    measured_dbm = np.asarray(measured_dbm, dtype=float)
    simulated_dbm = np.asarray(simulated_dbm, dtype=float)
    if measured_dbm.shape != simulated_dbm.shape:
        raise ValueError(
            f"The shapes must be equal, but they are {measured_dbm.shape} and "
            f"{simulated_dbm.shape}."
        )
    received = np.isfinite(measured_dbm)
    return dbm_to_mw(measured_dbm[received]), np.nan_to_num(dbm_to_mw(simulated_dbm[received]))


def smape(measured, simulated):
    """Calculate the symmetric absolute percentage error of each pair of values.

    This is Eq. 39 of arXiv:2311.18558.

    Args:
        measured: Measured powers, not negative. A Dr.Jit array or a NumPy array.
        simulated: Simulated powers, not negative, of the same type and size.

    Returns:
        One error for each pair, between 0 and 1. The error is 1 if only one of the two
        values is 0. The error is not a number if the two values are 0.
    """
    return abs(measured - simulated) / (measured + simulated)


def scale_estimate(measured: np.ndarray, simulated: np.ndarray) -> float:
    """Estimate the power scale that divides the simulated powers.

    This is the least-squares solution of Eq. 43 of arXiv:2311.18558. The simulated
    powers divided by the scale agree with the measured powers.

    Args:
        measured: Measured powers in a linear unit.
        simulated: Simulated powers in the same unit, with the same shape. They must have
            no scale. A scaled power gives the wrong scale.

    Returns:
        The factor `alpha` that gives the minimum of the mean of
        `(alpha * measured - simulated) ** 2`.

    Raises:
        ValueError: If all measured powers are 0, or if all simulated powers are 0. The
            factor is then not defined or is 0.
    """
    measured = np.asarray(measured, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    numerator = float(np.sum(measured * simulated))
    denominator = float(np.sum(measured**2))
    if denominator <= 0 or numerator <= 0:
        raise ValueError("The scale has no value: all measured or all simulated powers are 0.")
    return numerator / denominator


def update_scale(previous: float | None, estimate: float, decay: float) -> float:
    """Calculate the exponential moving average of the scale.

    This is Eq. 42 of arXiv:2311.18558.

    Args:
        previous: The average before this step. None at the first step.
        estimate: The estimate of this step, from `scale_estimate`.
        decay: Weight of the previous average, from 0 to less than 1.

    Returns:
        The new average. At the first step, it is `estimate`.

    Raises:
        ValueError: If `decay` is less than 0, or is 1 or more.
    """
    if not 0 <= decay < 1:
        raise ValueError(f"decay must be from 0 to less than 1, but it is {decay}.")
    if previous is None:
        return estimate
    return decay * previous + (1 - decay) * estimate


def log_error(measured_dbm: np.ndarray, simulated_dbm: np.ndarray) -> np.ndarray:
    """Calculate the signed error between measured and simulated powers in decibels.

    This is Eq. 44 of arXiv:2311.18558 without the absolute value. A positive error shows
    that the simulated power is less than the measured power.

    Args:
        measured_dbm: Measured powers in dBm.
        simulated_dbm: Simulated powers in dBm, with the same shape. They must have the
            power scale of the model (the argument `scale` of
            `src.simulation.compute_rsrp_dbm`).

    Returns:
        The error in dB, with the shape of the inputs. A value is NaN if the measured
        power or the simulated power is NaN.
    """
    return np.asarray(measured_dbm) - np.asarray(simulated_dbm)


def absolute_log_error(measured_dbm: np.ndarray, simulated_dbm: np.ndarray) -> np.ndarray:
    """Calculate the absolute error between measured and simulated powers in decibels.

    This is Eq. 44 of arXiv:2311.18558: the absolute value of `log_error`.

    Args:
        measured_dbm: Measured powers in dBm.
        simulated_dbm: Simulated powers in dBm with the power scale, as in `log_error`.

    Returns:
        The error in dB, with the shape of the inputs. A value is NaN if the measured
        power or the simulated power is NaN.
    """
    return np.abs(log_error(measured_dbm, simulated_dbm))
