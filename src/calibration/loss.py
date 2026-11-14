"""Loss, power scale, and error metric of the calibration (arXiv:2311.18558)."""

import numpy as np


def dbm_to_mw(power_dbm: np.ndarray) -> np.ndarray:
    """Change a power from dBm to milliwatts. NaN stays NaN."""
    return 10.0 ** (np.asarray(power_dbm, dtype=float) / 10.0)


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
    """Estimate the factor that makes the measured powers agree with the simulated powers.

    This is the least-squares solution of Eq. 43 of arXiv:2311.18558.

    Args:
        measured: Measured powers in a linear unit.
        simulated: Simulated powers in the same unit, with the same shape.

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


def absolute_log_error(
    measured_dbm: np.ndarray, simulated_dbm: np.ndarray, scale: float
) -> np.ndarray:
    """Calculate the absolute error between measured and simulated powers in decibels.

    This is Eq. 44 of arXiv:2311.18558. The scale multiplies the measured power, as in
    Algorithm 1 of the paper.

    Args:
        measured_dbm: Measured powers in dBm.
        simulated_dbm: Simulated powers in dBm, with the same shape.
        scale: The linear power scale from the calibration.

    Returns:
        The error in dB, with the shape of the inputs. A value is NaN if the measured
        power or the simulated power is NaN.
    """
    return np.abs(np.asarray(measured_dbm) + 10.0 * np.log10(scale) - np.asarray(simulated_dbm))
