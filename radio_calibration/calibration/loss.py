"""Error metrics of the path gain correction: SMAPE and the error in dB."""

import numpy as np


def dbm_to_mw(power_dbm: np.ndarray) -> np.ndarray:
    """Change a power from dBm to milliwatts. NaN stays NaN."""
    return 10.0 ** (np.asarray(power_dbm, dtype=float) / 10.0)


def smape(measured, simulated, epsilon: float):
    """Calculate the symmetric absolute percentage error of each pair of linear values.

    The error is `|m - s| / (m + s + epsilon)`. It is from 0 to 1.

    Args:
        measured: Measured linear values, not negative. A NumPy array or a PyTorch tensor.
        simulated: Simulated linear values, not negative, of the same type and shape.
        epsilon: A value of 0 or more. A value more than 0 gives an error of 0, not 0/0,
            when the two values are 0.

    Returns:
        One error for each pair, with the type and the shape of the inputs.
    """
    return abs(measured - simulated) / (measured + simulated + epsilon)


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


def absolute_log_error(measured_dbm: np.ndarray, simulated_dbm: np.ndarray) -> np.ndarray:
    """Calculate the absolute error between measured and simulated powers in decibels.

    Args:
        measured_dbm: Measured powers in dBm.
        simulated_dbm: Simulated powers in dBm, with the same shape.

    Returns:
        The absolute value of `log_error`. A value is NaN if an input is NaN.
    """
    return np.abs(log_error(measured_dbm, simulated_dbm))
