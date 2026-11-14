"""File format of a calibration result: the neural material and the power scale."""

import math
from pathlib import Path

import numpy as np

from src.simulation import NeuralRadioMaterial

_SCALE_KEY = "scale"


def save_calibration(path: Path, material: NeuralRadioMaterial, scale: float) -> None:
    """Write the material and the power scale to one NumPy `.npz` file.

    The two are in one file because the calibration learns them together: the weights
    are correct only with their scale.

    Args:
        path: Path of the file. Its suffix must be `.npz`.
        material: The calibrated material.
        scale: The linear power scale from the calibration.

    Raises:
        ValueError: If the suffix of `path` is not `.npz`.
    """
    if path.suffix != ".npz":
        raise ValueError(f"The file name must have the suffix .npz, but it is '{path.name}'.")
    np.savez(path, **material.state(), **{_SCALE_KEY: np.array(float(scale))})


def load_calibration(path: Path, name: str) -> tuple[NeuralRadioMaterial, float]:
    """Read a file that `save_calibration` wrote.

    Args:
        path: Path of the file.
        name: Name of the material.

    Returns:
        The material and the linear power scale.

    Raises:
        ValueError: If the file has an object array, if a key is missing, if the number
            of weights does not agree with the network structure, or if the scale is not
            a finite number more than 0.
    """
    # allow_pickle=False: a file from an unknown source must not run code.
    with np.load(path, allow_pickle=False) as data:
        state = {key: data[key] for key in data.files}
    if _SCALE_KEY not in state:
        raise ValueError(f"The file does not have the key '{_SCALE_KEY}'.")
    scale = float(state.pop(_SCALE_KEY))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"The scale must be a finite number more than 0, but it is {scale}.")
    return NeuralRadioMaterial.from_state(state, name), scale
