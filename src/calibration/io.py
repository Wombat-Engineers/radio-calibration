"""File format of a calibration result: the neural material with its power scale."""

from pathlib import Path

import numpy as np

from src.simulation import NeuralRadioMaterial


def save_calibration(path: Path, material: NeuralRadioMaterial) -> None:
    """Write the calibrated material to one NumPy `.npz` file.

    The file has the result of `NeuralRadioMaterial.state`. Thus it has the power scale:
    the weights are correct only with their scale.

    Args:
        path: Path of the file. Its suffix must be `.npz`.
        material: The calibrated material.

    Raises:
        ValueError: If the suffix of `path` is not `.npz`, or if the material has no scale.
    """
    if path.suffix != ".npz":
        raise ValueError(f"The file name must have the suffix .npz, but it is '{path.name}'.")
    if material.scale is None:
        raise ValueError("The material has no scale. Calibrate it before you save it.")
    np.savez(path, **material.state())


def load_calibration(path: Path, name: str) -> NeuralRadioMaterial:
    """Read a file that `save_calibration` wrote.

    Args:
        path: Path of the file.
        name: Name of the material.

    Returns:
        The material with its power scale.

    Raises:
        ValueError: If the file has an object array, if a key is missing, if the file has
            no scale, if the number of weights does not agree with the network structure,
            or if the scale is not a finite number more than 0.
    """
    # allow_pickle=False: a file from an unknown source must not run code.
    with np.load(path, allow_pickle=False) as data:
        state = {key: data[key] for key in data.files}
    material = NeuralRadioMaterial.from_state(state, name)
    if material.scale is None:
        raise ValueError("The file has no power scale.")
    return material
