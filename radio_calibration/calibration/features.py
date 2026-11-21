"""Input channels of the U-Net and their normalization."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from radio_calibration.core import MapInputs, RadioMapGrid

FEATURE_NAMES = (
    "path_gain_db",
    "has_path",
    "max_height",
    "building_fraction",
    "los",
    "distance",
)

# The floor of the path gain is this number of decibels below the smallest path gain that a
# UE can report. Assumption.
FLOOR_MARGIN_DB = 10.0


def sensitivity_floor_db(min_rsrp_dbm: float, power_rs_dbm: Sequence[float]) -> float:
    """Return the floor of the path gain in dB.

    A path gain below `min_rsrp_dbm - max(power_rs_dbm)` gives no MDT report for any cell.
    The floor is `FLOOR_MARGIN_DB` below this value. A smaller path gain has no effect on a
    measurement, thus the model replaces it with the floor.

    Args:
        min_rsrp_dbm: The smallest RSRP that a UE reports, in dBm.
        power_rs_dbm: Reference-signal power of each cell, in dBm.

    Returns:
        The floor in dB.
    """
    return float(min_rsrp_dbm - np.max(power_rs_dbm) - FLOOR_MARGIN_DB)


def _path_gain_db(path_gain: np.ndarray, floor_db: float) -> np.ndarray:
    """Return the path gain in dB, with `floor_db` where it is 0 or below the floor."""
    with np.errstate(divide="ignore"):
        return np.maximum(np.where(path_gain > 0, 10.0 * np.log10(path_gain), floor_db), floor_db)


@dataclass(frozen=True)
class Normalization:
    """Statistics that put the input channels on a similar scale.

    Attributes:
        floor_db: Path gain in dB that replaces a path gain of 0 or a smaller path gain. See
            `sensitivity_floor_db`.
        path_gain_mean: Mean of the path gain in dB, with the floor.
        path_gain_std: Standard deviation of the path gain in dB, with the floor.
        height_mean: Mean of the building height, in meters.
        height_std: Standard deviation of the building height, in meters.

    Raises:
        ValueError: If a value is not finite, or if a standard deviation is not more than 0.
    """

    floor_db: float
    path_gain_mean: float
    path_gain_std: float
    height_mean: float
    height_std: float

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError(f"All values must be finite, but they are {asdict(self)}.")
        if not (self.path_gain_std > 0 and self.height_std > 0):
            raise ValueError("The standard deviations must be more than 0.")

    @classmethod
    def fit(cls, inputs: Sequence[MapInputs], floor_db: float) -> "Normalization":
        """Calculate the statistics of the input maps.

        Args:
            inputs: The input maps of the pretraining data.
            floor_db: The floor of the path gain in dB, from `sensitivity_floor_db`.

        Returns:
            The normalization.

        Raises:
            ValueError: If there is no map, or if the maps have no spread.
        """
        if not inputs:
            raise ValueError("The normalization needs a minimum of one input map.")
        path_gain_db = np.concatenate(
            [_path_gain_db(m.path_gain, floor_db).ravel() for m in inputs]
        )
        heights = np.concatenate([m.max_height.ravel() for m in inputs])
        return cls(
            floor_db=floor_db,
            path_gain_mean=float(path_gain_db.mean()),
            path_gain_std=float(path_gain_db.std()),
            height_mean=float(heights.mean()),
            height_std=float(heights.std()),
        )

    def state(self) -> dict[str, float]:
        """Return the values as a dictionary of floats."""
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_state(cls, state: dict[str, float]) -> "Normalization":
        """Make the normalization from the dictionary of `state`.

        Raises:
            ValueError: If a key is missing.
        """
        missing = [name for name in cls.__dataclass_fields__ if name not in state]
        if missing:
            raise ValueError(f"The normalization state does not have the keys {missing}.")
        return cls(**{name: float(state[name]) for name in cls.__dataclass_fields__})


def padded_shape(grid: RadioMapGrid, multiple: int) -> tuple[int, int]:
    """Return the number of rows and columns of the grid, each up to a multiple of `multiple`."""
    return (-(-grid.ny // multiple) * multiple, -(-grid.nx // multiple) * multiple)


def pad(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Add zeros at the end of the last two axes of `array` until they have `shape`."""
    widths = [(0, 0)] * (array.ndim - 2) + [
        (0, shape[0] - array.shape[-2]),
        (0, shape[1] - array.shape[-1]),
    ]
    return np.pad(array, widths)


def base_path_gain_db(inputs: MapInputs, floor_db: float) -> np.ndarray:
    """Return the simulated path gain in dB, with `floor_db` where it is 0 or below the floor.

    The U-Net adds its output to this map. The same map is the path gain input channel
    before the normalization.

    Returns:
        The path gain in dB, with shape `[n_cells, ny, nx]`.
    """
    return _path_gain_db(inputs.path_gain, floor_db)


def feature_tensor(
    inputs: MapInputs,
    normalization: Normalization,
    multiple: int,
    channels: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Make the input channels of the U-Net for each cell.

    The channels are:

    - `path_gain_db`: simulated path gain in dB, with the floor, normalized.
    - `has_path`: 1 where the simulated path gain is more than 0.
    - `max_height`: highest building height in the grid cell, normalized.
    - `building_fraction`: fraction of the grid cell with a building.
    - `los`: 1 where the antenna has a line of sight to the grid cell.
    - `distance`: horizontal distance to the antenna, divided by the diagonal of the grid.

    Args:
        inputs: The input maps.
        normalization: The statistics of the pretraining data.
        multiple: The number of rows and columns becomes a multiple of this value. The added
            rows and columns are 0.
        channels: Names of the channels to use, in this order. Each name must be in
            `FEATURE_NAMES`.

    Returns:
        A float32 array with shape `[n_cells, len(channels), rows, columns]`.

    Raises:
        ValueError: If a channel name is not in `FEATURE_NAMES`, or if `channels` is empty.
    """
    unknown = [name for name in channels if name not in FEATURE_NAMES]
    if unknown or not channels:
        raise ValueError(f"The channels must be names of {FEATURE_NAMES}, but they are {channels}.")
    num_cells = inputs.num_cells
    plane = (num_cells, inputs.grid.ny, inputs.grid.nx)
    centers = inputs.grid.centers()[..., :2]
    distance = np.linalg.norm(centers[None] - inputs.cell_xy[:, None, None, :], axis=-1)
    path_gain_db = base_path_gain_db(inputs, normalization.floor_db)
    maps = {
        "path_gain_db": (path_gain_db - normalization.path_gain_mean) / normalization.path_gain_std,
        "has_path": (inputs.path_gain > 0).astype(float),
        "max_height": np.broadcast_to(
            (inputs.max_height - normalization.height_mean) / normalization.height_std, plane
        ),
        "building_fraction": np.broadcast_to(inputs.building_fraction, plane),
        "los": inputs.los.astype(float),
        "distance": distance / inputs.grid.diagonal,
    }
    stacked = np.stack([maps[name] for name in channels], axis=1)
    return pad(stacked, padded_shape(inputs.grid, multiple)).astype(np.float32)
