"""True RSRP of a synthetic scenario, from the true path gain map."""

from collections.abc import Callable, Sequence

import numpy as np

from radio_calibration.core import RadioMapGrid, path_gain_to_db, rsrp_from_map


def map_rsrp_fn(
    path_gain: np.ndarray, grid: RadioMapGrid, power_rs_dbm: Sequence[float]
) -> Callable[[np.ndarray], np.ndarray]:
    """Make a function that gives the true RSRP at positions.

    The RSRP comes from the true map with `rsrp_from_map`. Thus the MDT records need no more
    ray tracing after the true map.

    Args:
        path_gain: True linear path gain, with shape `[n_cells, ny, nx]`.
        grid: The grid of the map.
        power_rs_dbm: Reference-signal power of each cell, in dBm.

    Returns:
        A function that changes positions with shape `[n, 3]` into the RSRP in dBm with
        shape `[n, n_cells]`. A value is NaN where the path gain is 0 or the position is
        outside the grid.

    Raises:
        ValueError: If the shape of `path_gain` does not agree with the grid and the powers.
    """
    path_gain_db = path_gain_to_db(path_gain)
    power = np.asarray(power_rs_dbm, dtype=float)
    rsrp_from_map(path_gain_db, grid, np.empty((0, 2)), power)
    return lambda positions: rsrp_from_map(path_gain_db, grid, positions, power)
