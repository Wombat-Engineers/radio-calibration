"""Generation of the MDT records."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import truncnorm

from src.core import Cell
from src.scenario.traffic import slot_counts, spatial_weights
from src.simulation import HeightMap

_HOURS_PER_DAY = 24
_MINUTES_PER_HOUR = 60


def _draw_positions(
    rng: np.random.Generator,
    hours: np.ndarray,
    weights: dict[int, np.ndarray],
    height_map: HeightMap,
    ue_height: DictConfig,
) -> np.ndarray:
    """Draw one position for each record. Return the positions with shape `[n, 3]`."""
    nx = height_map.heights.shape[1]
    flat = np.empty(hours.size, dtype=int)
    for hour in np.unique(hours):
        chosen = hours == hour
        flat[chosen] = rng.choice(weights[hour].size, size=chosen.sum(), p=weights[hour].ravel())
    row, col = np.divmod(flat, nx)
    offset = rng.uniform(size=(hours.size, 2))
    low = (ue_height.min_m - ue_height.mean_m) / ue_height.std_m
    high = (ue_height.max_m - ue_height.mean_m) / ue_height.std_m
    height = truncnorm.rvs(
        low, high, loc=ue_height.mean_m, scale=ue_height.std_m, size=hours.size, random_state=rng
    )
    return np.column_stack(
        [
            height_map.x_min + (col + offset[:, 0]) * height_map.tile_size,
            height_map.y_min + (row + offset[:, 1]) * height_map.tile_size,
            height_map.z_ground + height,
        ]
    )


def sample_mdt(
    rng: np.random.Generator,
    rsrp_fn: Callable[[np.ndarray], np.ndarray],
    outdoor_fn: Callable[[np.ndarray], np.ndarray],
    cells: Sequence[Cell],
    height_map: HeightMap,
    min_building_height_m: float,
    density: np.ndarray,
    traffic: DictConfig,
    ue_height: DictConfig,
    measurement: DictConfig,
) -> pd.DataFrame:
    """Generate MDT records that are independent of each other.

    A record has the start time of its slot, a position with a GNSS error, and the
    measured RSRP of each cell. The function samples a record again if its true position
    is not outdoor or if the record has no cell.

    Args:
        rng: Random generator.
        rsrp_fn: Gives the true RSRP in dBm, with shape `[n, len(cells)]`, for positions
            with shape `[n, 3]`. NaN shows that the cell has no signal at the position.
        outdoor_fn: Gives a boolean array with shape `[n]` for positions with shape
            `[n, 3]`. True shows that the position is outdoor.
        cells: The cells, in the order of the columns that `rsrp_fn` gives.
        height_map: Height map of the scene.
        min_building_height_m: A grid tile above this height is a building.
        density: Building density of each grid tile.
        traffic: Keys `start`, `num_days`, `slot_minutes`, `records_per_day`,
            `business_exponent`, `residential_profile`, `business_profile`.
        ue_height: Keys `mean_m`, `std_m`, `min_m`, `max_m`.
        measurement: Keys `rsrp_noise_mean_db`, `rsrp_noise_std_db`, `gnss_std_m`,
            `min_rsrp_dbm`, `rsrp_resolution_db`, `max_resample_rounds`.

    Returns:
        A frame with the columns `timestamp`, `x`, `y`, `z`, and `rsrp_<cell name>` for
        each cell. A missing RSRP is NaN. The rows are in the order of the time.

    Raises:
        RuntimeError: If some records have no valid position after
            `max_resample_rounds` rounds.
    """
    counts = slot_counts(
        rng,
        traffic.num_days,
        traffic.slot_minutes,
        traffic.records_per_day,
        traffic.residential_profile,
        traffic.business_profile,
    )
    slots = np.repeat(np.arange(counts.size), counts)
    hours = (slots * traffic.slot_minutes // _MINUTES_PER_HOUR) % _HOURS_PER_DAY
    outdoor = height_map.heights <= min_building_height_m
    weights = {
        hour: spatial_weights(
            density,
            outdoor,
            hour,
            traffic.residential_profile,
            traffic.business_profile,
            traffic.business_exponent,
        )
        for hour in np.unique(hours)
    }

    positions = np.empty((slots.size, 3))
    rsrp = np.full((slots.size, len(cells)), np.nan)
    pending = np.arange(slots.size)
    for _ in range(measurement.max_resample_rounds):
        if pending.size == 0:
            break
        drawn = _draw_positions(rng, hours[pending], weights, height_map, ue_height)
        is_out = np.asarray(outdoor_fn(drawn), dtype=bool)
        measured = np.full((pending.size, len(cells)), np.nan)
        noise = rng.normal(
            measurement.rsrp_noise_mean_db,
            measurement.rsrp_noise_std_db,
            size=(int(is_out.sum()), len(cells)),
        )
        step = measurement.rsrp_resolution_db
        measured[is_out] = np.round((rsrp_fn(drawn[is_out]) + noise) / step) * step
        measured[measured < measurement.min_rsrp_dbm] = np.nan
        accepted = np.isfinite(measured).any(axis=1)
        positions[pending[accepted]] = drawn[accepted]
        rsrp[pending[accepted]] = measured[accepted]
        pending = pending[~accepted]
    if pending.size:
        raise RuntimeError(
            f"{pending.size} records have no valid position after "
            f"{measurement.max_resample_rounds} rounds."
        )

    reported_xy = positions[:, :2] + rng.normal(0.0, measurement.gnss_std_m, size=(slots.size, 2))
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp(traffic.start)
            + pd.to_timedelta(slots * traffic.slot_minutes, unit="min"),
            "x": reported_xy[:, 0],
            "y": reported_xy[:, 1],
            "z": positions[:, 2],
        }
    )
    for index, cell in enumerate(cells):
        frame[f"rsrp_{cell.name}"] = rsrp[:, index]
    return frame
