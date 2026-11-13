"""Where and when the UEs make MDT records.

The model has two parts. The residential part is in proportion to the building density.
The business part is in proportion to a power of the building density, thus it is more
concentrated. Each part has one weight for each hour of the day. The two daily patterns
agree with the residential and office patterns of arXiv:1510.04026.
"""

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import uniform_filter

from src.simulation import HeightMap

_HOURS_PER_DAY = 24
_MINUTES_PER_HOUR = 60


def _check_profile(profile: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(profile, dtype=float)
    if values.shape != (_HOURS_PER_DAY,) or (values < 0).any():
        raise ValueError(f"{name} must have 24 values that are not negative.")
    return values


def building_density(
    height_map: HeightMap, min_building_height_m: float, radius_m: float
) -> np.ndarray:
    """Calculate the fraction of building tiles near each grid tile.

    Args:
        height_map: Height map of the scene.
        min_building_height_m: A grid tile above this height is a building.
        radius_m: Half of the side of the square window around each tile, in meters.

    Returns:
        The density in [0, 1], with the shape of `height_map.heights`. The area outside
        the grid counts as an area with no building.
    """
    buildings = (height_map.heights > min_building_height_m).astype(float)
    window = 2 * round(radius_m / height_map.tile_size) + 1
    return np.clip(uniform_filter(buildings, size=window, mode="constant"), 0.0, 1.0)


def spatial_weights(
    density: np.ndarray,
    outdoor: np.ndarray,
    hour: int,
    residential_profile: Sequence[float],
    business_profile: Sequence[float],
    business_exponent: float,
) -> np.ndarray:
    """Calculate the probability that a record of this hour is in each grid tile.

    Args:
        density: Building density of each grid tile.
        outdoor: True for each grid tile that is not a building.
        hour: Hour of the day, from 0 to 23.
        residential_profile: Weight of the residential part for each hour.
        business_profile: Weight of the business part for each hour.
        business_exponent: Exponent of the density in the business part.

    Returns:
        The probabilities, with the shape of `density`. The sum is 1. A building tile
        has the probability 0.

    Raises:
        ValueError: If a profile does not have 24 values that are not negative, if the two
            profiles are 0 at this hour, or if no outdoor tile has a density above 0.
    """
    residential = _check_profile(residential_profile, "residential_profile")[hour]
    business = _check_profile(business_profile, "business_profile")[hour]
    if residential + business == 0:
        raise ValueError(f"The two profiles are 0 at hour {hour}.")
    base = np.where(outdoor, density, 0.0)
    if base.sum() == 0:
        raise ValueError("No outdoor tile has a building density above 0.")
    concentrated = base**business_exponent
    weights = residential * base / base.sum() + business * concentrated / concentrated.sum()
    return weights / weights.sum()


def slot_counts(
    rng: np.random.Generator,
    num_days: int,
    slot_minutes: int,
    records_per_day: float,
    residential_profile: Sequence[float],
    business_profile: Sequence[float],
) -> np.ndarray:
    """Draw the number of records in each time slot from a Poisson distribution.

    Args:
        rng: Random generator.
        num_days: Number of days.
        slot_minutes: Length of a slot in minutes. It must divide 60.
        records_per_day: Mean number of records in one day.
        residential_profile: Weight of the residential part for each hour.
        business_profile: Weight of the business part for each hour.

    Returns:
        The integer count of each slot, with shape `[num_days * 1440 / slot_minutes]`.

    Raises:
        ValueError: If a profile does not have 24 values that are not negative, if the sum
            of the profiles is 0, or if `slot_minutes` does not divide 60.
    """
    hourly = _check_profile(residential_profile, "residential_profile") + _check_profile(
        business_profile, "business_profile"
    )
    if hourly.sum() == 0:
        raise ValueError("The sum of the two profiles is 0.")
    if slot_minutes <= 0 or _MINUTES_PER_HOUR % slot_minutes:
        raise ValueError(f"slot_minutes must divide 60, but it is {slot_minutes}.")
    slots_per_hour = _MINUTES_PER_HOUR // slot_minutes
    day_share = np.repeat(hourly / hourly.sum(), slots_per_hour) / slots_per_hour
    return rng.poisson(records_per_day * np.tile(day_share, num_days))
