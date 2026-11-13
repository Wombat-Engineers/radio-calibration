"""Placement of the sites and their cells."""

from collections.abc import Sequence

import numpy as np
from omegaconf import DictConfig

from src.core import Cell
from src.simulation import HeightMap

# Directions from the center of an equilateral triangle to its corners.
_CORNER_ANGLES_DEG = (90.0, 210.0, 330.0)


def place_sites(
    height_map: HeightMap,
    min_building_height_m: float,
    center_xy: tuple[float, float],
    isd_m: float,
    mast_height_m: float,
    sector_azimuths_deg: Sequence[float],
    downtilt_deg: float,
    power_rs_dbm: float,
    transmitter: DictConfig,
) -> list[Cell]:
    """Put three sites at the corners of an equilateral triangle.

    A corner in a building tile moves to the center of the nearest outdoor tile. Each
    site has one cell for each azimuth.

    Args:
        height_map: Height map of the scene.
        min_building_height_m: A grid tile above this height is a building.
        center_xy: Center of the triangle, in meters.
        isd_m: Side of the triangle, in meters.
        mast_height_m: Antenna height above the ground, in meters.
        sector_azimuths_deg: Azimuth of each cell of a site.
        downtilt_deg: Downtilt of each cell.
        power_rs_dbm: Reference-signal power of each cell.
        transmitter: Array configuration of each cell: `num_rows`, `num_cols`,
            `vertical_spacing`, `horizontal_spacing`, `pattern`, `polarization`.

    Returns:
        The cells, with the name `site<i>_sector<k>`.

    Raises:
        ValueError: If a corner is outside the height map, or if the height map has no
            outdoor tile.
    """
    ny, nx = height_map.heights.shape
    size = height_map.tile_size
    outdoor_rows, outdoor_cols = np.nonzero(height_map.heights <= min_building_height_m)
    if outdoor_rows.size == 0:
        raise ValueError("The height map has no outdoor tile.")
    outdoor_x = height_map.x_min + (outdoor_cols + 0.5) * size
    outdoor_y = height_map.y_min + (outdoor_rows + 0.5) * size

    circumradius = isd_m / np.sqrt(3.0)
    cells = []
    for site, angle in enumerate(np.deg2rad(_CORNER_ANGLES_DEG)):
        x = center_xy[0] + circumradius * np.cos(angle)
        y = center_xy[1] + circumradius * np.sin(angle)
        col = int((x - height_map.x_min) // size)
        row = int((y - height_map.y_min) // size)
        if not (0 <= col < nx and 0 <= row < ny):
            raise ValueError(f"Site {site} at ({x:.1f}, {y:.1f}) is outside the scene.")
        if height_map.heights[row, col] > min_building_height_m:
            nearest = np.argmin((outdoor_x - x) ** 2 + (outdoor_y - y) ** 2)
            x, y = outdoor_x[nearest], outdoor_y[nearest]
        cells += [
            Cell(
                name=f"site{site}_sector{sector}",
                site=site,
                x=float(x),
                y=float(y),
                z=height_map.z_ground + mast_height_m,
                azimuth_deg=float(azimuth),
                downtilt_deg=downtilt_deg,
                power_rs_dbm=power_rs_dbm,
                **transmitter,
            )
            for sector, azimuth in enumerate(sector_azimuths_deg)
        ]
    return cells
