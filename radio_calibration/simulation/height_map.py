"""Height map of a scene, the outdoor test of a position, and the line-of-sight map.

The functions trace geometric rays with Mitsuba. They do no radio propagation.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import mitsuba as mi
import numpy as np
import sionna.rt

from radio_calibration.core import Cell, RadioMapGrid

# Relative tolerance of the test that the grid starts at the corner of the height map and
# that the cell size is a whole multiple of the tile size.
_ALIGNMENT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class HeightMap:
    """Height of the highest surface in each tile of a top-view grid.

    Attributes:
        heights: Height above the ground in meters, with shape `[ny, nx]`.
            `heights[i, j]` is for the tile with its center at
            `(x_min + (j + 0.5) * tile_size, y_min + (i + 0.5) * tile_size)`.
        x_min: Smallest x of the grid, in meters.
        y_min: Smallest y of the grid, in meters.
        z_ground: z of the ground in the scene, in meters.
        tile_size: Length of one side of a tile, in meters.
    """

    heights: np.ndarray
    x_min: float
    y_min: float
    z_ground: float
    tile_size: float


def _top_surface_z(scene: sionna.rt.Scene, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the z of the first surface below the top of the scene. NaN if there is none."""
    top = float(scene.mi_scene.bbox().max.z) + 1.0
    origins = mi.Point3f(x.astype(np.float32), y.astype(np.float32), np.full(x.size, top, "f4"))
    hit = scene.mi_scene.ray_intersect(mi.Ray3f(origins, mi.Vector3f(0.0, 0.0, -1.0)))
    return np.where(hit.is_valid().numpy(), top - hit.t.numpy(), np.nan)


def building_height_map(scene: sionna.rt.Scene, tile_size: float) -> HeightMap:
    """Measure the height of the scene on a grid that covers the scene.

    Args:
        scene: The scene.
        tile_size: Length of one side of a grid tile, in meters.

    Returns:
        The height map. A tile with no surface has the height 0.

    Raises:
        ValueError: If `tile_size` is not more than 0.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be more than 0, but it is {tile_size}.")
    bbox = scene.mi_scene.bbox()
    x_min, y_min, z_ground = float(bbox.min.x), float(bbox.min.y), float(bbox.min.z)
    nx = max(int((float(bbox.max.x) - x_min) // tile_size), 1)
    ny = max(int((float(bbox.max.y) - y_min) // tile_size), 1)
    x, y = np.meshgrid(
        x_min + (np.arange(nx) + 0.5) * tile_size, y_min + (np.arange(ny) + 0.5) * tile_size
    )
    z = _top_surface_z(scene, x.ravel(), y.ravel()).reshape(ny, nx)
    heights = np.clip(np.nan_to_num(z, nan=z_ground) - z_ground, 0.0, None)
    return HeightMap(heights, x_min, y_min, z_ground, tile_size)


def is_outdoor(scene: sionna.rt.Scene, positions: np.ndarray) -> np.ndarray:
    """Find the positions that have no object above them.

    Args:
        scene: The scene.
        positions: Positions in meters, with shape `[n, 3]`.

    Returns:
        A boolean array with shape `[n]`. A value is True if the first surface below
        the top of the scene is at or below the position. A position outside the scene
        gives False.
    """
    z_top = _top_surface_z(scene, positions[:, 0], positions[:, 1])
    return z_top <= positions[:, 2]


def cell_buildings(
    height_map: HeightMap, grid: RadioMapGrid, min_building_height: float
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize the buildings of the height map in each grid cell of a radio-map grid.

    The grid must start at the corner of the height map, and its cell size must be a whole
    multiple of the tile size. A part of a grid cell outside the height map has no building.

    Args:
        height_map: The height map of the scene.
        grid: The radio-map grid.
        min_building_height: A tile above this height, in meters, is a building.

    Returns:
        The height of the highest tile in each grid cell, in meters, and the fraction of the
        tiles of each grid cell that are a building. Each array has shape `[ny, nx]`.

    Raises:
        ValueError: If the grid does not start at the corner of the height map, or if the
            cell size is not a whole multiple of the tile size.
    """
    ratio = grid.cell_size / height_map.tile_size
    tiles = round(ratio)
    if tiles < 1 or abs(ratio - tiles) > _ALIGNMENT_TOLERANCE * ratio:
        raise ValueError(
            f"The cell size {grid.cell_size} m must be a whole multiple of the tile size "
            f"{height_map.tile_size} m."
        )
    corner = np.array([grid.x_min - height_map.x_min, grid.y_min - height_map.y_min])
    if np.any(np.abs(corner) > _ALIGNMENT_TOLERANCE * grid.cell_size):
        raise ValueError("The grid must start at the corner of the height map.")
    rows, columns = grid.ny * tiles, grid.nx * tiles
    heights = np.zeros((rows, columns))
    source = height_map.heights[:rows, :columns]
    heights[: source.shape[0], : source.shape[1]] = source
    blocks = heights.reshape(grid.ny, tiles, grid.nx, tiles)
    max_height = blocks.max(axis=(1, 3))
    fraction = (blocks > min_building_height).mean(axis=(1, 3))
    return max_height, fraction


def los_map(scene: sionna.rt.Scene, cells: Sequence[Cell], grid: RadioMapGrid) -> np.ndarray:
    """Find the grid cells that each antenna can see on a straight line.

    The function traces one geometric ray from the antenna of each cell to the center of
    each grid cell. It does not use the Monte Carlo rays of the radio-map solver, thus the
    result has no sampling noise at a long distance.

    Args:
        scene: The scene.
        cells: The cells. The antenna of a cell is at `(x, y, z)` of the cell.
        grid: The radio-map grid. The rays end at the height `grid.z`.

    Returns:
        A boolean array with shape `[len(cells), ny, nx]`. A value is True if no surface of
        the scene is between the antenna and the center of the grid cell.
    """
    targets = grid.centers().reshape(-1, 3)
    result = np.empty((len(cells), targets.shape[0]), dtype=bool)
    for index, cell in enumerate(cells):
        antenna = np.array([cell.x, cell.y, cell.z])
        offset = targets - antenna
        distance = np.linalg.norm(offset, axis=1)
        direction = offset / np.maximum(distance, np.finfo(float).tiny)[:, None]
        origins = np.broadcast_to(antenna, targets.shape)
        rays = mi.Ray3f(
            mi.Point3f(*origins.T.astype(np.float32)),
            mi.Vector3f(*direction.T.astype(np.float32)),
        )
        hit = scene.mi_scene.ray_intersect(rays)
        hit_distance = np.where(hit.is_valid().numpy(), hit.t.numpy(), np.inf)
        result[index] = hit_distance >= distance
    return result.reshape(len(cells), grid.ny, grid.nx)
