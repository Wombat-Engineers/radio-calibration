"""Height map of a scene and the outdoor test of a position.

The functions trace geometric rays with Mitsuba. They do no radio propagation.
"""

from dataclasses import dataclass

import mitsuba as mi
import numpy as np

# isort: off
# This import sets the Mitsuba variant. Sionna RT needs the variant before its own import.
import radio_calibration.simulation  # noqa: F401
import sionna.rt

# isort: on


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
