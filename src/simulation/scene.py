"""Scene load and scene geometry."""

import re
from dataclasses import dataclass

import mitsuba as mi
import numpy as np
import sionna.rt

from src.utils import DATA_DIR

SCENES_DIR = DATA_DIR / "external" / "scenes"


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


def load_rt_scene(name: str) -> sionna.rt.Scene:
    """Load a Sionna RT scene.

    Args:
        name: Name of a scene that Sionna RT supplies (for example `munich`), or the
            name of a file `<name>.xml` in `data/external/scenes`.

    Returns:
        The scene.

    Raises:
        ValueError: If the name has a character that is not a letter, a digit, or `_`,
            or if no scene has this name.
    """
    # The name becomes a part of a file path. This check prevents a path outside SCENES_DIR.
    if not re.fullmatch(r"\w+", name):
        raise ValueError(f"The scene name '{name}' must have only letters, digits, and '_'.")
    builtin = getattr(sionna.rt.scene, name, None)
    if isinstance(builtin, str) and builtin.endswith(".xml"):
        return sionna.rt.load_scene(builtin)
    path = SCENES_DIR / f"{name}.xml"
    if path.is_file():
        return sionna.rt.load_scene(str(path))
    raise ValueError(f"No scene has the name '{name}'.")


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
