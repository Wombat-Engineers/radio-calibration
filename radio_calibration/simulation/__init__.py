"""Sionna RT functions: scene load, path gain maps, and the geometry of the scene."""

import mitsuba as mi

# Sionna RT needs a mono, polarized variant, and the variant must be set before the
# import of `sionna.rt`. No fallback to the LLVM variant: a CUDA failure must be visible.
mi.set_variant("cuda_ad_mono_polarized")

from radio_calibration.simulation.height_map import (  # noqa: E402
    HeightMap,
    building_height_map,
    cell_buildings,
    is_outdoor,
    los_map,
)
from radio_calibration.simulation.radio_map import (  # noqa: E402
    compute_map_inputs,
    compute_path_gain_map,
    scene_grid,
)
from radio_calibration.simulation.scene import load_rt_scene  # noqa: E402

__all__ = [
    "HeightMap",
    "building_height_map",
    "cell_buildings",
    "compute_map_inputs",
    "compute_path_gain_map",
    "is_outdoor",
    "load_rt_scene",
    "los_map",
    "scene_grid",
]
