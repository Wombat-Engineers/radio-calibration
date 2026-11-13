"""Sionna RT functions: scene load, scene geometry, and RSRP calculation."""

import mitsuba as mi

# Sionna RT needs a mono, polarized variant, and the variant must be set before the
# import of `sionna.rt`. No fallback to the LLVM variant: a CUDA failure must be visible.
mi.set_variant("cuda_ad_mono_polarized")

from src.simulation.paths import compute_rsrp  # noqa: E402
from src.simulation.scene import (  # noqa: E402
    HeightMap,
    building_height_map,
    is_outdoor,
    load_rt_scene,
)

__all__ = ["HeightMap", "building_height_map", "compute_rsrp", "is_outdoor", "load_rt_scene"]
