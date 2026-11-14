"""Sionna RT functions: scene load, scene geometry, RSRP calculation, and neural material."""

import mitsuba as mi

# Sionna RT needs a mono, polarized variant, and the variant must be set before the
# import of `sionna.rt`. No fallback to the LLVM variant: a CUDA failure must be visible.
mi.set_variant("cuda_ad_mono_polarized")

from src.simulation.neural_material import (  # noqa: E402
    MaterialValues,
    NeuralRadioMaterial,
    apply_neural_material,
    assign_material,
    make_material_network,
)
from src.simulation.paths import compute_rsrp, compute_rsrp_mw  # noqa: E402
from src.simulation.scene import (  # noqa: E402
    HeightMap,
    building_height_map,
    is_outdoor,
    load_rt_scene,
)

__all__ = [
    "HeightMap",
    "MaterialValues",
    "NeuralRadioMaterial",
    "apply_neural_material",
    "assign_material",
    "building_height_map",
    "compute_rsrp",
    "compute_rsrp_mw",
    "is_outdoor",
    "load_rt_scene",
    "make_material_network",
]
