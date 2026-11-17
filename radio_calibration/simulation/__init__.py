"""Sionna RT functions: scene load, RSRP calculation, and neural material."""

import mitsuba as mi

# Sionna RT needs a mono, polarized variant, and the variant must be set before the
# import of `sionna.rt`. No fallback to the LLVM variant: a CUDA failure must be visible.
mi.set_variant("cuda_ad_mono_polarized")

from radio_calibration.simulation.neural_material import (  # noqa: E402
    MaterialValues,
    NeuralRadioMaterial,
    apply_neural_material,
    assign_material,
    check_scale,
    make_material_network,
)
from radio_calibration.simulation.paths import compute_rsrp_dbm, compute_rsrp_mw  # noqa: E402
from radio_calibration.simulation.scene import load_rt_scene  # noqa: E402

__all__ = [
    "MaterialValues",
    "NeuralRadioMaterial",
    "apply_neural_material",
    "assign_material",
    "check_scale",
    "compute_rsrp_dbm",
    "compute_rsrp_mw",
    "load_rt_scene",
    "make_material_network",
]
