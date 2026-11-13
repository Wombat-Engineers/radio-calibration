"""Tests for src.simulation. They use a small scene and CUDA."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.simulation import building_height_map, compute_rsrp, is_outdoor, load_rt_scene
from tests.test_core import make_cell

# In `simple_street_canyon`, the street is along the x axis and buildings are on each side.
STREET = [0.0, 0.0, 1.5]
IN_BUILDING = [-40.0, 30.0, 1.5]

# The sample count is small because the GPU of the development machine has 4 GB of memory.
SIMULATION = OmegaConf.create(
    {
        "radio": {"frequency_hz": 2.6e9, "subcarrier_spacing_hz": 30e3},
        "receiver": {
            "num_rows": 1,
            "num_cols": 1,
            "vertical_spacing": 0.5,
            "horizontal_spacing": 0.5,
            "pattern": "dipole",
            "polarization": "V",
        },
        "solver": {
            "samples_per_src": 10_000,
            "max_depth": 2,
            "diffraction": False,
            "refraction": False,
        },
        "rx_batch_size": 4,
    }
)
# Two cells above the street. They point along the street, one in each direction.
CELLS = [
    make_cell("east", x=-60.0, y=0.0, z=10.0, azimuth_deg=0.0, num_rows=2, num_cols=2),
    make_cell("west", x=60.0, y=0.0, z=10.0, azimuth_deg=180.0, num_rows=2, num_cols=2),
]


@pytest.fixture(scope="module")
def scene():
    return load_rt_scene("simple_street_canyon")


def test_load_rt_scene_builtin(scene):
    assert len(scene.objects) > 0


@pytest.mark.parametrize("name", ["nope", "../munich", "sphere"])
def test_load_rt_scene_unknown(name):
    with pytest.raises(ValueError):
        load_rt_scene(name)


def test_height_map_street_canyon(scene):
    height_map = building_height_map(scene, tile_size=5.0)

    assert height_map.heights.ndim == 2
    assert (height_map.heights > 10.0).any()
    assert (height_map.heights < 0.5).any()
    assert (height_map.heights >= 0.0).all()


def test_height_map_bad_tile_size(scene):
    with pytest.raises(ValueError):
        building_height_map(scene, tile_size=0.0)


def test_is_outdoor_street(scene):
    assert is_outdoor(scene, np.array([STREET])).tolist() == [True]


def test_is_outdoor_in_building(scene):
    outside_scene = [1000.0, 1000.0, 1.5]
    assert is_outdoor(scene, np.array([IN_BUILDING, outside_scene])).tolist() == [False, False]


def test_compute_rsrp_shape(scene):
    positions = np.array([[x, 0.0, 1.5] for x in (-30.0, -10.0, 10.0, 30.0)])

    assert compute_rsrp(scene, CELLS, positions, SIMULATION).shape == (4, 2)
    assert len(scene.transmitters) == 0
    assert len(scene.receivers) == 0


def test_compute_rsrp_los_finite(scene):
    rsrp = compute_rsrp(scene, CELLS, np.array([STREET]), SIMULATION)

    assert np.isfinite(rsrp).all()
    # A path has a loss, and the gain of the 2 x 2 array with the pattern is below 30 dB.
    assert (rsrp < CELLS[0].power_rs_dbm + 30.0).all()


def test_compute_rsrp_indoor_nan(scene):
    rsrp = compute_rsrp(scene, CELLS, np.array([IN_BUILDING]), SIMULATION)

    assert np.isnan(rsrp).all()


def test_compute_rsrp_batches_equal(scene):
    positions = np.array([[x, 0.0, 1.5] for x in (-30.0, -10.0, 10.0, 30.0)])
    one_by_one = OmegaConf.merge(SIMULATION, {"rx_batch_size": 1})

    np.testing.assert_allclose(
        compute_rsrp(scene, CELLS, positions, one_by_one),
        compute_rsrp(scene, CELLS, positions, SIMULATION),
        atol=1.0,
    )


def test_compute_rsrp_mixed_arrays(scene):
    cells = [CELLS[0], make_cell("other", num_rows=4, num_cols=2)]

    with pytest.raises(ValueError, match="same array"):
        compute_rsrp(scene, cells, np.array([STREET]), SIMULATION)
    assert len(scene.transmitters) == 0


def test_compute_rsrp_bad_positions(scene):
    with pytest.raises(ValueError, match="shape"):
        compute_rsrp(scene, CELLS, np.zeros((4, 2)), SIMULATION)


def test_compute_rsrp_refraction_reaches_indoor(scene):
    with_refraction = OmegaConf.merge(SIMULATION, {"solver": {"refraction": True}})

    rsrp = compute_rsrp(scene, CELLS, np.array([IN_BUILDING]), with_refraction)

    assert np.isfinite(rsrp).any()


def test_compute_rsrp_unknown_solver_key(scene):
    unknown = OmegaConf.merge(SIMULATION, {"solver": {"nope": 1}})

    with pytest.raises(TypeError):
        compute_rsrp(scene, CELLS, np.array([STREET]), unknown)
    assert len(scene.transmitters) == 0
    assert len(scene.receivers) == 0
