"""Tests for src.simulation. They use a small scene and CUDA."""

import drjit as dr
import numpy as np
import pytest
from omegaconf import OmegaConf

from src.simulation import (
    NeuralRadioMaterial,
    apply_neural_material,
    building_height_map,
    compute_rsrp,
    compute_rsrp_mw,
    is_outdoor,
    load_rt_scene,
    make_material_network,
)
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
        "deterministic": True,
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


def test_compute_rsrp_deterministic_repeats(scene):
    positions = np.array([[x, 0.0, 1.5] for x in (-30.0, -10.0, 10.0, 30.0)])

    first = compute_rsrp(scene, CELLS, positions, SIMULATION)
    second = compute_rsrp(scene, CELLS, positions, SIMULATION)

    np.testing.assert_array_equal(first, second)


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


# --- compute_rsrp_mw ---


def test_compute_rsrp_mw_matches_compute_rsrp(scene):
    positions = np.array([[x, 0.0, 1.5] for x in (-30.0, -10.0, 10.0, 30.0)])

    power_mw = compute_rsrp_mw(scene, CELLS, positions, SIMULATION).numpy()

    np.testing.assert_allclose(
        10.0 * np.log10(power_mw), compute_rsrp(scene, CELLS, positions, SIMULATION), atol=1.0
    )


def test_compute_rsrp_mw_no_path_is_zero(scene):
    power_mw = compute_rsrp_mw(scene, CELLS, np.array([IN_BUILDING]), SIMULATION).numpy()

    assert power_mw.shape == (1, 2)
    assert (power_mw == 0).all()


def test_compute_rsrp_mw_cleans_scene(scene):
    compute_rsrp_mw(scene, CELLS, np.array([STREET]), SIMULATION)

    assert len(scene.transmitters) == 0
    assert len(scene.receivers) == 0


@pytest.mark.parametrize("positions", [np.zeros((4, 2)), np.zeros((0, 3))])
def test_compute_rsrp_mw_bad_positions(scene, positions):
    with pytest.raises(ValueError, match="positions"):
        compute_rsrp_mw(scene, CELLS, positions, SIMULATION)


# --- NeuralRadioMaterial ---

# A small network: the GPU of the development machine has 4 GB of memory.
NETWORK = {"encoding_octaves": 2, "hidden_layers": 1, "hidden_units": 8}
POINTS = np.array([[0.0, 0.0, 0.0], [20.0, 5.0, 3.0], [-40.0, 30.0, 10.0]])


def weights_of(network) -> np.ndarray:
    return np.array(network["weights"], dtype=np.float32)


@pytest.fixture
def neural_scene():
    """A new scene for each test: the neural material replaces the materials of the scene."""
    scene = load_rt_scene("simple_street_canyon")
    material = apply_neural_material(scene, make_material_network(**NETWORK, seed=1))
    return scene, material


def test_make_material_network_seed():
    first = weights_of(make_material_network(**NETWORK, seed=1))

    np.testing.assert_array_equal(first, weights_of(make_material_network(**NETWORK, seed=1)))
    assert (first != weights_of(make_material_network(**NETWORK, seed=2))).any()


@pytest.mark.parametrize("name", list(NETWORK))
def test_make_material_network_bad_size(name):
    with pytest.raises(ValueError, match=name):
        make_material_network(**(NETWORK | {name: 0}), seed=1)


def test_values_at_ranges(neural_scene):
    _, material = neural_scene
    values = material.values_at(POINTS)

    assert (values.relative_permittivity.numpy() > 1).all()
    assert (values.conductivity.numpy() > 0).all()
    assert (values.thickness.numpy() > 0).all()
    for coefficient in (values.scattering_coefficient, values.xpd_coefficient):
        assert ((coefficient.numpy() > 0) & (coefficient.numpy() < 1)).all()


def test_values_at_depends_on_position(neural_scene):
    _, material = neural_scene
    conductivity = material.values_at(POINTS).conductivity.numpy()

    assert conductivity.shape == (len(POINTS),)
    assert len(np.unique(conductivity)) == len(POINTS)


def test_apply_neural_material_all_objects(neural_scene):
    scene, material = neural_scene

    assert all(obj.radio_material is material for obj in scene.objects.values())


def test_apply_neural_material_survives_frequency(neural_scene):
    scene, material = neural_scene
    before = material.values_at(POINTS).conductivity.numpy()

    scene.frequency = SIMULATION.radio.frequency_hz

    np.testing.assert_array_equal(material.values_at(POINTS).conductivity.numpy(), before)


def test_compute_rsrp_with_neural_material(neural_scene):
    scene, _ = neural_scene
    rsrp = compute_rsrp(scene, CELLS, np.array([STREET]), SIMULATION)

    assert rsrp.shape == (1, 2)
    assert np.isfinite(rsrp).all()


def test_compute_rsrp_mw_gradient(neural_scene):
    scene, material = neural_scene
    # No weight decay: only a gradient can change the weights.
    optimizer = dr.opt.AdamW(lr=1e-2, weight_decay=0.0)
    optimizer.update(material.network)
    material.network.update(optimizer)
    before = weights_of(material.network)

    power_mw = compute_rsrp_mw(scene, CELLS, np.array([STREET]), SIMULATION)
    dr.backward(dr.sum(power_mw, axis=None))
    optimizer.step()
    material.network.update(optimizer)

    assert (weights_of(material.network) != before).any()


def test_state_round_trip(neural_scene):
    _, material = neural_scene
    copy = NeuralRadioMaterial.from_state(material.state(), "copy")

    for value, expected in zip(copy.values_at(POINTS), material.values_at(POINTS), strict=True):
        np.testing.assert_array_equal(value.numpy(), expected.numpy())


def test_from_state_missing_key(neural_scene):
    _, material = neural_scene
    state = material.state()
    del state["weights"]

    with pytest.raises(ValueError, match="weights"):
        NeuralRadioMaterial.from_state(state, "copy")


def test_from_state_wrong_weight_count(neural_scene):
    _, material = neural_scene
    state = material.state()
    state["weights"] = state["weights"][:-1]

    with pytest.raises(ValueError, match="weights"):
        NeuralRadioMaterial.from_state(state, "copy")
