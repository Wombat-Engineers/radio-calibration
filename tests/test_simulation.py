"""Tests for radio_calibration.simulation. They use a small scene and CUDA."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from radio_calibration.core import RadioMapGrid, SimulationSettings
from radio_calibration.simulation import (
    HeightMap,
    building_height_map,
    cell_buildings,
    compute_map_inputs,
    compute_path_gain_map,
    is_outdoor,
    load_rt_scene,
    los_map,
    scene_grid,
)
from tests.test_core import make_cell

# In `simple_street_canyon`, the street is along the x axis and buildings are on each side.
STREET = [0.0, 0.0, 1.5]
IN_BUILDING = [-40.0, 30.0, 1.5]

# The sample count is small because the GPU of the development machine has 4 GB of memory.
SIMULATION = SimulationSettings(
    frequency_hz=2.6e9,
    radio_map={
        "samples_per_tx": 100_000,
        "max_depth": 2,
        "diffraction": False,
        "refraction": False,
        "seed": 1,
    },
    radio_map_cell_size_m=10.0,
    radio_map_tx_batch_size=2,
)
# Two cells above the street. They point along the street, one in each direction.
CELLS = [
    make_cell("east", x=-60.0, y=0.0, z=10.0, azimuth_deg=0.0, num_rows=2, num_cols=2),
    make_cell("west", x=60.0, y=0.0, z=10.0, azimuth_deg=180.0, num_rows=2, num_cols=2),
]


@pytest.fixture(scope="module")
def scene():
    return load_rt_scene("simple_street_canyon")


@pytest.fixture(scope="module")
def grid(scene):
    return scene_grid(scene, SIMULATION.radio_map_cell_size_m, 1.5)


def test_load_rt_scene_builtin(scene):
    assert len(scene.objects) > 0


@pytest.mark.parametrize("name", ["nope", "../munich", "sphere"])
def test_load_rt_scene_unknown(name):
    with pytest.raises(ValueError):
        load_rt_scene(name)


def test_load_rt_scene_path():
    import sionna.rt

    scene = load_rt_scene(Path(sionna.rt.scene.simple_street_canyon))

    assert len(scene.objects) > 0


def test_load_rt_scene_missing_file(tmp_path):
    with pytest.raises(ValueError, match="xml"):
        load_rt_scene(tmp_path / "no.xml")


def test_load_rt_scene_bad_suffix(tmp_path):
    path = tmp_path / "scene.txt"
    path.write_text("<scene/>")

    with pytest.raises(ValueError, match="xml"):
        load_rt_scene(path)


def test_load_rt_scene_unmerged(scene):
    unmerged = load_rt_scene("simple_street_canyon", merge_shapes=False)

    # One object for each shape: 6 buildings and the floor.
    assert len(unmerged.objects) == 7
    assert len(unmerged.objects) >= len(scene.objects)


# --- package boundary ---

LAB_MODULES = ("scenario", "utils", "preprocessing", "experiments")


def test_package_does_not_import_lab():
    # A new process: this test process imported other modules already. The list has no
    # matplotlib: sionna.rt imports it.
    names = [*LAB_MODULES, "hydra"]
    code = (
        "import sys, radio_calibration.simulation, radio_calibration.calibration, "
        "radio_calibration.evaluation; "
        f"print([name for name in {names!r} if name in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_simulation_does_not_import_torch():
    code = "import sys, radio_calibration.simulation; print('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_package_has_only_deployment_modules():
    import radio_calibration

    root = Path(radio_calibration.__file__).parent
    subpackages = {path.parent.name for path in root.glob("*/__init__.py")}

    assert subpackages == {"core", "simulation", "calibration", "evaluation"}


# --- height map, outdoor test, building summary, line of sight ---


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


def test_cell_buildings_max_and_fraction():
    heights = np.zeros((4, 4))
    heights[1, 2] = 10.0
    height_map = HeightMap(heights, 0.0, 0.0, 0.0, 5.0)
    grid = RadioMapGrid(0.0, 0.0, 1.5, 20.0, 1, 1)

    max_height, fraction = cell_buildings(height_map, grid, 2.0)

    assert max_height.tolist() == [[10.0]]
    assert fraction.tolist() == [[1 / 16]]


def test_cell_buildings_pads_outside():
    height_map = HeightMap(np.full((2, 2), 10.0), 0.0, 0.0, 0.0, 5.0)
    grid = RadioMapGrid(0.0, 0.0, 1.5, 20.0, 2, 1)

    _, fraction = cell_buildings(height_map, grid, 2.0)

    assert fraction.tolist() == [[4 / 16, 0.0]]


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        (RadioMapGrid(0.0, 0.0, 1.5, 18.0, 1, 1), "multiple"),
        (RadioMapGrid(3.0, 0.0, 1.5, 20.0, 1, 1), "corner"),
    ],
)
def test_cell_buildings_bad_grid(grid, match):
    height_map = HeightMap(np.zeros((4, 4)), 0.0, 0.0, 0.0, 5.0)

    with pytest.raises(ValueError, match=match):
        cell_buildings(height_map, grid, 2.0)


def test_los_map_street(scene, grid):
    los = los_map(scene, CELLS[:1], grid)
    row, column, _ = grid.cell_index(np.array([STREET, IN_BUILDING]))

    assert los.shape == (1, grid.ny, grid.nx)
    assert los[0, row[0], column[0]]
    assert not los[0, row[1], column[1]]


# --- path gain maps ---


def test_scene_grid_covers_scene(scene, grid):
    bbox = scene.mi_scene.bbox()

    assert grid.x_min == pytest.approx(float(bbox.min.x))
    assert grid.x_min + grid.nx * grid.cell_size >= float(bbox.max.x)
    assert grid.y_min + grid.ny * grid.cell_size >= float(bbox.max.y)
    assert grid.z == pytest.approx(float(bbox.min.z) + 1.5)


def test_path_gain_map_shape_and_range(scene, grid):
    path_gain = compute_path_gain_map(scene, CELLS, grid, SIMULATION)
    row, column, _ = grid.cell_index(np.array([STREET]))

    assert path_gain.shape == (2, grid.ny, grid.nx)
    assert ((path_gain >= 0) & (path_gain <= 1)).all()
    assert (path_gain[:, row[0], column[0]] > 0).all()


def test_path_gain_map_batches_equal(scene, grid):
    one = compute_path_gain_map(scene, CELLS, grid, replace(SIMULATION, radio_map_tx_batch_size=1))
    two = compute_path_gain_map(scene, CELLS, grid, SIMULATION)
    covered = (one > 0) & (two > 0)

    assert one.shape == two.shape
    # The same seed traces the same rays for each transmitter, thus the maps agree within the
    # float error of the accumulation.
    np.testing.assert_allclose(10 * np.log10(one[covered]), 10 * np.log10(two[covered]), atol=0.1)


def test_path_gain_map_cleans_scene(scene, grid):
    compute_path_gain_map(scene, CELLS, grid, SIMULATION)

    assert len(scene.transmitters) == 0


def test_path_gain_map_unknown_key(scene, grid):
    settings = replace(SIMULATION, radio_map={**SIMULATION.radio_map, "nope": 1})

    with pytest.raises(TypeError):
        compute_path_gain_map(scene, CELLS, grid, settings)
    assert len(scene.transmitters) == 0


def test_path_gain_map_duplicate_cell(scene, grid):
    with pytest.raises(ValueError, match="unique"):
        compute_path_gain_map(scene, [CELLS[0], CELLS[0]], grid, SIMULATION)


def test_compute_map_inputs(scene, grid):
    height_map = building_height_map(scene, tile_size=5.0)

    inputs = compute_map_inputs(scene, CELLS, grid, height_map, SIMULATION, 2.0)

    assert inputs.num_cells == 2
    assert inputs.los.shape == inputs.path_gain.shape == (2, grid.ny, grid.nx)
    assert inputs.cell_xy.tolist() == [[-60.0, 0.0], [60.0, 0.0]]
    assert inputs.building_fraction.max() > 0
