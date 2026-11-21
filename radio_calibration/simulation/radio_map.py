"""Path gain maps with the Sionna RT radio-map solver."""

import math
from collections.abc import Sequence

import numpy as np
import sionna.rt

from radio_calibration.core import Cell, MapInputs, RadioMapGrid, SimulationSettings, check_cells
from radio_calibration.simulation.cells import add_cells, remove_cells
from radio_calibration.simulation.height_map import HeightMap, cell_buildings, los_map


def scene_grid(scene: sionna.rt.Scene, cell_size: float, height: float) -> RadioMapGrid:
    """Make the grid that covers the horizontal extent of a scene.

    Args:
        scene: The scene.
        cell_size: Length of one side of a grid cell, in meters.
        height: Height of the measurement plane above the lowest point of the scene, in
            meters.

    Returns:
        The grid. It starts at the smallest x and the smallest y of the scene. It has enough
        grid cells to cover the scene.
    """
    bbox = scene.mi_scene.bbox()
    x_min, y_min = float(bbox.min.x), float(bbox.min.y)
    nx = max(math.ceil((float(bbox.max.x) - x_min) / cell_size), 1)
    ny = max(math.ceil((float(bbox.max.y) - y_min) / cell_size), 1)
    return RadioMapGrid(x_min, y_min, float(bbox.min.z) + height, cell_size, nx, ny)


def compute_path_gain_map(
    scene: sionna.rt.Scene,
    cells: Sequence[Cell],
    grid: RadioMapGrid,
    simulation: SimulationSettings,
) -> np.ndarray:
    """Calculate the path gain map of each cell with the Sionna RT `RadioMapSolver`.

    The path gain includes the gain of the transmitter array with the default precoding of
    Sionna RT. The receiver of a radio map is isotropic. The function adds the transmitters
    and removes them before it returns. It calls the solver one time for each batch of
    `simulation.radio_map_tx_batch_size` cells.

    Args:
        scene: The scene. It must have no transmitter.
        cells: The cells. All cells must have the same array configuration.
        grid: The grid of the map.
        simulation: The settings of the simulation. `simulation.radio_map` goes to the
            solver call with no change.

    Returns:
        The linear path gain, with shape `[len(cells), ny, nx]`. A value is 0 where no ray
        reaches the grid cell.

    Raises:
        ValueError: If a cell is not valid, or if the cells do not have the same array
            configuration.
        TypeError: If `simulation.radio_map` has a key that is not an argument of the solver.
    """
    check_cells(cells)
    solver = sionna.rt.RadioMapSolver()
    size = [grid.nx * grid.cell_size, grid.ny * grid.cell_size]
    center = [grid.x_min + size[0] / 2, grid.y_min + size[1] / 2, grid.z]
    batch_size = simulation.radio_map_tx_batch_size
    maps = []
    for start in range(0, len(cells), batch_size):
        batch = cells[start : start + batch_size]
        add_cells(scene, batch, simulation.frequency_hz)
        try:
            radio_map = solver(
                scene,
                center=center,
                orientation=[0.0, 0.0, 0.0],
                size=size,
                cell_size=[grid.cell_size, grid.cell_size],
                **simulation.radio_map,
            )
            maps.append(radio_map.path_gain.numpy().astype(float))
        finally:
            remove_cells(scene, batch)
    return np.concatenate(maps)


def compute_map_inputs(
    scene: sionna.rt.Scene,
    cells: Sequence[Cell],
    grid: RadioMapGrid,
    height_map: HeightMap,
    simulation: SimulationSettings,
    min_building_height: float,
) -> MapInputs:
    """Calculate the input maps of the U-Net for the cells of one scene.

    The function calls `compute_path_gain_map` and `los_map` on the scene, and
    `cell_buildings` on the height map of the same scene.

    Args:
        scene: The scene. It must have no transmitter.
        cells: The cells.
        grid: The grid of the maps. It must start at the corner of the height map.
        height_map: The height map of the scene.
        simulation: The settings of the simulation.
        min_building_height: A height-map tile above this height, in meters, is a building.

    Returns:
        The input maps.

    Raises:
        ValueError: As for `compute_path_gain_map` and `cell_buildings`.
        TypeError: As for `compute_path_gain_map`.
    """
    path_gain = compute_path_gain_map(scene, cells, grid, simulation)
    max_height, fraction = cell_buildings(height_map, grid, min_building_height)
    return MapInputs(
        grid=grid,
        path_gain=path_gain,
        los=los_map(scene, cells, grid),
        max_height=max_height,
        building_fraction=fraction,
        cell_xy=np.array([[cell.x, cell.y] for cell in cells], dtype=float),
    )
