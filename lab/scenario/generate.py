"""Command that generates the data of a synthetic scenario."""

import dataclasses
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from preprocessing import MapsFile, read_cells, write_cells, write_maps
from radio_calibration.core import Cell
from radio_calibration.simulation import (
    HeightMap,
    building_height_map,
    compute_map_inputs,
    compute_path_gain_map,
    is_outdoor,
    load_rt_scene,
    scene_grid,
)
from scenario.buildings import BuildingSettings, perturb_buildings
from scenario.mdt import sample_mdt
from scenario.rsrp import map_rsrp_fn
from scenario.sites import place_sites
from scenario.traffic import building_density
from utils import CONFIG_DIR, scenario_dirs, scene_source, seed_everything, simulation_settings

log = logging.getLogger(__name__)


def _scene_center(height_map: HeightMap) -> np.ndarray:
    ny, nx = height_map.heights.shape
    return np.array(
        [
            height_map.x_min + nx * height_map.tile_size / 2,
            height_map.y_min + ny * height_map.tile_size / 2,
        ]
    )


def _cells(
    scenario: DictConfig, height_map: HeightMap, center_xy: np.ndarray, azimuth_offset: float
) -> list[Cell]:
    """Place the sites of the scenario with a center and a turn of all sector azimuths."""
    sites = scenario.sites
    return place_sites(
        height_map,
        scenario.grid.min_building_height_m,
        (float(center_xy[0]), float(center_xy[1])),
        sites.isd_m,
        sites.mast_height_m,
        [(azimuth + azimuth_offset) % 360.0 for azimuth in sites.sector_azimuths_deg],
        sites.downtilt_deg,
        scenario.antenna.power_rs,
        scenario.antenna.transmitter,
    )


def _layouts(
    scenario: DictConfig, seed: int, height_map: HeightMap, center_xy: np.ndarray
) -> list[list[Cell]]:
    """Make the random site layouts of the pretraining, in the order (scene, layout).

    Each layout has its own random generator, thus a layout does not change when the number
    of layouts changes. The cell names start with `layout<scene>_<layout>_`, thus all names
    are unique.
    """
    pretraining = scenario.pretraining
    layouts = []
    for scene_index in range(pretraining.num_scenes):
        for layout_index in range(pretraining.layouts_per_scene):
            rng = np.random.default_rng((seed, scene_index + 1, layout_index))
            shift = rng.uniform(-pretraining.center_shift_m, pretraining.center_shift_m, size=2)
            azimuth_offset = rng.uniform(0.0, pretraining.azimuth_shift_deg)
            cells = _cells(scenario, height_map, center_xy + shift, azimuth_offset)
            prefix = f"layout{scene_index}_{layout_index}_"
            layouts.append([dataclasses.replace(c, name=prefix + c.name) for c in cells])
    return layouts


def _solver_calls(num_cells: int, batch_size: int) -> int:
    return math.ceil(num_cells / batch_size)


def generate(cfg: DictConfig, output_dir: Path) -> None:
    """Generate the data of a synthetic scenario and write it to a directory.

    The function writes `cells.csv`, `mdt.csv`, and `maps.npz`. It replaces files that exist.
    The steps are:

    1. Load the original scene, and place the cells of the scenario.
    2. Calculate the input maps on the original scene for the cells of the scenario and for
       all pretraining layouts, in one group of solver calls.
    3. Change the buildings at random (the true world). Calculate the true path gain map of
       the cells. Sample the MDT records from the true map.
    4. For each changed pretraining scene, calculate the label maps of its layouts.

    Args:
        cfg: The configuration from `lab/configs/config.yaml`. The function uses `seed`,
            `scenario`, and `simulation`.
        output_dir: Directory for the files. The function creates it if necessary.

    Raises:
        ValueError: If the scenario is not synthetic, or if the configuration has a value
            that is not permitted.
        RuntimeError: If some MDT records have no valid position.
    """
    started = time.perf_counter()
    scenario = cfg.scenario
    if not scenario.synthetic:
        raise ValueError("The generator makes only synthetic scenarios (scenario.synthetic).")
    seed_everything(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation = simulation_settings(cfg)
    buildings = BuildingSettings.from_config(scenario.buildings)
    source = scene_source(cfg)
    tile = scenario.grid.tile_size_m
    min_height = scenario.grid.min_building_height_m
    batch = simulation.radio_map_tx_batch_size
    calls = 0

    original = load_rt_scene(source)
    height_map = building_height_map(original, tile)
    grid = scene_grid(original, simulation.radio_map_cell_size_m, scenario.ue_height_m)
    center_xy = _scene_center(height_map) + np.asarray(scenario.sites.center_offset_m)
    # Read the file again: the maps then come from the same data as with a real network file.
    cells_path = output_dir / "cells.csv"
    write_cells(_cells(scenario, height_map, center_xy, 0.0), cells_path)
    cells = read_cells(cells_path)
    layouts = _layouts(scenario, cfg.seed, height_map, center_xy)

    step = time.perf_counter()
    all_cells = cells + [cell for layout in layouts for cell in layout]
    inputs = compute_map_inputs(original, all_cells, grid, height_map, simulation, min_height)
    calls += _solver_calls(len(all_cells), batch)
    log.info("Original scene: %d cells in %.1f s", len(all_cells), time.perf_counter() - step)
    starts = np.cumsum([len(cells)] + [len(layout) for layout in layouts])
    target = inputs.subset(np.arange(starts[0]))
    variants = [
        inputs.subset(np.arange(start - len(layout), start))
        for start, layout in zip(starts[1:], layouts, strict=True)
    ]

    step = time.perf_counter()
    true_scene = load_rt_scene(source, merge_shapes=False)
    changes = perturb_buildings(true_scene, np.random.default_rng((cfg.seed, 0)), buildings)
    true_path_gain = compute_path_gain_map(true_scene, cells, grid, simulation)
    calls += _solver_calls(len(cells), batch)
    log.info(
        "True world: %d buildings moved, %d resized, %d removed; map in %.1f s",
        int(((changes["dx_m"] != 0) | (changes["dy_m"] != 0)).sum()),
        int((changes["height_factor"] != 1).sum()),
        int(changes["deleted"].sum()),
        time.perf_counter() - step,
    )
    true_height_map = building_height_map(true_scene, tile)
    mdt = sample_mdt(
        np.random.default_rng(cfg.seed),
        rsrp_fn=map_rsrp_fn(true_path_gain, grid, [cell.power_rs_dbm for cell in cells]),
        outdoor_fn=lambda positions: is_outdoor(true_scene, positions),
        cells=cells,
        height_map=true_height_map,
        min_building_height_m=min_height,
        density=building_density(true_height_map, min_height, scenario.traffic.density_radius_m),
        traffic=scenario.traffic,
        ue_height_m=scenario.ue_height_m,
        measurement=scenario.measurement,
    )
    mdt.to_csv(output_dir / "mdt.csv", index=False)

    labels, scene_indices = [], []
    per_scene = scenario.pretraining.layouts_per_scene
    for scene_index in range(scenario.pretraining.num_scenes):
        step = time.perf_counter()
        changed = load_rt_scene(source, merge_shapes=False)
        perturb_buildings(changed, np.random.default_rng((cfg.seed, scene_index + 1)), buildings)
        scene_layouts = layouts[scene_index * per_scene : (scene_index + 1) * per_scene]
        scene_cells = [cell for layout in scene_layouts for cell in layout]
        label = compute_path_gain_map(changed, scene_cells, grid, simulation)
        calls += _solver_calls(len(scene_cells), batch)
        labels += _split(label, [len(layout) for layout in scene_layouts])
        scene_indices += [scene_index] * len(scene_layouts)
        log.info("Pretraining scene %d: %.1f s", scene_index, time.perf_counter() - step)

    write_maps(
        output_dir / "maps.npz",
        MapsFile(target, true_path_gain, variants, labels, np.array(scene_indices)),
    )
    log.info(
        "Total: %d MDT records, %d radio-map solver calls, %.1f s",
        len(mdt),
        calls,
        time.perf_counter() - started,
    )


def _split(array: np.ndarray, sizes: Sequence[int]) -> list[np.ndarray]:
    return np.split(array, np.cumsum(sizes)[:-1])


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate the data of the selected scenario in `lab/data/<scenario>/raw`."""
    generate(cfg, scenario_dirs(cfg.scenario.name).raw)
