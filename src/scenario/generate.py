"""Command that generates the cells and the MDT records."""

import logging
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.core import read_cells, write_cells
from src.scenario.mdt import sample_mdt
from src.scenario.sites import place_sites
from src.scenario.traffic import building_density
from src.simulation import building_height_map, compute_rsrp, is_outdoor, load_rt_scene
from src.utils import DATA_DIR, PROJECT_ROOT, seed_everything

log = logging.getLogger(__name__)


def generate(cfg: DictConfig, output_dir: Path) -> None:
    """Generate the scenario data and write it to a directory.

    The function writes `cells.csv`, `mdt.csv`, and `scenario_config.yaml`. It replaces
    files that exist. It logs the time of the RSRP calculation and the total time.

    Args:
        cfg: The configuration from `configs/scenario.yaml`, with the content of
            `configs/simulation.yaml` in the key `simulation`.
        output_dir: Directory for the files. The function creates it if necessary.

    Raises:
        ValueError: If the configuration has a value that is not permitted.
        RuntimeError: If some MDT records have no valid position.
    """
    started = time.perf_counter()
    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = load_rt_scene(cfg.simulation.scene.name)
    height_map = building_height_map(scene, cfg.grid.tile_size_m)
    ny, nx = height_map.heights.shape
    center_xy = (
        height_map.x_min + nx * height_map.tile_size / 2 + cfg.sites.center_offset_m[0],
        height_map.y_min + ny * height_map.tile_size / 2 + cfg.sites.center_offset_m[1],
    )
    cells = place_sites(
        height_map,
        cfg.grid.min_building_height_m,
        center_xy,
        cfg.sites.isd_m,
        cfg.sites.mast_height_m,
        cfg.sites.sector_azimuths_deg,
        cfg.sites.downtilt_deg,
        cfg.antenna.power_rs,
        cfg.antenna.transmitter,
    )
    # Read the file again: the RSRP then comes from the same data as with a real network file.
    cells_path = output_dir / "cells.csv"
    write_cells(cells, cells_path)
    cells = read_cells(cells_path)

    rsrp_calls = 0
    rsrp_positions = 0
    rsrp_seconds = 0.0

    def timed_rsrp(positions: np.ndarray) -> np.ndarray:
        nonlocal rsrp_calls, rsrp_positions, rsrp_seconds
        call_started = time.perf_counter()
        rsrp = compute_rsrp(scene, cells, positions, cfg.simulation)
        rsrp_seconds += time.perf_counter() - call_started
        rsrp_calls += 1
        rsrp_positions += len(positions)
        return rsrp

    mdt = sample_mdt(
        rng,
        rsrp_fn=timed_rsrp,
        outdoor_fn=lambda positions: is_outdoor(scene, positions),
        cells=cells,
        height_map=height_map,
        min_building_height_m=cfg.grid.min_building_height_m,
        density=building_density(
            height_map, cfg.grid.min_building_height_m, cfg.traffic.density_radius_m
        ),
        traffic=cfg.traffic,
        ue_height=cfg.ue_height,
        measurement=cfg.measurement,
    )
    mdt.to_csv(output_dir / "mdt.csv", index=False)
    OmegaConf.save(cfg, output_dir / "scenario_config.yaml", resolve=True)

    log.info(
        "RSRP: %d positions in %d calls, %.1f s, %.0f ms for each position",
        rsrp_positions,
        rsrp_calls,
        rsrp_seconds,
        1000 * rsrp_seconds / max(rsrp_positions, 1),
    )
    log.info("Total: %d MDT records in %.1f s", len(mdt), time.perf_counter() - started)


@hydra.main(config_path=str(PROJECT_ROOT / "configs"), config_name="scenario", version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate the scenario data in `data/raw`."""
    generate(cfg, DATA_DIR / "raw")
