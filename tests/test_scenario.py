"""Tests for scenario."""

import logging

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import scenario.generate as generate_module
from preprocessing import read_cells, read_maps
from radio_calibration.core import RadioMapGrid
from radio_calibration.simulation import HeightMap, load_rt_scene
from scenario.buildings import BuildingSettings, building_groups, perturb_buildings
from scenario.generate import generate
from scenario.mdt import sample_mdt
from scenario.rsrp import map_rsrp_fn
from scenario.sites import place_sites
from scenario.traffic import building_density, slot_counts, spatial_weights
from utils import load_config

MIN_BUILDING_HEIGHT = 2.0
FLAT = [1.0] * 24
DAY = [1.0 if 8 <= hour < 18 else 0.0 for hour in range(24)]
NIGHT = [0.0 if 8 <= hour < 18 else 1.0 for hour in range(24)]
TRANSMITTER = OmegaConf.create(
    {
        "num_rows": 8,
        "num_cols": 8,
        "vertical_spacing": 0.5,
        "horizontal_spacing": 0.5,
        "pattern": "tr38901",
        "polarization": "cross",
    }
)


def make_height_map(heights: np.ndarray, tile_size: float = 10.0) -> HeightMap:
    """Make a height map with its center at the origin and the ground at z = 0."""
    ny, nx = heights.shape
    return HeightMap(heights, -nx * tile_size / 2, -ny * tile_size / 2, 0.0, tile_size)


def make_sites(height_map: HeightMap, isd_m: float = 300.0):
    return place_sites(
        height_map, MIN_BUILDING_HEIGHT, (0.0, 0.0), isd_m, 25.0, [0.0, 120.0, 240.0], 5.0, 4.85,
        TRANSMITTER,
    )  # fmt: skip


# --- buildings (CUDA, small scene) ---

# In `simple_street_canyon`, each building is one shape `building_<i>`, and the floor is
# `floor`.
CANYON_BUILDINGS = BuildingSettings(
    ground_objects=("floor",),
    move_fraction=1.0,
    max_shift_m=5.0,
    height_fraction=1.0,
    height_factor=(0.5, 0.5),
    delete_fraction=0.0,
)


def test_building_groups_names():
    names = ["Heilig_Geist-itu_marble", "Heilig_Geist-itu_metal", "Alter_Hof", "ground"]

    groups = building_groups(names, ["ground"])

    assert groups == {
        "Alter_Hof": ["Alter_Hof"],
        "Heilig_Geist": ["Heilig_Geist-itu_marble", "Heilig_Geist-itu_metal"],
    }


@pytest.mark.parametrize(
    "change", [{"move_fraction": 1.5}, {"max_shift_m": -1.0}, {"height_factor": (0.0, 1.0)}]
)
def test_building_settings_bad_value(change):
    values = {
        "ground_objects": (),
        "move_fraction": 0.5,
        "max_shift_m": 1.0,
        "height_fraction": 0.5,
        "height_factor": (0.5, 1.5),
        "delete_fraction": 0.1,
    }
    with pytest.raises(ValueError):
        BuildingSettings(**(values | change))


def bbox_of(scene, name):
    box = scene.objects[name].mi_mesh.bbox()
    return np.array([*box.min, *box.max], dtype=float)


def perturbed(seed: int, settings: BuildingSettings = CANYON_BUILDINGS):
    scene = load_rt_scene("simple_street_canyon", merge_shapes=False)
    changes = perturb_buildings(scene, np.random.default_rng(seed), settings)
    return scene, changes


def test_perturb_repeatable_seed():
    _, first = perturbed(3)
    _, second = perturbed(3)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 6


def test_perturb_delete_fraction():
    scene, changes = perturbed(0, BuildingSettings(("floor",), 0.0, 0.0, 0.0, (1.0, 1.0), 1.0))

    assert changes["deleted"].all()
    assert list(scene.objects) == ["floor"]


def test_perturb_height_keeps_base():
    before = bbox_of(load_rt_scene("simple_street_canyon", merge_shapes=False), "building_1")

    scene, changes = perturbed(0)
    after = bbox_of(scene, "building_1")

    assert changes.set_index("building").loc["building_1", "height_factor"] == 0.5
    assert after[2] == pytest.approx(before[2], abs=1e-3)
    assert after[5] - after[2] == pytest.approx(0.5 * (before[5] - before[2]), rel=1e-4)


def test_perturb_move_shifts_xy():
    before = bbox_of(load_rt_scene("simple_street_canyon", merge_shapes=False), "building_2")

    scene, changes = perturbed(0)
    after = bbox_of(scene, "building_2")

    row = changes.set_index("building").loc["building_2"]
    np.testing.assert_allclose(
        after[[0, 1]] - before[[0, 1]], [row["dx_m"], row["dy_m"]], atol=1e-3
    )


# --- true RSRP from the map ---

GRID = RadioMapGrid(0.0, 0.0, 1.5, 10.0, 2, 1)


def test_map_rsrp_value():
    rsrp_fn = map_rsrp_fn(np.array([[[1e-10, 0.0]]]), GRID, [4.85])

    rsrp = rsrp_fn(np.array([[5.0, 5.0, 1.5], [15.0, 5.0, 1.5], [-5.0, 5.0, 1.5]]))

    # 4.85 dBm + 10 log10(1e-10) = -95.15 dBm. A path gain of 0 and a position outside the
    # grid give NaN.
    assert rsrp[0, 0] == pytest.approx(-95.15)
    assert np.isnan(rsrp[1:, 0]).all()


def test_map_rsrp_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        map_rsrp_fn(np.zeros((2, 1, 2)), GRID, [4.85])


# --- sites ---


def test_place_sites_count():
    cells = make_sites(make_height_map(np.zeros((100, 100))))

    assert len(cells) == 9
    assert len({cell.name for cell in cells}) == 9
    assert [cell.azimuth_deg for cell in cells[:3]] == [0.0, 120.0, 240.0]
    assert cells[0].num_rows == 8 and cells[0].pattern == "tr38901"


def test_place_sites_triangle():
    cells = make_sites(make_height_map(np.zeros((100, 100))))
    corners = np.array([[cell.x, cell.y] for cell in cells[::3]])
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 3]) for i in range(3)]

    np.testing.assert_allclose(sides, 300.0)
    assert {cell.z for cell in cells} == {25.0}


def test_place_sites_moves_off_building():
    # The top corner of the triangle is at (0, 173.2). Columns 45 to 54 are a building.
    heights = np.zeros((100, 100))
    heights[:, 45:55] = 30.0
    cells = make_sites(make_height_map(heights))

    # The nearest outdoor tiles have their centers 55 m from the y axis.
    assert abs(cells[0].x) == pytest.approx(55.0)
    assert cells[0].y == pytest.approx(175.0)
    assert cells[0].z == 25.0


def test_place_sites_outside_scene():
    with pytest.raises(ValueError, match="outside"):
        make_sites(make_height_map(np.zeros((10, 10))), isd_m=500.0)


# --- traffic ---


def test_building_density_range():
    full = building_density(make_height_map(np.full((20, 20), 30.0)), MIN_BUILDING_HEIGHT, 20.0)
    empty = building_density(make_height_map(np.zeros((20, 20))), MIN_BUILDING_HEIGHT, 20.0)

    assert full[10, 10] == pytest.approx(1.0)
    assert (empty == 0.0).all()


def test_spatial_weights_sum_and_mask():
    density = np.array([[0.2, 0.8], [0.5, 0.9]])
    outdoor = np.array([[True, True], [True, False]])
    weights = spatial_weights(density, outdoor, 12, FLAT, FLAT, 2.0)

    assert weights.sum() == pytest.approx(1.0)
    assert weights[1, 1] == 0.0
    assert weights[0, 1] > weights[1, 0] > weights[0, 0]


def test_spatial_weights_time_shift():
    density = np.array([[0.1, 0.2, 0.4, 0.8]])
    outdoor = np.ones_like(density, dtype=bool)

    def entropy(hour: int) -> float:
        weights = spatial_weights(density, outdoor, hour, NIGHT, DAY, 2.0)
        return float(-(weights * np.log(weights)).sum())

    # Hour 12 has only the business part, which is more concentrated.
    assert entropy(12) < entropy(22)


def test_spatial_weights_no_density():
    with pytest.raises(ValueError, match="density"):
        spatial_weights(np.zeros((2, 2)), np.ones((2, 2), dtype=bool), 0, FLAT, FLAT, 2.0)


def test_slot_counts_length_and_mean():
    rng = np.random.default_rng(0)

    assert slot_counts(rng, 1, 15, 1000, FLAT, FLAT).size == 96
    counts = slot_counts(rng, 2, 15, 100_000, NIGHT, DAY)
    assert counts.size == 192
    assert counts.sum() == pytest.approx(200_000, rel=0.02)


def test_slot_counts_follow_profile():
    counts = slot_counts(np.random.default_rng(0), 1, 15, 10_000, [0.0] * 24, DAY)

    assert counts[: 8 * 4].sum() == 0
    assert counts[8 * 4 : 18 * 4].sum() > 0


def test_profile_wrong_length():
    with pytest.raises(ValueError, match="24"):
        slot_counts(np.random.default_rng(0), 1, 15, 1000, [1.0] * 23, FLAT)


def test_slot_minutes_must_divide_60():
    with pytest.raises(ValueError, match="slot_minutes"):
        slot_counts(np.random.default_rng(0), 1, 7, 1000, FLAT, FLAT)


# --- mdt ---

# Rows 0 to 9 (y < 0) are open ground. Rows 10 to 19 (y > 0) have a building in each
# second column, thus the tiles between the buildings have a high density.
HEIGHTS = np.zeros((20, 20))
HEIGHTS[10:, ::2] = 30.0
HEIGHT_MAP = make_height_map(HEIGHTS)
DENSITY = building_density(HEIGHT_MAP, MIN_BUILDING_HEIGHT, 30.0)
CELLS = make_sites(make_height_map(np.zeros((100, 100))))
TRAFFIC = OmegaConf.create(
    {
        "start": "2026-01-05T00:00",
        "num_days": 1,
        "slot_minutes": 15,
        "records_per_day": 500,
        "business_exponent": 2.0,
        "residential_profile": NIGHT,
        "business_profile": DAY,
    }
)
UE_HEIGHT_M = 1.5
MEASUREMENT = OmegaConf.create(
    {
        "rsrp_noise_mean_db": -5.0,
        "rsrp_noise_std_db": 2.0,
        "gnss_std_m": 0.0,
        "min_rsrp_dbm": -140.0,
        "rsrp_resolution_db": 1.0,
        "max_resample_rounds": 20,
    }
)
RSRP_COLUMNS = [f"rsrp_{cell.name}" for cell in CELLS]


def all_outdoor(positions: np.ndarray) -> np.ndarray:
    return np.ones(len(positions), dtype=bool)


def constant_rsrp(positions: np.ndarray) -> np.ndarray:
    """Give -80.3 dBm for the first cell and no signal for the other cells."""
    rsrp = np.full((len(positions), len(CELLS)), np.nan)
    rsrp[:, 0] = -80.3
    return rsrp


def run(rsrp_fn=constant_rsrp, outdoor_fn=all_outdoor, seed=0, **measurement) -> pd.DataFrame:
    return sample_mdt(
        np.random.default_rng(seed),
        rsrp_fn,
        outdoor_fn,
        CELLS,
        HEIGHT_MAP,
        MIN_BUILDING_HEIGHT,
        DENSITY,
        TRAFFIC,
        UE_HEIGHT_M,
        OmegaConf.merge(MEASUREMENT, measurement),
    )


def test_mdt_schema():
    mdt = run()

    assert list(mdt.columns) == ["timestamp", "x", "y", "z", *RSRP_COLUMNS]
    assert len(mdt) == pytest.approx(500, rel=0.2)
    assert (mdt["timestamp"].dt.minute % 15 == 0).all()
    assert mdt["timestamp"].is_monotonic_increasing
    assert mdt["timestamp"].min() >= pd.Timestamp("2026-01-05")
    assert mdt["timestamp"].max() < pd.Timestamp("2026-01-06")


def test_mdt_every_row_has_signal():
    assert run()[RSRP_COLUMNS].notna().any(axis=1).all()


def test_mdt_quantized():
    values = run()[RSRP_COLUMNS].to_numpy()
    finite = values[np.isfinite(values)]

    assert (finite == np.round(finite)).all()
    assert (finite >= -140.0).all()


def test_mdt_repeatable():
    pd.testing.assert_frame_equal(run(seed=3), run(seed=3))
    assert not run(seed=3).equals(run(seed=4))


def test_mdt_noise_mean():
    mdt = run(rsrp_noise_std_db=0.0)

    # round(-80.3 - 5) = -85
    assert (mdt[RSRP_COLUMNS[0]] == -85.0).all()
    assert mdt[RSRP_COLUMNS[1:]].isna().all().all()


def test_mdt_below_min_rsrp_is_missing():
    def two_cells(positions: np.ndarray) -> np.ndarray:
        rsrp = constant_rsrp(positions)
        rsrp[:, 1] = -139.0
        return rsrp

    mdt = run(two_cells, rsrp_noise_std_db=0.0)

    # round(-139 - 5) = -144, which is below -140.
    assert mdt[RSRP_COLUMNS[1]].isna().all()
    assert mdt[RSRP_COLUMNS[0]].notna().all()


def test_mdt_no_coverage():
    def no_signal(positions: np.ndarray) -> np.ndarray:
        return np.full((len(positions), len(CELLS)), np.nan)

    with pytest.raises(RuntimeError, match="no valid position"):
        run(no_signal, max_resample_rounds=3)


def test_mdt_resamples_dead_zone():
    def dead_west(positions: np.ndarray) -> np.ndarray:
        rsrp = constant_rsrp(positions)
        rsrp[positions[:, 0] < 0] = np.nan
        return rsrp

    assert (run(dead_west)["x"] >= 0).all()


def test_mdt_never_indoor():
    def west_is_outdoor(positions: np.ndarray) -> np.ndarray:
        return positions[:, 0] <= 0.0

    mdt = run(outdoor_fn=west_is_outdoor)

    assert (mdt["x"] <= 0.0).all()


def test_mdt_never_in_building_tile():
    mdt = run()
    col = ((mdt["x"] - HEIGHT_MAP.x_min) // HEIGHT_MAP.tile_size).astype(int)
    row = ((mdt["y"] - HEIGHT_MAP.y_min) // HEIGHT_MAP.tile_size).astype(int)

    assert (HEIGHTS[row, col] == 0.0).all()


def test_mdt_fixed_height():
    # The ground of HEIGHT_MAP is at z = 0.
    assert (run()["z"] == UE_HEIGHT_M).all()


def test_mdt_density_bias():
    mdt = run()

    # The tiles between the buildings (y > 0) have a higher density than the open ground.
    assert (mdt["y"] > 0).mean() > 0.6


def test_mdt_gnss_error_moves_xy_only():
    exact = run(seed=5)
    noisy = run(seed=5, gnss_std_m=5.0)

    assert not np.allclose(exact["x"], noisy["x"])
    np.testing.assert_allclose(exact["z"], noisy["z"])


# --- generate ---


def compose_scenario(*overrides: str):
    """Compose lab/configs/config.yaml, as `task scenario` does."""
    return load_config(*overrides)


# A small scene and a small sample count: the GPU of the development machine has 4 GB.
SMALL_RUN = (
    "scenario.scene=simple_street_canyon",
    "scenario.buildings.ground_objects=[floor]",
    "scenario.pretraining.num_scenes=2",
    "scenario.pretraining.layouts_per_scene=2",
    "scenario.pretraining.center_shift_m=5",
    "simulation.radio_map.samples_per_tx=100000",
    "simulation.radio_map.max_depth=2",
    "simulation.radio_map.diffraction=false",
    "simulation.radio_map_cell_size_m=10",
    "simulation.radio_map_tx_batch_size=50",
    "scenario.sites.isd_m=60",
    "scenario.sites.mast_height_m=10",
    "scenario.traffic.records_per_day=200",
    "scenario.traffic.density_radius_m=20",
)
SCENARIOS = (
    "munich_map_errors",
    "munich_new_site",
    "munich_unmeasured_area",
    "munich_sparse_mdt",
    "munich_ablation",
)


def test_scenario_config_composes():
    cfg = compose_scenario()

    assert cfg.scenario.name == "munich_map_errors"
    assert cfg.scenario.synthetic is True
    assert cfg.simulation.radio_map.refraction is True
    assert cfg.scenario.grid.tile_size_m > 0
    assert "radio" not in cfg.scenario
    assert "seed" not in cfg.scenario and "seed" not in cfg.calibration


@pytest.mark.parametrize("name", SCENARIOS)
def test_each_scenario_composes(name):
    cfg = compose_scenario(f"scenario={name}")

    assert cfg.scenario.name == name
    assert cfg.output.figures.dir == f"lab/reports/{name}/figures"


def test_generate_writes_outputs(tmp_path):
    generate(compose_scenario(*SMALL_RUN), tmp_path)

    cells = read_cells(tmp_path / "cells.csv")
    mdt = pd.read_csv(tmp_path / "mdt.csv", parse_dates=["timestamp"])
    maps = read_maps(tmp_path / "maps.npz")
    rsrp_columns = [f"rsrp_{cell.name}" for cell in cells]

    assert len(cells) == 9
    assert list(mdt.columns) == ["timestamp", "x", "y", "z", *rsrp_columns]
    assert len(mdt) == pytest.approx(200, rel=0.3)
    assert mdt[rsrp_columns].notna().any(axis=1).all()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cells.csv", "maps.npz", "mdt.csv"]
    assert maps.target.num_cells == 9
    assert maps.true_path_gain.shape == maps.target.path_gain.shape
    assert len(maps.variants) == len(maps.variant_labels) == 4
    assert maps.variant_scenes.tolist() == [0, 0, 1, 1]


def test_generate_counts_radio_map_calls(tmp_path, monkeypatch, caplog):
    calls = []
    real_map = generate_module.compute_path_gain_map
    real_inputs = generate_module.compute_map_inputs

    def counted_map(scene, cells, grid, simulation):
        calls.append(len(cells))
        return real_map(scene, cells, grid, simulation)

    def counted_inputs(scene, cells, grid, height_map, simulation, min_height):
        calls.append(len(cells))
        return real_inputs(scene, cells, grid, height_map, simulation, min_height)

    monkeypatch.setattr(generate_module, "compute_path_gain_map", counted_map)
    monkeypatch.setattr(generate_module, "compute_map_inputs", counted_inputs)
    with caplog.at_level(logging.INFO, logger="scenario.generate"):
        generate(compose_scenario(*SMALL_RUN), tmp_path)

    # The original scene (9 cells and 4 layouts of 9 cells), the true world, and each of
    # the 2 changed pretraining scenes (2 layouts of 9 cells).
    assert calls == [45, 9, 18, 18]
    total = [r for r in caplog.records if r.getMessage().startswith("Total:")]
    assert len(total) == 1 and total[0].args[1] == 4


def test_generate_rejects_not_synthetic(tmp_path):
    with pytest.raises(ValueError, match="synthetic"):
        generate(compose_scenario(*SMALL_RUN, "scenario.synthetic=false"), tmp_path)
