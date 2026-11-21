"""Format of `maps.npz`: the input maps, the true map, and the pretraining maps of a scenario."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radio_calibration.core import MapInputs, RadioMapGrid

_GRID_FIELDS = ("x_min", "y_min", "z", "cell_size", "nx", "ny")
_KEYS = (
    "grid",
    "max_height",
    "building_fraction",
    "target_path_gain",
    "target_los",
    "target_cell_xy",
    "true_path_gain",
    "variant_path_gain",
    "variant_los",
    "variant_cell_xy",
    "variant_label",
    "variant_scene",
)


@dataclass(frozen=True)
class MapsFile:
    """Content of `maps.npz`.

    Attributes:
        target: Input maps of the cells of the scenario, on the original scene.
        true_path_gain: True linear path gain of the cells of the scenario, with shape
            `[n_cells, ny, nx]`, or None if the scenario has no true map.
        variants: Input maps of each pretraining layout, on the original scene.
        variant_labels: True linear path gain of each pretraining layout on its changed
            scene, with the shape of `variants[i].path_gain`.
        variant_scenes: Index of the changed scene of each pretraining layout.
    """

    target: MapInputs
    true_path_gain: np.ndarray | None
    variants: list[MapInputs]
    variant_labels: list[np.ndarray]
    variant_scenes: np.ndarray


def write_maps(path: Path, maps: MapsFile) -> None:
    """Write the maps to a compressed `.npz` file.

    All pretraining layouts must have the same number of cells. Path gains are written as
    float32: the smallest normal float32 (about 1e-38) is far below a path gain of the
    radio-map solver.

    Raises:
        ValueError: If the layouts do not have the same number of cells, or if there is no
            layout.
    """
    if not maps.variants or len({variant.num_cells for variant in maps.variants}) != 1:
        raise ValueError(
            "There must be a minimum of one layout, and each must have the same cells."
        )
    grid = maps.target.grid
    true_path_gain = (
        np.zeros((0, grid.ny, grid.nx)) if maps.true_path_gain is None else maps.true_path_gain
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        grid=np.array([getattr(grid, name) for name in _GRID_FIELDS], dtype=float),
        max_height=maps.target.max_height,
        building_fraction=maps.target.building_fraction,
        target_path_gain=maps.target.path_gain.astype(np.float32),
        target_los=maps.target.los,
        target_cell_xy=maps.target.cell_xy,
        true_path_gain=np.asarray(true_path_gain, dtype=np.float32),
        variant_path_gain=np.stack([v.path_gain for v in maps.variants]).astype(np.float32),
        variant_los=np.stack([v.los for v in maps.variants]),
        variant_cell_xy=np.stack([v.cell_xy for v in maps.variants]),
        variant_label=np.stack(maps.variant_labels).astype(np.float32),
        variant_scene=np.asarray(maps.variant_scenes, dtype=int),
    )


def read_maps(path: Path) -> MapsFile:
    """Read a file of `write_maps`. The function does not permit pickled objects.

    Raises:
        ValueError: If the file does not have a key.
    """
    with np.load(path, allow_pickle=False) as content:
        missing = [key for key in _KEYS if key not in content]
        if missing:
            raise ValueError(f"The maps file '{path}' does not have the keys {missing}.")
        data = {key: content[key] for key in _KEYS}
    values = data["grid"]
    grid = RadioMapGrid(*(float(v) for v in values[:4]), int(values[4]), int(values[5]))

    def inputs(path_gain: np.ndarray, los: np.ndarray, cell_xy: np.ndarray) -> MapInputs:
        return MapInputs(
            grid=grid,
            path_gain=path_gain.astype(float),
            los=los.astype(bool),
            max_height=data["max_height"],
            building_fraction=data["building_fraction"],
            cell_xy=cell_xy,
        )

    variants = [
        inputs(path_gain, los, cell_xy)
        for path_gain, los, cell_xy in zip(
            data["variant_path_gain"], data["variant_los"], data["variant_cell_xy"], strict=True
        )
    ]
    true_path_gain = data["true_path_gain"].astype(float)
    return MapsFile(
        target=inputs(data["target_path_gain"], data["target_los"], data["target_cell_xy"]),
        true_path_gain=true_path_gain if len(true_path_gain) else None,
        variants=variants,
        variant_labels=[label.astype(float) for label in data["variant_label"]],
        variant_scenes=data["variant_scene"],
    )
