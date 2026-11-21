"""Random changes of the buildings of a scene: move, change the height, remove.

The changed scene is the "true world" of a synthetic scenario. The model sees only the
original scene. Thus the difference between the two scenes is the error of the 3D map that
the model must correct.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import mitsuba as mi
import numpy as np
import pandas as pd
import sionna.rt
from omegaconf import DictConfig

# The scenes of Sionna RT give each shape the name `<building>-itu_<material>`, for example
# `Heilig_Geist-itu_marble` in `munich.xml` of Sionna RT 2.2.0. The shapes of one building
# have the same `<building>` part.
_MATERIAL_SUFFIX = re.compile(r"-itu_\w+$")

CHANGE_COLUMNS = ("building", "dx_m", "dy_m", "height_factor", "deleted")


@dataclass(frozen=True)
class BuildingSettings:
    """Settings of the random changes. Each building gets its changes independently.

    Attributes:
        ground_objects: Names of the scene objects that are not buildings.
        move_fraction: Probability that a building moves.
        max_shift_m: A moved building moves by a uniform value from `-max_shift_m` to
            `max_shift_m` on x and on y, in meters.
        height_fraction: Probability that the height of a building changes.
        height_factor: Lowest and highest factor of a changed height. The factor is uniform.
        delete_fraction: Probability that a building is removed.

    Raises:
        ValueError: If a probability is not from 0 to 1, if `max_shift_m` is negative, or
            if a height factor is not more than 0.
    """

    ground_objects: tuple[str, ...]
    move_fraction: float
    max_shift_m: float
    height_fraction: float
    height_factor: tuple[float, float]
    delete_fraction: float

    def __post_init__(self) -> None:
        for name in ("move_fraction", "height_fraction", "delete_fraction"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be from 0 to 1, but it is {getattr(self, name)}.")
        if self.max_shift_m < 0:
            raise ValueError(f"max_shift_m must be 0 or more, but it is {self.max_shift_m}.")
        low, high = self.height_factor
        if not 0 < low <= high:
            raise ValueError(f"height_factor must be 0 < low <= high, but it is {low}, {high}.")

    @classmethod
    def from_config(cls, config: DictConfig) -> "BuildingSettings":
        """Make the settings from the block `scenario.buildings` of the configuration."""
        return cls(
            ground_objects=tuple(config.ground_objects),
            move_fraction=float(config.move_fraction),
            max_shift_m=float(config.max_shift_m),
            height_fraction=float(config.height_fraction),
            height_factor=(float(config.height_factor[0]), float(config.height_factor[1])),
            delete_fraction=float(config.delete_fraction),
        )


def building_groups(
    object_names: Iterable[str], ground_objects: Sequence[str]
) -> dict[str, list[str]]:
    """Group the scene objects by building.

    Args:
        object_names: Names of the scene objects.
        ground_objects: Names of the objects that are not buildings.

    Returns:
        For each building name, the names of its objects, in sorted order. The building
        name is the object name with no `-itu_<material>` suffix.
    """
    groups: dict[str, list[str]] = {}
    for name in sorted(object_names):
        if name not in ground_objects:
            groups.setdefault(_MATERIAL_SUFFIX.sub("", name), []).append(name)
    return groups


def perturb_buildings(
    scene: sionna.rt.Scene, rng: np.random.Generator, settings: BuildingSettings
) -> pd.DataFrame:
    """Change the buildings of the scene in place at random.

    Load the scene with `merge_shapes=False`: then each building has its own objects. A
    building with a new height keeps its lowest point at the same z. The function draws all
    random values before it changes the scene, thus the same seed gives the same changes.

    Args:
        scene: The scene.
        rng: Random generator.
        settings: The settings of the changes.

    Returns:
        One row for each building, with the columns `CHANGE_COLUMNS`. A removed building
        has no shift and a height factor of 1.
    """
    groups = building_groups(scene.objects, settings.ground_objects)
    names = list(groups)
    count = len(names)
    deleted = rng.random(count) < settings.delete_fraction
    moved = (rng.random(count) < settings.move_fraction) & ~deleted
    resized = (rng.random(count) < settings.height_fraction) & ~deleted
    shift = rng.uniform(-settings.max_shift_m, settings.max_shift_m, size=(count, 2))
    shift = np.where(moved[:, None], shift, 0.0)
    factor = np.where(resized, rng.uniform(*settings.height_factor, size=count), 1.0)

    objects = scene.objects
    for index in np.nonzero(moved | resized)[0]:
        members = [objects[name] for name in groups[names[index]]]
        base_z = min(float(member.mi_mesh.bbox().min.z) for member in members)
        for member in members:
            center = member.position
            if resized[index]:
                member.scaling = mi.Vector3f(1.0, 1.0, float(factor[index]))
            member.position = mi.Point3f(
                float(center.x[0]) + float(shift[index, 0]),
                float(center.y[0]) + float(shift[index, 1]),
                base_z + float(factor[index]) * (float(center.z[0]) - base_z),
            )
    removed = [objects[name] for index in np.nonzero(deleted)[0] for name in groups[names[index]]]
    if removed:
        scene.edit(remove=removed)
    return pd.DataFrame(
        {
            "building": names,
            "dx_m": shift[:, 0],
            "dy_m": shift[:, 1],
            "height_factor": factor,
            "deleted": deleted,
        },
        columns=list(CHANGE_COLUMNS),
    )
