"""Scene load."""

from pathlib import Path

import sionna.rt


def load_rt_scene(source: str | Path, merge_shapes: bool = True) -> sionna.rt.Scene:
    """Load a Sionna RT scene.

    Args:
        source: The name of a scene that Sionna RT supplies (for example `munich`), or the
            path of a Mitsuba scene file with the suffix `.xml`.
        merge_shapes: If True, Sionna RT merges the shapes that have the same radio
            material. This makes the ray tracing faster. Use False to move, scale, or
            remove one building: then each shape of the file is one scene object.

    Returns:
        The scene.

    Raises:
        ValueError: If `source` is a name and Sionna RT has no scene with this name, or if
            `source` is a path that does not exist or does not have the suffix `.xml`.
    """
    if isinstance(source, Path):
        if source.suffix != ".xml" or not source.is_file():
            raise ValueError(f"The scene file '{source}' does not exist or is not an .xml file.")
        return sionna.rt.load_scene(str(source), merge_shapes=merge_shapes)
    builtin = getattr(sionna.rt.scene, source, None)
    if not (isinstance(builtin, str) and builtin.endswith(".xml")):
        raise ValueError(f"Sionna RT has no scene with the name '{source}'.")
    return sionna.rt.load_scene(builtin, merge_shapes=merge_shapes)
