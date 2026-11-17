"""Scene load."""

from pathlib import Path

import sionna.rt


def load_rt_scene(source: str | Path) -> sionna.rt.Scene:
    """Load a Sionna RT scene.

    Args:
        source: The name of a scene that Sionna RT supplies (for example `munich`), or the
            path of a Mitsuba scene file with the suffix `.xml`.

    Returns:
        The scene.

    Raises:
        ValueError: If `source` is a name and Sionna RT has no scene with this name, or if
            `source` is a path that is not a file with the suffix `.xml`.
    """
    if isinstance(source, Path):
        if source.suffix != ".xml" or not source.is_file():
            raise ValueError(f"The scene file '{source}' is not an .xml file.")
        return sionna.rt.load_scene(str(source))
    builtin = getattr(sionna.rt.scene, source, None)
    if not (isinstance(builtin, str) and builtin.endswith(".xml")):
        raise ValueError(f"Sionna RT has no scene with the name '{source}'.")
    return sionna.rt.load_scene(builtin)
