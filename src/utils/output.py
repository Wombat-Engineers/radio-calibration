"""Output files of the notebooks: figures and tables."""

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig
from pandas.io.formats.style import Styler

from src.utils.paths import PROJECT_ROOT

# Directories that `configure_output` sets, with the keys "figures" and "tables".
_directories: dict[str, Path] = {}


def configure_output(output: DictConfig) -> None:
    """Set the directories of the figures and the tables.

    Args:
        output: The block `output` of `configs/config.yaml`, with `figures.dir` and
            `tables.dir`. A relative path starts at the repository root.
    """
    _directories["figures"] = PROJECT_ROOT / output.figures.dir
    _directories["tables"] = PROJECT_ROOT / output.tables.dir


def _directory(kind: str) -> Path:
    """Return the directory of `kind` and create it if necessary.

    Raises:
        RuntimeError: If `configure_output` was not called.
    """
    if kind not in _directories:
        raise RuntimeError(
            f"No directory for the {kind}. Call setup_notebook or configure_output first."
        )
    directory = _directories[kind]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_figure(figure: Figure, name: str) -> Path:
    """Save a figure as a PNG file in the figure directory.

    Args:
        figure: The figure.
        name: File name without the extension.

    Returns:
        The path of the file.

    Raises:
        RuntimeError: If `configure_output` was not called.
    """
    path = _directory("figures") / f"{name}.png"
    figure.savefig(path)
    return path


def save_table[T: (pd.DataFrame, Styler)](table: T, name: str) -> T:
    """Save a table as a CSV file in the table directory, with its index.

    Args:
        table: The table. For a styled table, the function saves the data without the style.
        name: File name without the extension.

    Returns:
        `table`, so that the call can be the last expression of a notebook cell.

    Raises:
        RuntimeError: If `configure_output` was not called.
    """
    frame = table.data if isinstance(table, Styler) else table
    frame.to_csv(_directory("tables") / f"{name}.csv")
    return table
