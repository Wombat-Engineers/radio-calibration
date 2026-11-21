"""Figures of the maps, the errors, and the training of one scenario."""

import string
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from experiments.data import ScenarioData
from radio_calibration.core import path_gain_to_db
from utils import COLORS, FIGURE_WIDTH_DOUBLE

# Percentile of the absolute error that sets the limit of the color scale of an error map.
# Assumption.
_ERROR_PERCENTILE = 95


def _panels(count: int, columns: int = 3) -> tuple[Figure, np.ndarray]:
    rows = -(-count // columns)
    width = FIGURE_WIDTH_DOUBLE
    figure, axes = plt.subplots(
        rows, columns, figsize=(width, width * 0.3 * rows), squeeze=False, sharex=True, sharey=True
    )
    for axis in axes.ravel()[count:]:
        axis.set_visible(False)
    return figure, axes.ravel()[:count]


def _extent(data: ScenarioData) -> list[float]:
    grid = data.maps.target.grid
    return [
        grid.x_min,
        grid.x_min + grid.nx * grid.cell_size,
        grid.y_min,
        grid.y_min + grid.ny * grid.cell_size,
    ]


def _label(axis: plt.Axes, index: int, name: str) -> None:
    axis.set_title(f"({string.ascii_lowercase[index]}) {name}", loc="left")


def map_figure(data: ScenarioData, maps_db: Mapping[str, np.ndarray], cell: int) -> Figure:
    """Show the path gain map of one cell for each model, with one color scale.

    The true map comes first if the scenario has one.
    """
    maps = {name: values[cell] for name, values in maps_db.items()}
    if data.maps.true_path_gain is not None:
        maps = {"True map": path_gain_to_db(data.maps.true_path_gain[cell])} | maps
    finite = np.concatenate([values[np.isfinite(values)] for values in maps.values()])
    low, high = np.percentile(finite, [1, 99])
    figure, axes = _panels(len(maps))
    for index, (axis, (name, values)) in enumerate(zip(axes, maps.items(), strict=True)):
        image = axis.imshow(
            values, origin="lower", extent=_extent(data), vmin=low, vmax=high, cmap="viridis"
        )
        axis.plot(*data.maps.target.cell_xy[cell], marker="^", color=COLORS[1])
        _label(axis, index, name)
    for axis in axes:
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    figure.colorbar(image, ax=list(axes), label="Path gain (dB)", shrink=0.8)
    return figure


def error_map_figure(data: ScenarioData, maps_db: Mapping[str, np.ndarray], cell: int) -> Figure:
    """Show the true path gain minus the path gain of each model, in dB, for one cell.

    Raises:
        ValueError: If the scenario has no true map.
    """
    if data.maps.true_path_gain is None:
        raise ValueError("The scenario has no true map.")
    truth = path_gain_to_db(data.maps.true_path_gain[cell])
    errors = {name: truth - values[cell] for name, values in maps_db.items()}
    finite = np.concatenate([np.abs(e[np.isfinite(e)]) for e in errors.values()])
    limit = float(np.percentile(finite, _ERROR_PERCENTILE)) if finite.size else 1.0
    figure, axes = _panels(len(errors))
    for index, (axis, (name, error)) in enumerate(zip(axes, errors.items(), strict=True)):
        image = axis.imshow(
            error, origin="lower", extent=_extent(data), vmin=-limit, vmax=limit, cmap="RdBu_r"
        )
        axis.plot(*data.maps.target.cell_xy[cell], marker="^", color=COLORS[7])
        _label(axis, index, name)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    figure.colorbar(image, ax=list(axes), label="True minus predicted path gain (dB)", shrink=0.8)
    return figure


def history_figure(histories: Mapping[str, pd.DataFrame]) -> Figure:
    """Show the train loss and the validation loss of each history against the epoch."""
    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_DOUBLE, FIGURE_WIDTH_DOUBLE * 0.35))
    for color, (name, history) in zip(COLORS, histories.items(), strict=False):
        axes[0].plot(history["epoch"], history["loss"], color=color, label=name)
        axes[1].plot(history["epoch"], history["validation_loss"], color=color, label=name)
    for index, (axis, quantity) in enumerate(zip(axes, ["Train", "Validation"], strict=True)):
        axis.set_xlabel("Epoch")
        axis.set_ylabel(f"{quantity} MAE (dB)")
        _label(axis, index, "")
    axes[1].legend()
    return figure


def error_cdf_figure(errors_db: Mapping[str, np.ndarray]) -> Figure:
    """Show the empirical distribution of the absolute error of each model."""
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH_DOUBLE / 2, FIGURE_WIDTH_DOUBLE * 0.3))
    for color, (name, error) in zip(COLORS, errors_db.items(), strict=False):
        values = np.sort(np.abs(error[np.isfinite(error)]))
        axis.plot(values, np.arange(1, values.size + 1) / values.size, color=color, label=name)
    axis.set_xlabel("Absolute error (dB)")
    axis.set_ylabel("Empirical CDF")
    axis.legend()
    return figure
