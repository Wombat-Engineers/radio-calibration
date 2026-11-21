"""Shared steps of the scenario notebooks: data, training, evaluation, and figures.

The notebooks of each scenario call these functions, thus the steps are the same in each
scenario. This package does no ray tracing.
"""

from experiments.data import ScenarioData, first_records, floor_db, load_scenario, measurable_db
from experiments.evaluation import (
    SIONNA_RT,
    SIONNA_RT_OFFSET,
    SPM_ATOLL,
    baseline_maps,
    cell_table,
    evaluated_cells,
    map_rsrp_dbm,
    map_table,
    mdt_table,
)
from experiments.figures import error_cdf_figure, error_map_figure, history_figure, map_figure
from experiments.training import (
    channels_without,
    fine_tune_model,
    load_model,
    new_model,
    pretrain_model,
    save_model,
)

__all__ = [
    "SIONNA_RT",
    "SIONNA_RT_OFFSET",
    "SPM_ATOLL",
    "ScenarioData",
    "baseline_maps",
    "cell_table",
    "channels_without",
    "error_cdf_figure",
    "error_map_figure",
    "evaluated_cells",
    "fine_tune_model",
    "first_records",
    "floor_db",
    "history_figure",
    "load_model",
    "load_scenario",
    "map_figure",
    "map_rsrp_dbm",
    "map_table",
    "mdt_table",
    "measurable_db",
    "new_model",
    "pretrain_model",
    "save_model",
]
