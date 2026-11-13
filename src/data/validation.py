"""Rules that the cells and the MDT records must obey before the calibration uses them."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.core import MDT_POSITION_COLUMNS, Cell, rsrp_column

# Tolerance for the floating-point error of RSRP / resolution. A reported RSRP is an exact
# multiple of the resolution, so a larger difference is a real error.
_MULTIPLE_TOLERANCE = 1e-6


def validate_cells(cells: Sequence[Cell]) -> dict[str, int]:
    """Count the cells that break each rule.

    Args:
        cells: The cells.

    Returns:
        For each rule, the number of cells that break it. A value of 0 shows that all cells
        obey the rule.
    """
    names = pd.Series([cell.name for cell in cells], dtype=object)
    positions = np.array([[cell.x, cell.y, cell.z] for cell in cells], dtype=float).reshape(-1, 3)
    return {
        "Duplicate cell name": int(names.duplicated().sum()),
        "Cell position is not finite": int((~np.isfinite(positions)).any(axis=1).sum()),
    }


def validate_mdt(
    mdt: pd.DataFrame,
    cells: Sequence[Cell],
    min_rsrp_dbm: float,
    rsrp_resolution_db: float,
) -> dict[str, int]:
    """Count the MDT records that break each rule.

    Args:
        mdt: The MDT records, with the columns of `mdt.csv`. The `timestamp` column must be
            of a datetime type.
        cells: The cells. Each cell must have one RSRP column in `mdt`.
        min_rsrp_dbm: The lowest RSRP that a record can contain, in dBm.
        rsrp_resolution_db: The resolution of a reported RSRP, in dB.

    Returns:
        For each rule, the number of records that break it. A value of 0 shows that all
        records obey the rule. A missing RSRP is NaN.

    Raises:
        ValueError: If the columns are not the MDT columns of `cells`, or if `timestamp`
            is not of a datetime type.
    """
    rsrp_columns = [rsrp_column(cell.name) for cell in cells]
    expected = [*MDT_POSITION_COLUMNS, *rsrp_columns]
    missing = [column for column in expected if column not in mdt.columns]
    extra = [column for column in mdt.columns if column not in expected]
    if missing or extra:
        raise ValueError(
            f"The MDT columns are not correct. Missing columns: {missing}. "
            f"Columns that are not MDT columns: {extra}."
        )
    if not pd.api.types.is_datetime64_any_dtype(mdt["timestamp"]):
        raise ValueError(
            f"The column timestamp has the type {mdt['timestamp'].dtype}, not datetime."
        )

    positions = mdt[["x", "y", "z"]].to_numpy(dtype=float)
    rsrp = mdt[rsrp_columns].to_numpy(dtype=float)
    received = ~np.isnan(rsrp)
    steps = rsrp / rsrp_resolution_db
    on_grid = np.abs(steps - np.round(steps)) <= _MULTIPLE_TOLERANCE
    return {
        "Timestamp is missing": int(mdt["timestamp"].isna().sum()),
        "Position is not finite": int((~np.isfinite(positions)).any(axis=1).sum()),
        "Record has no cell": int((~received).all(axis=1).sum()),
        "RSRP is below min_rsrp_dbm": int((received & (rsrp < min_rsrp_dbm)).any(axis=1).sum()),
        "RSRP is not a multiple of the resolution": int((received & ~on_grid).any(axis=1).sum()),
        "Record is a duplicate": int(mdt.duplicated().sum()),
    }
