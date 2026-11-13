"""Change of the MDT records to the long format of `data/processed/mdt.parquet`."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.core import MDT_POSITION_COLUMNS, Cell, rsrp_column


def mdt_to_long(mdt: pd.DataFrame, cells: Sequence[Cell]) -> pd.DataFrame:
    """Change the MDT records to one row for each received cell of each record.

    Validate `mdt` with `validate_mdt` before you call this function.

    Args:
        mdt: The MDT records, with the columns of `mdt.csv`. A missing RSRP is NaN.
        cells: The cells. Each cell must have one RSRP column in `mdt`.

    Returns:
        A frame with the columns `record`, `timestamp`, `x`, `y`, `z`, `cell`, and
        `rsrp_dbm`. `record` is the position of the row in `mdt`, from 0. A missing RSRP
        gives no row. The rows are in the order of `record`, then in the order of `cells`.
    """
    rsrp = mdt[[rsrp_column(cell.name) for cell in cells]].to_numpy(dtype=float)
    record, cell_index = np.nonzero(~np.isnan(rsrp))
    frame = mdt[list(MDT_POSITION_COLUMNS)].iloc[record].reset_index(drop=True)
    frame.insert(0, "record", record.astype(np.int64))
    frame["cell"] = pd.Series([cells[i].name for i in cell_index], dtype="str")
    frame["rsrp_dbm"] = rsrp[record, cell_index]
    return frame
