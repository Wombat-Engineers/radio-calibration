"""Long format of the MDT records in `data/processed/`, and its division into parts."""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.core import MDT_POSITION_COLUMNS, Cell, rsrp_column

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")


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


def split_records(
    mdt: pd.DataFrame, fractions: Sequence[float], seed: int
) -> dict[str, pd.DataFrame]:
    """Divide the MDT records at random into the train, validation, and test parts.

    All rows of one record go to the same part. The number of records of the train part
    and of the validation part is the fraction of the number of records, rounded down.
    The test part gets all other records.

    Args:
        mdt: The MDT records in the long format of `mdt_to_long`.
        fractions: The fractions of the records for the train, validation, and test parts.
        seed: Seed of the random order of the records.

    Returns:
        One frame for each name in `SPLIT_NAMES`, with the columns and the row order of
        `mdt`. The `record` values do not change.

    Raises:
        ValueError: If `fractions` does not have one value for each part, if a value is
            less than 0, or if the sum is not 1.
    """
    if len(fractions) != len(SPLIT_NAMES):
        raise ValueError(
            f"fractions must have {len(SPLIT_NAMES)} values, but it has {len(fractions)}."
        )
    if min(fractions) < 0 or not math.isclose(sum(fractions), 1.0):
        raise ValueError(
            f"fractions must be 0 or more with a sum of 1, but it is {list(fractions)}."
        )
    records = np.random.default_rng(seed).permutation(mdt["record"].unique())
    num_train = int(fractions[0] * len(records))
    num_validation = int(fractions[1] * len(records))
    parts = np.split(records, [num_train, num_train + num_validation])
    return {
        name: mdt[mdt["record"].isin(part)].reset_index(drop=True)
        for name, part in zip(SPLIT_NAMES, parts, strict=True)
    }


def mdt_to_arrays(mdt: pd.DataFrame, cells: Sequence[Cell]) -> tuple[np.ndarray, np.ndarray]:
    """Change MDT records in the long format to one position and one RSRP row for each record.

    Args:
        mdt: The MDT records in the long format of `mdt_to_long`.
        cells: The cells. Each `cell` value of `mdt` must be the name of one of them.

    Returns:
        The positions in meters, with shape `[n, 3]`, and the RSRP in dBm, with shape
        `[n, len(cells)]`. `n` is the number of records. The rows are in the order of
        `record`. An RSRP is NaN if the record does not have the cell.

    Raises:
        ValueError: If a `cell` value of `mdt` is not the name of a cell.
    """
    names = [cell.name for cell in cells]
    unknown = sorted(set(mdt["cell"]) - set(names))
    if unknown:
        raise ValueError(f"The cells {unknown} are not in `cells`.")
    records, row = np.unique(mdt["record"].to_numpy(), return_inverse=True)
    first_rows = mdt.drop_duplicates("record").sort_values("record")
    positions = first_rows[["x", "y", "z"]].to_numpy(dtype=float)
    rsrp = np.full((len(records), len(cells)), np.nan)
    column = mdt["cell"].map({name: index for index, name in enumerate(names)})
    rsrp[row, column.to_numpy(dtype=int)] = mdt["rsrp_dbm"].to_numpy(dtype=float)
    return positions, rsrp
