"""Long format of the MDT records in `lab/data/<scenario>/processed/`, and its division."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.mdt import MDT_POSITION_COLUMNS, rsrp_column
from radio_calibration.core import Cell

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


def _check_fractions(fractions: Sequence[float]) -> None:
    if len(fractions) != len(SPLIT_NAMES):
        raise ValueError(
            f"fractions must have {len(SPLIT_NAMES)} values, but it has {len(fractions)}."
        )
    if min(fractions) < 0 or not math.isclose(sum(fractions), 1.0):
        raise ValueError(
            f"fractions must be 0 or more with a sum of 1, but it is {list(fractions)}."
        )


def _parts_by_key(
    mdt: pd.DataFrame, keys: pd.Series, fractions: Sequence[float], seed: int
) -> dict[str, pd.DataFrame]:
    """Put all rows of one key into one part.

    The train part and the validation part get their fraction of the keys, rounded down. The
    test part gets all other keys.
    """
    unique = np.random.default_rng(seed).permutation(keys.unique())
    num_train = int(fractions[0] * len(unique))
    num_validation = int(fractions[1] * len(unique))
    parts = np.split(unique, [num_train, num_train + num_validation])
    return {
        name: mdt[keys.isin(part).to_numpy()].reset_index(drop=True)
        for name, part in zip(SPLIT_NAMES, parts, strict=True)
    }


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
    _check_fractions(fractions)
    return _parts_by_key(mdt, mdt["record"], fractions, seed)


def split_by_site(
    mdt: pd.DataFrame,
    cells: Sequence[Cell],
    test_sites: Sequence[int],
    validation_fraction: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Put all rows of the cells of the test sites into the test part.

    The other rows go to the train part and the validation part by record. Thus no train row
    and no validation row is for a cell of a test site. One record can have rows in the test
    part and in another part, each for a different cell.

    Args:
        mdt: The MDT records in the long format of `mdt_to_long`.
        cells: The cells. Each `cell` value of `mdt` must be the name of one of them.
        test_sites: Site indices of the test part.
        validation_fraction: Fraction of the other records for the validation part.
        seed: Seed of the random order of the records.

    Returns:
        One frame for each name in `SPLIT_NAMES`, with the columns of `mdt`.

    Raises:
        ValueError: If no cell is on a test site, if all cells are on test sites, or if
            `validation_fraction` is not from 0 to less than 1.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError(
            f"validation_fraction must be from 0 to less than 1, but it is {validation_fraction}."
        )
    test_cells = {cell.name for cell in cells if cell.site in set(test_sites)}
    if not test_cells or len(test_cells) == len(cells):
        raise ValueError(f"The test sites {list(test_sites)} must have some cells, not all.")
    is_test = mdt["cell"].isin(test_cells).to_numpy()
    others = mdt[~is_test].reset_index(drop=True)
    records = np.random.default_rng(seed).permutation(others["record"].unique())
    validation = records[: int(validation_fraction * len(records))]
    in_validation = others["record"].isin(validation).to_numpy()
    return {
        "train": others[~in_validation].reset_index(drop=True),
        "validation": others[in_validation].reset_index(drop=True),
        "test": mdt[is_test].reset_index(drop=True),
    }


def split_spatial(
    mdt: pd.DataFrame, block_size_m: float, fractions: Sequence[float], seed: int
) -> dict[str, pd.DataFrame]:
    """Divide the MDT records by square blocks of their reported position.

    All records of one block go to the same part. Thus a test record has no train record in
    its block.

    Args:
        mdt: The MDT records in the long format of `mdt_to_long`.
        block_size_m: Side of a block, in meters.
        fractions: Fractions of the blocks for the train, validation, and test parts.
        seed: Seed of the random order of the blocks.

    Returns:
        One frame for each name in `SPLIT_NAMES`, with the columns of `mdt`.

    Raises:
        ValueError: If `block_size_m` is not more than 0, or if `fractions` is not valid.
    """
    if not block_size_m > 0:
        raise ValueError(f"block_size_m must be more than 0, but it is {block_size_m}.")
    _check_fractions(fractions)
    column = np.floor(mdt["x"].to_numpy() / block_size_m).astype(np.int64)
    row = np.floor(mdt["y"].to_numpy() / block_size_m).astype(np.int64)
    keys = pd.Series([f"{r}_{c}" for r, c in zip(row, column, strict=True)], index=mdt.index)
    return _parts_by_key(mdt, keys, fractions, seed)


def split_mdt(
    mdt: pd.DataFrame, cells: Sequence[Cell], split: Mapping[str, Any], seed: int
) -> dict[str, pd.DataFrame]:
    """Divide the MDT records with the method of the block `scenario.split`.

    Args:
        mdt: The MDT records in the long format of `mdt_to_long`.
        cells: The cells.
        split: The key `method` and the keys of that method. `random` uses `fractions`.
            `site` uses `test_sites` and `validation_fraction`. `spatial` uses
            `block_size_m` and `fractions`.
        seed: The seed.

    Returns:
        One frame for each name in `SPLIT_NAMES`.

    Raises:
        ValueError: If the method is not known, or if a value is not valid.
        KeyError: If a key of the method is missing.
    """
    method = split["method"]
    if method == "random":
        return split_records(mdt, split["fractions"], seed)
    if method == "site":
        return split_by_site(mdt, cells, split["test_sites"], split["validation_fraction"], seed)
    if method == "spatial":
        return split_spatial(mdt, split["block_size_m"], split["fractions"], seed)
    raise ValueError(f"The split method '{method}' is not random, site, or spatial.")


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
