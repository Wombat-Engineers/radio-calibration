"""The column names of `mdt.csv`."""

MDT_POSITION_COLUMNS: tuple[str, ...] = ("timestamp", "x", "y", "z")


def rsrp_column(cell_name: str) -> str:
    """Give the name of the RSRP column of a cell in `mdt.csv`.

    Args:
        cell_name: Name of the cell.

    Returns:
        The column name.
    """
    return f"rsrp_{cell_name}"
