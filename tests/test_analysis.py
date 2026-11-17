"""Tests for analysis."""

import math

import numpy as np
import pandas as pd
import pytest

from analysis import PAIR_COLUMNS, pair_errors, paired_bootstrap
from radio_calibration.core import Cell


def make_cell(name: str, x: float, y: float, z: float) -> Cell:
    return Cell(
        name=name,
        site=0,
        x=x,
        y=y,
        z=z,
        azimuth_deg=0.0,
        downtilt_deg=5.0,
        power_rs_dbm=4.85,
        num_rows=8,
        num_cols=8,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="cross",
    )


CELLS = [make_cell("a", 0.0, 0.0, 10.0), make_cell("b", 100.0, 0.0, 10.0)]


def long_part(rows) -> pd.DataFrame:
    """Make a part in the long format from (record, cell, x, y, z, rsrp_dbm) rows."""
    frame = pd.DataFrame(rows, columns=["record", "cell", "x", "y", "z", "rsrp_dbm"])
    frame.insert(1, "timestamp", pd.Timestamp("2026-01-05"))
    return frame


# Record 7 has the two cells. Record 9 has only cell "a".
PART = long_part(
    [
        (7, "a", 3.0, 4.0, 10.0, -80.0),
        (7, "b", 3.0, 4.0, 10.0, -90.0),
        (9, "a", 0.0, 0.0, 1.5, -70.0),
    ]
)
SIMULATED = np.array([[-85.0, -90.0], [-70.0, -60.0]])


def pairs_with_errors(errors, records=None) -> pd.DataFrame:
    """Make a frame of `pair_errors` with these signed errors and a measured RSRP of -80 dBm."""
    errors = np.asarray(errors, dtype=float)
    records = np.arange(errors.size) if records is None else np.asarray(records)
    return pd.DataFrame(
        {
            "record": records,
            "cell": ["a"] * errors.size,
            "x": 0.0,
            "y": 0.0,
            "z": 1.5,
            "rsrp_dbm": -80.0,
            "simulated_dbm": -80.0 - errors,
            "error_db": errors,
            "distance_m": 1.0,
        }
    )


# --- pair_errors ---


def test_pair_errors_rows():
    pairs = pair_errors(PART, CELLS, SIMULATED)

    assert list(pairs.columns) == list(PAIR_COLUMNS)
    assert len(pairs) == len(PART)
    assert pairs[["record", "cell"]].equals(PART[["record", "cell"]])


def test_pair_errors_values():
    pairs = pair_errors(PART, CELLS, SIMULATED)

    # -80 dBm measured and -85 dBm simulated give +5 dB.
    np.testing.assert_allclose(pairs["simulated_dbm"], [-85.0, -90.0, -70.0])
    np.testing.assert_allclose(pairs["error_db"], [5.0, 0.0, 0.0])


def test_pair_errors_distance():
    pairs = pair_errors(PART, CELLS, SIMULATED)

    np.testing.assert_allclose(pairs["distance_m"].iloc[0], 5.0)
    np.testing.assert_allclose(pairs["distance_m"].iloc[1], math.hypot(97.0, 4.0))


def test_pair_errors_no_path():
    simulated = SIMULATED.copy()
    simulated[0, 0] = np.nan

    pairs = pair_errors(PART, CELLS, simulated)

    assert np.isnan(pairs["error_db"].iloc[0])
    assert np.isfinite(pairs["error_db"].iloc[1:]).all()


@pytest.mark.parametrize("shape", [(1, 2), (2, 3)])
def test_pair_errors_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        pair_errors(PART, CELLS, np.zeros(shape))


def test_pair_errors_unknown_cell():
    with pytest.raises(ValueError, match="not in"):
        pair_errors(PART, CELLS[:1], SIMULATED[:, :1])


# --- paired_bootstrap ---


def test_paired_bootstrap_identical():
    pairs = pairs_with_errors([1.0, -2.0, 3.0, -4.0])

    assert paired_bootstrap(pairs, pairs, 100, 0.9, seed=0) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_constant_shift():
    first = pairs_with_errors([1.0, -2.0, 3.0, -4.0], records=[0, 0, 1, 2])
    second = pairs_with_errors([3.0, -4.0, 5.0, -6.0], records=[0, 0, 1, 2])

    assert paired_bootstrap(first, second, 100, 0.9, seed=0) == (2.0, 2.0, 2.0)


def test_paired_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(1)
    first = pairs_with_errors(rng.normal(0.0, 5.0, size=40))
    second = pairs_with_errors(rng.normal(0.0, 3.0, size=40))

    result = paired_bootstrap(first, second, 500, 0.9, seed=0)

    assert result.lower <= result.estimate <= result.upper
    assert result.lower < result.upper


def test_paired_bootstrap_ignores_no_path():
    first = pairs_with_errors([1.0, np.nan, 1.0])
    second = pairs_with_errors([2.0, 0.0, np.nan])

    assert paired_bootstrap(first, second, 50, 0.9, seed=0).estimate == pytest.approx(1.0)


def test_paired_bootstrap_seed():
    rng = np.random.default_rng(2)
    first = pairs_with_errors(rng.normal(size=30))
    second = pairs_with_errors(rng.normal(size=30))

    assert paired_bootstrap(first, second, 200, 0.9, seed=3) == paired_bootstrap(
        first, second, 200, 0.9, seed=3
    )


def test_paired_bootstrap_mismatch():
    first = pairs_with_errors([1.0, 2.0])
    second = pairs_with_errors([1.0, 2.0], records=[0, 5])

    with pytest.raises(ValueError, match="same record"):
        paired_bootstrap(first, second, 10, 0.9, seed=0)


def test_paired_bootstrap_no_common_path():
    first = pairs_with_errors([1.0, np.nan])
    second = pairs_with_errors([np.nan, 1.0])

    with pytest.raises(ValueError, match="No pair"):
        paired_bootstrap(first, second, 10, 0.9, seed=0)


@pytest.mark.parametrize(
    ("num_samples", "confidence", "match"),
    [(0, 0.9, "num_samples"), (10, 0.0, "confidence"), (10, 1.0, "confidence")],
)
def test_paired_bootstrap_bad_args(num_samples, confidence, match):
    pairs = pairs_with_errors([1.0])

    with pytest.raises(ValueError, match=match):
        paired_bootstrap(pairs, pairs, num_samples, confidence, seed=0)
