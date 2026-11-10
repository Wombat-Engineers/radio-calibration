"""Tests for src.utils."""

import matplotlib as mpl
import numpy as np
import pytest
import torch

from src.utils import (
    PROJECT_ROOT,
    configure_matplotlib,
    get_device,
    seed_everything,
    setup_notebook,
)


def test_seed_everything_repeatable():
    seed_everything(0)
    first = (torch.rand(3), np.random.rand(3))
    seed_everything(0)
    second = (torch.rand(3), np.random.rand(3))

    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_get_device_matches_cuda():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device().type == expected


def test_paths_point_to_project():
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_configure_matplotlib_sets_rc():
    with mpl.rc_context():
        configure_matplotlib()
        assert mpl.rcParams["savefig.bbox"] == "tight"


def test_setup_notebook_outside_ipython():
    with pytest.raises(RuntimeError):
        setup_notebook()
