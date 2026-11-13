"""Tests for src.utils."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.utils import (
    COLORS,
    PROJECT_ROOT,
    configure_matplotlib,
    get_device,
    save_figure,
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


def test_configure_matplotlib_report_style():
    with mpl.rc_context():
        configure_matplotlib()
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["savefig.dpi"] == 300
        assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == COLORS


def test_save_figure_writes_png(tmp_path):
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = save_figure(figure, "line", tmp_path / "figures")
    plt.close(figure)

    assert path == tmp_path / "figures" / "line.png"
    assert path.stat().st_size > 0


def test_setup_notebook_outside_ipython():
    with pytest.raises(RuntimeError):
        setup_notebook()
