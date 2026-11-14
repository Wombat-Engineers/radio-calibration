"""Tests for src.utils."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

import src.utils.output as output_module
from src.utils import (
    COLORS,
    CONFIG_DIR,
    PROJECT_ROOT,
    configure_matplotlib,
    configure_output,
    get_device,
    load_config,
    save_figure,
    save_table,
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


@pytest.fixture
def output_dirs(tmp_path):
    """Configure the output directories in tmp_path. Remove the configuration after the test."""
    configure_output(
        OmegaConf.create(
            {
                "figures": {"dir": str(tmp_path / "figures")},
                "tables": {"dir": str(tmp_path / "tables")},
            }
        )
    )
    yield tmp_path
    output_module._directories.clear()


@pytest.fixture
def no_output(monkeypatch):
    monkeypatch.setattr(output_module, "_directories", {})


def line_figure():
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return figure


def test_save_figure_writes_png(output_dirs):
    figure = line_figure()
    path = save_figure(figure, "line")
    plt.close(figure)

    assert path == output_dirs / "figures" / "line.png"
    assert path.stat().st_size > 0


def test_save_figure_not_configured(no_output):
    figure = line_figure()
    with pytest.raises(RuntimeError, match="configure_output"):
        save_figure(figure, "line")
    plt.close(figure)


def test_save_table_frame_csv(output_dirs):
    table = pd.DataFrame({"Value": [1.5, 2.0]}, index=["a", "b"])

    returned = save_table(table, "values")

    assert returned is table
    saved = pd.read_csv(output_dirs / "tables" / "values.csv", index_col=0)
    pd.testing.assert_frame_equal(saved, table)


def test_save_table_styler_returns_input(output_dirs):
    table = pd.DataFrame({"Value": [1.5, 2.0]}, index=["a", "b"])
    styler = table.style.set_caption("Table 1. Values.")

    assert save_table(styler, "styled") is styler
    saved = pd.read_csv(output_dirs / "tables" / "styled.csv", index_col=0)
    pd.testing.assert_frame_equal(saved, table)


def test_save_table_not_configured(no_output):
    with pytest.raises(RuntimeError, match="configure_output"):
        save_table(pd.DataFrame({"Value": [1.0]}), "values")


def test_load_config_output():
    output = load_config().output

    assert output.figures.dir == "reports/figures"
    assert output.tables.dir == "reports/tables"


def test_setup_notebook_outside_ipython():
    with pytest.raises(RuntimeError):
        setup_notebook(load_config())


def test_config_dir_points_to_configs():
    assert CONFIG_DIR == PROJECT_ROOT / "configs"
    assert (CONFIG_DIR / "config.yaml").is_file()


def test_load_config_branches():
    cfg = load_config()

    assert {"seed", "scenario", "simulation", "calibration"} <= set(cfg)
    assert "hydra" not in cfg


def test_load_config_one_seed():
    cfg = load_config()

    assert cfg.seed == 42
    assert cfg.simulation.solver.seed == 42


def test_load_config_override():
    assert load_config("scenario.sites.isd_m=60").scenario.sites.isd_m == 60


def test_load_config_unknown_key():
    with pytest.raises(ConfigCompositionException):
        load_config("scenario.nope=1")
