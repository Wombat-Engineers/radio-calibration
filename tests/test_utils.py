"""Tests for utils."""

import json
import sys
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

import utils.output as output_module
import utils.paths as paths_module
from utils import (
    COLORS,
    CONFIG_DIR,
    DATA_DIR,
    NOTEBOOK_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    calibration_settings,
    configure_matplotlib,
    configure_output,
    get_device,
    load_config,
    save_figure,
    save_table,
    scenario_dirs,
    scene_source,
    seed_everything,
    setup_notebook,
    simulation_settings,
)
from utils.runs import (
    environment_info,
    notebook_command,
    result_paths,
    run_steps,
    select_notebooks,
    write_archive,
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

    assert output.figures.dir == "lab/reports/munich_map_errors/figures"
    assert output.tables.dir == "lab/reports/munich_map_errors/tables"


def test_setup_notebook_outside_ipython():
    with pytest.raises(RuntimeError):
        setup_notebook(load_config())


def test_config_dir_points_to_configs():
    assert CONFIG_DIR == PROJECT_ROOT / "lab" / "configs"
    assert (CONFIG_DIR / "config.yaml").is_file()


def test_data_and_outputs_in_lab():
    assert DATA_DIR == PROJECT_ROOT / "lab" / "data"
    assert OUTPUTS_DIR == PROJECT_ROOT / "lab" / "outputs"


def test_simulation_settings_resolves_seed():
    cfg = load_config()
    settings = simulation_settings(cfg)

    assert type(settings.radio_map) is dict
    assert settings.radio_map["seed"] == cfg.seed
    assert settings.frequency_hz == cfg.simulation.radio.frequency_hz
    assert settings.radio_map_tx_batch_size == cfg.simulation.radio_map_tx_batch_size


def test_calibration_settings_from_config():
    cfg = load_config()
    settings = calibration_settings(cfg)

    assert settings.base_channels == cfg.calibration.network.base_channels
    assert settings.depth == cfg.calibration.network.depth
    assert settings.pretraining.num_epochs == cfg.calibration.pretraining.num_epochs


def test_scene_source_builtin(monkeypatch, tmp_path):
    monkeypatch.setattr(paths_module, "SCENES_DIR", tmp_path)

    assert scene_source(load_config("scenario.scene=munich")) == "munich"


def test_scene_source_local_file(monkeypatch, tmp_path):
    monkeypatch.setattr(paths_module, "SCENES_DIR", tmp_path)
    (tmp_path / "site.xml").write_text("<scene/>")

    assert scene_source(load_config("scenario.scene=site")) == tmp_path / "site.xml"


def test_scene_source_bad_name():
    with pytest.raises(ValueError, match="letters"):
        scene_source(load_config("scenario.scene=../munich"))


def test_load_config_branches():
    cfg = load_config()

    assert {"seed", "scenario", "simulation", "calibration"} <= set(cfg)
    assert "hydra" not in cfg


def test_load_config_one_seed():
    cfg = load_config()

    assert cfg.seed == 42
    assert cfg.simulation.radio_map.seed == 42


def test_load_config_override():
    assert load_config("scenario.sites.isd_m=60").scenario.sites.isd_m == 60


def test_load_config_unknown_key():
    with pytest.raises(ConfigCompositionException):
        load_config("scenario.nope=1")


def test_load_config_evaluation():
    evaluation = load_config().evaluation

    assert all(0 <= quantile <= 1 for quantile in evaluation.quantiles)
    assert evaluation.num_bins > 0


# --- runs ---


def test_scenario_dirs():
    dirs = scenario_dirs("munich_map_errors")

    assert dirs.raw == DATA_DIR / "munich_map_errors" / "raw"
    assert dirs.processed == DATA_DIR / "munich_map_errors" / "processed"
    assert dirs.outputs == OUTPUTS_DIR / "munich_map_errors"
    assert dirs.notebooks == NOTEBOOK_DIR / "munich_map_errors"


@pytest.mark.parametrize("name", ["../x", "a/b", ""])
def test_scenario_dirs_bad_name(name):
    with pytest.raises(ValueError, match="letters"):
        scenario_dirs(name)


def test_result_paths():
    dirs = scenario_dirs("munich_new_site")

    assert result_paths("munich_new_site") == (dirs.raw, dirs.processed, dirs.outputs, CONFIG_DIR)


def test_select_notebooks_default():
    notebooks = select_notebooks("munich_map_errors", [])

    assert notebooks == sorted(notebooks)
    assert "05_evaluation.ipynb" in [path.name for path in notebooks]


def test_select_notebooks_given():
    name = "lab/notebooks/munich_map_errors/05_evaluation.ipynb"

    assert select_notebooks("munich_map_errors", [name]) == [(PROJECT_ROOT / name).resolve()]


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "lab/notebooks/munich_map_errors/../README.md",
        "../x.ipynb",
        "lab/notebooks/munich_map_errors/nosuch.ipynb",
        "lab/notebooks/munich_new_site/05_evaluation.ipynb",
    ],
)
def test_select_notebooks_outside(name):
    with pytest.raises(ValueError, match="not a notebook"):
        select_notebooks("munich_map_errors", [name])


def test_notebook_command_output_dir():
    command = notebook_command(NOTEBOOK_DIR / "munich_ablation" / "05_evaluation.ipynb")

    assert command[:4] == [sys.executable, "-m", "jupyter", "nbconvert"]
    assert command[command.index("--output-dir") + 1] == str(
        OUTPUTS_DIR / "munich_ablation" / "notebooks"
    )


def test_run_steps_stops_at_failure():
    steps = run_steps(
        [
            [sys.executable, "-c", "pass"],
            [sys.executable, "-c", "raise SystemExit(3)"],
            [sys.executable, "-c", "pass"],
        ]
    )

    assert [step.returncode for step in steps] == [0, 3]
    assert all(step.seconds >= 0 for step in steps)


def test_write_archive(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("a")
    (root / "sub" / "b.txt").write_text("b")

    archive = write_archive(
        tmp_path / "archives",
        "run1",
        root,
        [root / "a.txt", root / "sub", root / "missing"],
        {"manifest.json": "{}"},
    )

    assert archive == tmp_path / "archives" / "run1.zip"
    with zipfile.ZipFile(archive) as package:
        assert sorted(package.namelist()) == ["run1/a.txt", "run1/manifest.json", "run1/sub/b.txt"]
        assert package.read("run1/sub/b.txt") == b"b"


@pytest.mark.parametrize("name", ["../x", "a/b", ""])
def test_write_archive_bad_name(tmp_path, name):
    with pytest.raises(ValueError, match="not permitted"):
        write_archive(tmp_path, name, tmp_path, [], {})


def test_write_archive_path_outside_root(tmp_path):
    (tmp_path / "outside.txt").write_text("x")
    (tmp_path / "root").mkdir()

    with pytest.raises(ValueError):
        write_archive(tmp_path, "run1", tmp_path / "root", [tmp_path / "outside.txt"], {})


def test_environment_info_keys():
    info = environment_info()

    assert set(info) == {"python", "platform", "git_commit", "git_dirty", "packages", "gpus"}
    assert set(info["packages"]) >= {"torch", "drjit", "mitsuba", "sionna-rt"}
    json.dumps(info)
