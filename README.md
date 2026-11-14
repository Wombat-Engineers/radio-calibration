# radio-calibration

A research project that calibrates Sionna RT radio maps against measurements with PyTorch models.

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Related work](#related-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

A radio map gives the channel gain at each cell of a grid around a transmitter.
[Sionna RT](https://nvlabs.github.io/sionna/rt/) computes radio maps with a ray tracer.
The result depends on the scene geometry, the material parameters, and the antenna patterns.
These inputs are approximate, so a simulated radio map differs from a measured radio map.

This project calibrates the simulated radio map with measurements.
The calibration method is "Neural Materials" of Hoydis et al.,
[Learning Radio Environments by Differentiable Ray Tracing](https://arxiv.org/abs/2311.18558).
A neural network gives the material properties at each point of the scene. Gradient descent
changes the network until the simulated RSRP agrees with the measured RSRP.

### Capabilities

The repository contains these parts:

- A locked Python environment with PyTorch (CUDA 13.4), Sionna RT, Mitsuba, and Dr.Jit.
- A scenario generator. It puts 3 sites with 3 cells each in a Sionna RT scene and writes
  synthetic MDT (minimization of drive tests) records. See [Scenario data](#scenario-data).
- Sionna RT functions that load a scene, measure the building heights, and calculate RSRP.
- A radio material for Sionna RT that gets its properties from a neural network
  (`NeuralRadioMaterial`). A training loop calibrates this network against the MDT records.
  See [Material calibration](#material-calibration).
- Shared utilities for paths, figure style, random seeds, device selection, and notebook setup.
- Tests for each package, with Sionna RT on the CUDA backend.
- A notebook that examines the scenario data.
- A notebook for the exploratory data analysis (EDA) of the MDT records.
- Validation rules for `cells.csv` and `mdt.csv`, and a notebook that validates the data.
  The notebook divides the records into train, validation, and test parts and writes them to
  `data/processed/` in the Parquet format. See
  [Exploratory analysis and preprocessing](#exploratory-analysis-and-preprocessing).
- A notebook that calibrates the materials and compares the result with the ITU materials
  of the scene.
- One empty notebook for the evaluation.

### Non-goals

_Not documented yet._

## Architecture

The first diagram shows the data flow of the scenario generator.
[Material calibration](#material-calibration) shows the data flow of the calibration.

```mermaid
flowchart LR
    sim[configs/simulation.yaml] --> cfg[configs/scenario.yaml]
    cfg --> gen[src.scenario.generate.main]
    gen --> scene[src.simulation.load_rt_scene]
    scene --> hm[src.simulation.building_height_map]
    hm --> sites[src.scenario.sites.place_sites]
    hm --> mdt[src.scenario.mdt.sample_mdt]
    sites --> mdt
    mdt -- positions --> rsrp[src.simulation.compute_rsrp]
    rsrp -- RSRP --> mdt
    sites --> cells[(data/raw/cells.csv)]
    mdt --> out[(data/raw/mdt.csv)]
```

### Components

| Component | Responsibility | Location |
|---|---|---|
| Core | The `Cell` record, the format of `cells.csv`, and the column names of `mdt.csv` | [`src/core/`](src/core/) |
| Data | Validation rules of `cells.csv` and `mdt.csv`, the long format of `data/processed/`, and the division into train, validation, and test parts. It has no Sionna RT import. | [`src/data/`](src/data/) |
| Simulation | All Sionna RT calls: scene load, building heights, RSRP from the path solver, and the neural radio material | [`src/simulation/`](src/simulation/) |
| Calibration | Loss, power scale, training loop, and file format of the calibration result. It uses Dr.Jit and has no Sionna RT import. | [`src/calibration/`](src/calibration/) |
| Scenario | Site placement, traffic model, MDT records. It has no Sionna RT import. | [`src/scenario/`](src/scenario/) |
| Utilities | Paths, figure style, random seeds, device selection, notebook setup | [`src/utils/`](src/utils/) |
| Configuration | Hydra configuration files | [`configs/`](configs/) |
| Notebooks | Exploration, preprocessing, material calibration, evaluation | [`notebooks/`](notebooks/) |
| Tests | Checks of the environment and of the utilities | [`tests/`](tests/) |

### External dependencies

| Dependency | Purpose | Criticality |
|---|---|---|
| [Sionna RT](https://github.com/NVlabs/sionna-rt) 2.2.0 | Ray tracer that computes radio maps | Critical |
| [Mitsuba](https://github.com/mitsuba-renderer/mitsuba3) 3.9.1 | Renderer below Sionna RT. Sionna RT pins this version. | Critical |
| [Dr.Jit](https://github.com/mitsuba-renderer/drjit) 1.5.0 | Just-in-time compiler below Mitsuba. Sionna RT pins this version. The calibration uses its automatic differentiation, its neural network module, and its AdamW optimizer. | Critical |
| [PyTorch](https://pytorch.org/) 2.15.0+cu134 | Calibration models and training | Critical |
| [Hydra](https://hydra.cc/) | Experiment configuration | Critical |

[`uv.lock`](uv.lock) records the exact version of each package.

## Getting started

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.13 | `uv` installs it from [`.python-version`](.python-version). |
| [uv](https://docs.astral.sh/uv/) | >=0.11.30, <0.12 | The build backend `uv_build` sets this range. |
| NVIDIA GPU and driver | A driver that supports CUDA 13.4 | The environment was tested on Windows 11 with driver 610.88 and an RTX 3050 Ti (4 GB). |
| [Task](https://taskfile.dev/) | 3.52 or later | Optional. It gives short names to the commands below. |

### Install

```bash
git clone https://github.com/Wombat-Engineers/radio-calibration.git
cd radio-calibration
uv sync
uv run pre-commit install
```

`uv sync` installs PyTorch from the PyTorch package index for CUDA 13.4.
On Windows, PyPI has only the CPU build of PyTorch.

### Run

```bash
uv run jupyter lab
```

### Verify

```bash
uv run pytest
```

Expected last line of the output:

```
============================= 82 passed in 10.91s =============================
```

The time changes from run to run.
The tests for Mitsuba and Sionna RT use the CUDA variant `cuda_ad_mono_polarized`.
They fail if CUDA is not available. They do not use the CPU variant.

Dr.Jit can print `jitc_llvm_init(): LLVM API initialization failed` when LLVM is not installed.
The CUDA variants operate without LLVM.

## Configuration

[Hydra](https://hydra.cc/) reads the configuration from [`configs/`](configs/).

[`configs/config.yaml`](configs/config.yaml) is the root configuration. It loads each other
file of `configs/` into its own key:

| Key | File | Content |
|---|---|---|
| `scenario` | [`configs/scenario.yaml`](configs/scenario.yaml) | The synthetic scenario |
| `simulation` | [`configs/simulation.yaml`](configs/simulation.yaml) | Sionna RT |
| `calibration` | [`configs/calibration.yaml`](configs/calibration.yaml) | The data split and the material calibration |

It also has these values:

| Name | Type | Default | Description |
|---|---|---|---|
| `seed` | integer | `42` | The only seed: scenario generator, path solver (`simulation.solver.seed` is `${seed}`), data split, network, and calibration. |
| `output.figures.dir` | string | `reports/figures` | Directory of the figures (PNG) of the notebooks. A relative path starts at the repository root. |
| `output.tables.dir` | string | `reports/tables` | Directory of the tables (CSV) of the notebooks. |
| `hydra.run.dir` | string | `outputs/scenario/<date>/<time>` | Run directory of `task scenario`. Hydra writes the configuration of the run (`.hydra/config.yaml`, `.hydra/overrides.yaml`) and the log there. |

`task scenario` and `src.utils.load_config` compose this file in the same way. Each file has a
comment for each value. No other file holds a copy of the configuration.

[`configs/simulation.yaml`](configs/simulation.yaml) is the configuration of Sionna RT.
These are the primary values:

| Name | Type | Default | Description |
|---|---|---|---|
| `scene.name` | string | `munich` | A scene that Sionna RT supplies, or a file `<name>.xml` in `data/external/scenes`. |
| `radio.frequency_hz` | number | `2.6e9` | Carrier frequency. |
| `radio.subcarrier_spacing_hz` | number | `30e3` | Bandwidth of the scene: one resource element. |
| `receiver` | mapping | 1 x 1 dipole, V | Planar array of the UE. |
| `solver.max_depth` | integer | `5` | Maximum number of interactions of a path. |
| `solver.samples_per_src` | integer | `100000` | Number of rays for each source. |
| `solver.max_num_paths_per_src` | integer | `50000` | Size of the path buffer for each source. |
| `solver.refraction` | boolean | `true` | With `true`, a signal can go through a wall. With `false`, an indoor position gets no signal. |
| `solver.diffraction` | boolean | `true` | Diffraction at building edges. |
| `deterministic` | boolean | `true` | Argument of the `PathSolver` constructor. With `true`, two runs with the same seed give the same paths. Set `false` if the GPU runs out of memory. |
| `rx_batch_size` | integer | `10` | Number of UE positions in one solver call. |

The `solver` block has all arguments of a call of the Sionna RT `PathSolver`.
The code gives the block to the solver without a change.

[`configs/scenario.yaml`](configs/scenario.yaml) is the configuration of the test scenario.
These are the primary values:

| Name | Type | Default | Description |
|---|---|---|---|
| `grid.tile_size_m` | number | `5` | Side of a tile of the grid for the height map, in meters. |
| `grid.min_building_height_m` | number | `2` | A grid tile above this height is a building. |
| `antenna.power_rs` | number | `4.85` | Reference-signal power for each resource element, in dBm. |
| `antenna.transmitter` | mapping | 8 x 8 tr38901, cross | Planar array of each cell. The generator writes it to `cells.csv`. |
| `sites.isd_m` | number | `500` | Side of the triangle of sites, in meters. |
| `sites.mast_height_m` | number | `25` | Antenna height above the ground, in meters. |
| `sites.downtilt_deg` | number | `5` | Downtilt of each cell, in degrees. |
| `traffic.num_days` | integer | `1` | Number of days of MDT records. |
| `traffic.records_per_day` | number | `1000` | Mean number of MDT records in one day. |
| `ue_height.mean_m` | number | `1.5` | Mean UE height above the ground, in meters. |
| `ue_height.min_m`, `ue_height.max_m` | number | `0.5`, `2.5` | Limits of the UE height, in meters. |
| `measurement.rsrp_noise_mean_db` | number | `-5` | Mean of the RSRP measurement error, in dB. |
| `measurement.min_rsrp_dbm` | number | `-140` | A cell below this RSRP is not in the record. |

An override names the key of the file, for example `scenario.sites.isd_m=600`,
`simulation.solver.max_depth=5`, or `seed=1`.

Precedence: command-line overrides, then the YAML files.

The project reads no environment variables and holds no secrets.

## Usage

Call `setup_notebook` in the first cell of each notebook.
The function enables module autoreload, applies the figure style, seeds the random generators
with `seed`, and sets the output directories from `output`.
It returns the device that tensors must use.

```python
from src.utils import DATA_DIR, load_config, save_figure, save_table, setup_notebook

cfg = load_config()
device = setup_notebook(cfg)
print(device)  # cuda
print(DATA_DIR)  # <repository root>/data
```

`save_figure(figure, "name")` writes `name.png` to `output.figures.dir`.
`save_table(table, "name")` writes `name.csv` to `output.tables.dir`, with the index. It accepts a
DataFrame or a styled DataFrame and returns its argument, thus it can wrap the last expression of
a cell: `save_table(frame.style.set_caption("Table 1. ..."), "name")`.

### Scenario data

Generate the data for the experiments:

```bash
task scenario
```

Without Task, use `uv run python -m src.scenario`. Put Hydra overrides after the command,
for example `task scenario -- scenario.traffic.records_per_day=200 simulation.solver.max_depth=5`.
The command prints the time of the RSRP calculation and the total time.

The command writes two files to `data/raw/`. Each run replaces the files of the last run.
Hydra keeps the configuration and the log of each run in `outputs/scenario/<date>/<time>/`.
The notebooks read `configs/`, thus run `task scenario` again after a change of `configs/`.

| File | Content |
|---|---|
| `cells.csv` | One row for each cell: `cell`, `site`, `x`, `y`, `z`, `azimuth_deg`, `downtilt_deg`, `power_rs_dbm`, and the array columns `num_rows`, `num_cols`, `vertical_spacing`, `horizontal_spacing`, `pattern`, `polarization`. |
| `mdt.csv` | One row for each MDT record: `timestamp`, `x`, `y`, `z`, and one `rsrp_<cell>` column for each cell. An empty value shows that the UE did not receive the cell. [`src/core/mdt.py`](src/core/mdt.py) sets the column names. |

Positions are in meters, in the coordinate system of the scene.

The generator does these steps:

1. It measures the height of the scene on a grid. A grid tile above 2 m is a building.
2. It puts 3 sites at the corners of an equilateral triangle, 25 m above the ground.
   Each site has 3 cells. A corner in a building moves to the nearest outdoor grid tile.
3. It draws the number of records of each 15-minute slot from a Poisson distribution.
   Each record is independent of the other records.
4. It draws an outdoor position for each record. The probability increases with the
   building density. A residential part and a business part have different daily patterns,
   as in [Wang et al., 2015](https://arxiv.org/abs/1510.04026). Thus the positions change
   during the day.
5. It calculates the RSRP of each cell at each position with the Sionna RT path solver.
6. It adds a measurement error to the RSRP and a GNSS error to the position. It rounds the
   RSRP to 1 dB.
7. It samples a record again if the position is in a building or if no cell has a signal.

The notebook [`notebooks/00_scenario.ipynb`](notebooks/00_scenario.ipynb) examines the scene,
the cells, and each step of the MDT generation. Run all its cells with this command:

```bash
task notebook
```

Without Task, use
`uv run jupyter nbconvert --execute --to notebook --output-dir outputs/notebooks notebooks/00_scenario.ipynb`.
The copy with the outputs is in `outputs/notebooks/`.
The notebook also writes each figure as a PNG file with 300 dpi to `reports/figures/`, and each
table as a CSV file to `reports/tables/`. Git ignores `reports/`.

To use real data, write `cells.csv` and `mdt.csv` with the same columns.
A converter from latitude and longitude to scene coordinates is not implemented.

### Exploratory analysis and preprocessing

Two notebooks read the files in `data/raw/`. Run `task scenario` first.

```mermaid
flowchart LR
    raw[(data/raw)] --> eda[notebooks/01_eda.ipynb]
    scene[src.simulation.building_height_map] --> eda
    raw --> pre[notebooks/02_preprocessing.ipynb]
    pre --> val[src.data.validate_cells, src.data.validate_mdt]
    val -- all counts 0 --> long[src.data.mdt_to_long]
    val -- a count above 0 --> stop[AssertionError: no file]
    split_cfg[configs/calibration.yaml] --> split[src.data.split_records]
    long --> split
    split --> out[(data/processed)]
```

[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) is the exploratory data analysis. It has
10 sections: setup, load data, schema and dtypes, missingness, univariate distributions,
target analysis, group structure, relationships, temporal and spatial structure, and findings.
The target is the reported RSRP of one (record, cell) pair. The notebook loads the Sionna RT
scene for the building height map, thus it needs an NVIDIA GPU. It writes its figures with
the prefix `eda_` to `reports/figures/`.

```bash
task notebook -- notebooks/01_eda.ipynb
```

[`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb) validates the data and
writes it to `data/processed/`. It does not change a value. `validate_cells` and
`validate_mdt` count the rows that break each rule. If a count is not zero, the notebook stops
and writes no file. `split_records` then divides the records at random into the train,
validation, and test parts. `calibration.split.fractions` and `seed` set the division. All rows of one
record go to the same part. The notebook does not need a GPU.

```bash
task notebook -- notebooks/02_preprocessing.ipynb
```

| File | Content |
|---|---|
| `data/processed/train.parquet` | The train records. One row for each received (record, cell) pair: `record`, `timestamp`, `x`, `y`, `z`, `cell`, `rsrp_dbm`. `record` is the row index in `mdt.csv`, from 0. A missing RSRP gives no row. |
| `data/processed/validation.parquet` | The validation records, in the same format |
| `data/processed/test.parquet` | The test records, in the same format |
| `data/processed/cells.parquet` | The content of `cells.csv` |

The notebook does not write `data/processed/mdt.parquet`. A file with this name is from an
older version of the notebook.

### Material calibration

[`notebooks/03_material_calibration.ipynb`](notebooks/03_material_calibration.ipynb) explains
and runs the calibration. Run `task scenario` and `notebooks/02_preprocessing.ipynb` first.

```mermaid
flowchart LR
    net[src.simulation.make_material_network] --> mat[src.simulation.NeuralRadioMaterial]
    mat -- apply_neural_material --> scene[Sionna RT scene]
    scene --> rsrp[src.simulation.compute_rsrp_mw]
    train[(data/processed/train.parquet)] --> scale[src.calibration.scale_estimate]
    rsrp --> scale
    scale --> loss[src.calibration.smape]
    rsrp --> loss
    loss -- gradient --> opt[drjit.opt.AdamW]
    opt -- new weights --> net
```

The parts of the method are these:

- **Material.** `NeuralRadioMaterial` is a subclass of the Sionna RT `RadioMaterial`. One
  multilayer perceptron gives the relative permittivity, the conductivity, the scattering
  coefficient, the XPD coefficient, and the thickness at the point of each ray interaction.
  All objects of the scene get the same material. The file
  [`src/simulation/neural_material.py`](src/simulation/neural_material.py) imports only
  Sionna RT, Mitsuba, Dr.Jit, and NumPy.
- **Loss.** The loss is the power term of the loss of the paper: the symmetric absolute
  percentage error between the measured RSRP and the simulated RSRP, in linear power. The
  MDT records have no delay spread, thus the delay spread term of the paper is not used.
- **Power scale.** A scale multiplies the measured power. Each step estimates it in closed
  form, and an exponential moving average smooths it.
- **Optimizer.** AdamW from Dr.Jit. The path solver runs with the loop mode `evaluated`,
  because Sionna RT gives no gradients in the loop mode `symbolic`.

[`configs/calibration.yaml`](configs/calibration.yaml) sets the network, the optimizer, the
number of steps, and the batch size. The Sionna RT configuration stays in
[`configs/simulation.yaml`](configs/simulation.yaml).

```bash
task notebook -- notebooks/03_material_calibration.ipynb
```

The notebook writes these files to `outputs/calibration/`:

| File | Content |
|---|---|
| `calibration.npz` | The weights of the network, the network structure, the normalization of the positions, and the power scale. `src.calibration.load_calibration` reads it and gives a `NeuralRadioMaterial` and the scale. |
| `history.csv` | The loss, the power scale, and the validation loss at each step |

The simulated RSRP is the same in each run, bit for bit. The calibration is not: Dr.Jit adds
the gradients on the GPU with atomic float operations, in an order that changes from run to run.
Small differences then grow during the gradient steps, thus notebook 03 gives slightly different
results in each run. Report the result of more than one run with its spread.

The values in use in [`configs/calibration.yaml`](configs/calibration.yaml) and
[`configs/simulation.yaml`](configs/simulation.yaml) are small. They let the calibration run on
a GPU with 4 GB of memory. A comment `Recommended` beside a value gives the value for a GPU
with more memory. The recommended network size is that of the paper.

With the recommended values, one gradient step did not complete in 10 minutes on the 4 GB
development GPU. The largest part of the memory is the path buffer: Sionna RT allocates it for
`solver.max_num_paths_per_src` paths before it traces the rays.

The scenario generator uses the same solver values. After a change of a solver value, run
`task scenario` and `notebooks/02_preprocessing.ipynb` again before the calibration. The tests
use a small scene and a small network.

## Development

### Layout

```
radio-calibration/
├── configs/          Hydra configuration
├── notebooks/        Jupyter notebooks, numbered in the order of the workflow
├── src/              Python package (import name: src)
│   ├── calibration/  Loss, power scale, training loop, result file
│   ├── core/         Cell record, cells.csv format, mdt.csv column names
│   ├── data/         Validation rules, long format and parts of data/processed
│   ├── scenario/     Scenario generator: sites, traffic, MDT records
│   ├── simulation/   Sionna RT calls: scene, building heights, RSRP, neural material
│   └── utils/        Paths, figure style, seeds, device, notebook setup
├── tests/            pytest tests
├── Taskfile.yml      Task definitions
├── pyproject.toml    Dependencies and tool configuration
└── uv.lock           Locked dependency versions
```

Git ignores `data/` and `outputs/`. Put data sets in `data/`. Hydra writes the run directories of
`task scenario` to `outputs/scenario/`.

### Standards

| | |
|---|---|
| **Style and lint** | [Ruff](https://docs.astral.sh/ruff/), enforced by pre-commit |
| **Notebooks** | `nbstripout` removes cell outputs before each commit |
| **Dependencies** | Change them with `uv add` or `uv remove`. Do not edit `uv.lock` by hand. |
| **Commits** | _Not documented yet._ |
| **Branching** | _Not documented yet._ |

If you add `torchvision` or `torchaudio`, add the package to `[tool.uv.sources]` in `pyproject.toml`.
Without that line, `uv` installs the CPU build from PyPI.

### Local loop

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

`Taskfile.yml` gives the same commands short names. Run `task --list` to see them.

## Testing

| Tier | Scope | Command | Where it runs |
|---|---|---|---|
| Environment | Each main library runs: PyTorch on CUDA, Mitsuba and Sionna RT on CUDA, pandas with PyArrow, matplotlib, Hydra, scikit-learn, SciPy | `uv run pytest tests/test_environment.py` | Local machine with an NVIDIA GPU |
| Unit | `src/utils` | `uv run pytest tests/test_utils.py` | Local machine |
| Unit | `src/core` | `uv run pytest tests/test_core.py` | Local machine |
| Unit | `src/data` | `uv run pytest tests/test_data.py` | Local machine |
| Unit | `src/simulation`, with the scene `simple_street_canyon` | `uv run pytest tests/test_simulation.py` | Local machine with an NVIDIA GPU |
| Unit | `src/scenario`. One test runs the full generator on a small scene. | `uv run pytest tests/test_scenario.py` | Local machine with an NVIDIA GPU |
| Unit | `src/calibration`, with the scene `simple_street_canyon` and a small network | `uv run pytest tests/test_calibration.py` | Local machine with an NVIDIA GPU |

Run one test:

```bash
uv run pytest tests/test_environment.py::test_torch_autograd
```

Continuous integration and a coverage floor are not set.

## Related work

| Topic | Work |
|---|---|
| Ray tracer | Hoydis et al., [Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling](https://arxiv.org/abs/2303.11103), 2023. The paper shows that gradient descent can learn radio materials. |
| Ray tracer | Aït Aoudia et al., [Sionna RT: Technical Report](https://arxiv.org/abs/2504.21719), 2025. The report describes the path solver and the radio map solver. |
| Calibration by gradient descent | Hoydis et al., [Learning Radio Environments by Differentiable Ray Tracing](https://arxiv.org/abs/2311.18558), IEEE TMLCN, 2024. The method calibrates material properties, scattering, and antenna patterns against channel measurements. |
| Calibration with phase errors | Ruah et al., [Calibrating Wireless Ray Tracing for Digital Twinning using Local Phase Error Estimates](https://arxiv.org/abs/2312.12625), 2023. The method models the phase errors that geometry errors cause. |
| Calibration with signal strength | Chmela and Tripkovic, [Material Parameter Estimation for Sionna RT](https://doi.org/10.48436/ycw2e-xyw13), TU Wien, 2025. The code changes the material parameters to decrease the mean absolute error between simulated and measured signal strength. |
| Calibration with few measurements | An et al., [Taming Vision Priors for Data Efficient mmWave Channel Modeling](https://arxiv.org/abs/2603.13383), 2026. A vision-language model gives initial material properties. A few dozen channel measurements then calibrate the ray tracer. |
| Radio maps from deep learning | Li et al., [Geo2SigMap: High-Fidelity RF Signal Mapping Using Geographic Databases](https://arxiv.org/abs/2312.14303), 2023. A cascaded U-Net uses Sionna ray tracing and sparse measurements to predict signal maps. |
| Radio maps from deep learning | Saeizadeh et al., [AIRMap: AI-Generated Radio Maps for Wireless Digital Twins](https://arxiv.org/abs/2511.05522), 2025. A U-Net predicts path gain. Field measurements then calibrate the prediction. |
| Neural surrogates | Orekondy et al., [WiNeRT: Towards Neural Ray Tracing for Wireless Channel Modelling and Differentiable Simulations](https://iclr.cc/virtual/2023/poster/10694), ICLR 2023. |
| Neural surrogates | Hehn et al., [Differentiable and Learnable Wireless Simulation with Geometric Transformers](https://mlanthology.org/iclr/2025/hehn2025iclr-differentiable), ICLR 2025. |
| Review | Ruah et al., [How to Bridge the Sim-to-Real Gap in Digital Twin-Aided Telecommunication Networks](https://arxiv.org/abs/2507.07067), 2025. |

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

This project uses [Sionna RT](https://github.com/NVlabs/sionna-rt), [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3), and [Dr.Jit](https://github.com/mitsuba-renderer/drjit).
