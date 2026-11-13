"""Checks that the main libraries run in this environment, not only that they import."""

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression

from src.utils import PROJECT_ROOT, get_device

# Sionna RT needs a mono, polarized Mitsuba variant. The tests use CUDA only:
# a CUDA failure must fail the test, not fall back to the LLVM (CPU) variant.
CUDA_VARIANT = "cuda_ad_mono_polarized"


def test_torch_cuda_tensor_op():
    assert torch.cuda.is_available()
    device = get_device()
    total = torch.tensor([1.0, 2.0], device=device) + torch.tensor([3.0, 4.0], device=device)

    assert total.device.type == "cuda"
    assert total.tolist() == [4.0, 6.0]


def test_torch_autograd():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x**2).sum().backward()

    assert x.grad.tolist() == [2.0, 4.0, 6.0]


def test_drjit_mitsuba_cuda():
    mi.set_variant(CUDA_VARIANT)
    total = mi.Float([1.0, 2.0]) + mi.Float([3.0, 4.0])

    assert dr.backend_v(total) == dr.JitBackend.CUDA
    assert total.numpy().tolist() == [4.0, 6.0]


def test_sionna_rt_radio_map_cuda():
    mi.set_variant(CUDA_VARIANT)
    from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene

    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(Transmitter("tx", position=[0.0, 0.0, 10.0]))

    # A 20 m x 20 m plane with 5 m cells gives a 4 x 4 map. The sample count is small
    # because the GPU of the development machine has 4 GB of memory.
    radio_map = RadioMapSolver()(
        scene,
        center=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0, 0.0],
        size=[20.0, 20.0],
        cell_size=[5.0, 5.0],
        samples_per_tx=10_000,
        max_depth=1,
    )
    path_gain = radio_map.path_gain

    assert dr.backend_v(path_gain) == dr.JitBackend.CUDA
    assert path_gain.shape == (1, 4, 4)
    values = path_gain.numpy()
    assert np.isfinite(values).all()
    assert (values > 0).any()


def test_numpy_pandas_pyarrow(tmp_path):
    frame = pd.DataFrame({"x": np.arange(3), "rss_dbm": [-70.5, -80.0, -91.25]})
    path = tmp_path / "frame.parquet"
    frame.to_parquet(path, engine="pyarrow")

    pd.testing.assert_frame_equal(pd.read_parquet(path, engine="pyarrow"), frame)


def test_matplotlib_headless(tmp_path):
    path = tmp_path / "map.png"
    fig, ax = plt.subplots()
    ax.imshow(np.arange(16).reshape(4, 4))
    fig.savefig(path)
    plt.close(fig)

    assert path.stat().st_size > 0


def test_hydra_compose():
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config")

    assert isinstance(cfg, DictConfig)
    assert cfg.seed == 0


def test_sklearn_scipy():
    x = np.arange(5, dtype=float)
    model = LinearRegression().fit(x.reshape(-1, 1), 2 * x)
    assert np.isclose(model.coef_[0], 2.0)

    interpolate = RegularGridInterpolator((x,), 2 * x)
    assert np.isclose(interpolate([1.5])[0], 3.0)
