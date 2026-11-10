"""Notebook setup."""

import torch
from IPython import get_ipython

from src.utils.plotting import configure_matplotlib
from src.utils.reproducibility import get_device, seed_everything


def setup_notebook(seed: int = 0) -> torch.device:
    """Prepare a notebook session. Call this in the first cell.

    Enables autoreload of edited modules, applies the project figure style,
    and seeds the random generators.

    Args:
        seed: Seed for the Python, NumPy, and PyTorch random generators.

    Returns:
        The device that models and tensors must use.

    Raises:
        RuntimeError: If the caller is not an IPython session.
    """
    shell = get_ipython()
    if shell is None:
        raise RuntimeError("setup_notebook() must run in an IPython or Jupyter session.")
    shell.run_line_magic("load_ext", "autoreload")
    shell.run_line_magic("autoreload", "2")
    configure_matplotlib()
    seed_everything(seed)
    return get_device()
