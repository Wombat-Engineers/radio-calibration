"""Notebook setup."""

import torch
from IPython import get_ipython
from omegaconf import DictConfig

from src.utils.output import configure_output
from src.utils.plotting import configure_matplotlib
from src.utils.reproducibility import get_device, seed_everything


def setup_notebook(cfg: DictConfig) -> torch.device:
    """Prepare a notebook session. Call this in the first cell.

    Enables autoreload of edited modules, applies the project figure style, seeds the
    random generators, and sets the directories of the figures and the tables.

    Args:
        cfg: The configuration from `load_config`. The function uses `seed` and `output`.

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
    seed_everything(cfg.seed)
    configure_output(cfg.output)
    return get_device()
