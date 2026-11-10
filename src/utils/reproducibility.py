"""Random seeds and device selection."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy, and PyTorch (CPU and CUDA) random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the CUDA device if one is available, else the CPU device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
