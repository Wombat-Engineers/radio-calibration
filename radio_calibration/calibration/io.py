"""File format of a trained U-Net."""

from collections.abc import Sequence
from pathlib import Path

import torch

from radio_calibration.calibration.features import Normalization
from radio_calibration.calibration.unet import UNet

_SUFFIX = ".pt"
_KEYS = ("state_dict", "architecture", "normalization", "channels", "residual")
_ARCHITECTURE = ("in_channels", "base_channels", "depth", "groups")


def save_model(
    path: Path,
    network: UNet,
    normalization: Normalization,
    channels: Sequence[str],
    residual: bool,
) -> None:
    """Write the network, its normalization, and its input format to a `.pt` file.

    The file has only tensors, numbers, strings, lists, and dictionaries. Thus
    `torch.load(..., weights_only=True)` can read it.

    Raises:
        ValueError: If the suffix of `path` is not `.pt`.
    """
    if path.suffix != _SUFFIX:
        raise ValueError(f"The model file must have the suffix {_SUFFIX}, but it is '{path}'.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {name: value.cpu() for name, value in network.state_dict().items()},
            "architecture": {name: int(getattr(network, name)) for name in _ARCHITECTURE},
            "normalization": normalization.state(),
            "channels": list(channels),
            "residual": bool(residual),
        },
        path,
    )


def load_model(
    path: Path, device: torch.device
) -> tuple[UNet, Normalization, tuple[str, ...], bool]:
    """Read a file of `save_model`.

    The function reads with `weights_only=True`, thus a file cannot run code.

    Returns:
        The network on `device` in evaluation mode, its normalization, its input channels,
        and its output type (True for a residual output).

    Raises:
        ValueError: If the suffix is not `.pt`, or if the file does not have a key.
    """
    if path.suffix != _SUFFIX:
        raise ValueError(f"The model file must have the suffix {_SUFFIX}, but it is '{path}'.")
    content = torch.load(path, map_location=device, weights_only=True)
    missing = [key for key in _KEYS if key not in content]
    if missing:
        raise ValueError(f"The model file '{path}' does not have the keys {missing}.")
    network = UNet(**{name: content["architecture"][name] for name in _ARCHITECTURE})
    network.load_state_dict(content["state_dict"])
    network.to(device).eval()
    normalization = Normalization.from_state(content["normalization"])
    return network, normalization, tuple(content["channels"]), bool(content["residual"])
