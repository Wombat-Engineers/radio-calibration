"""Typed settings of the simulation and of the calibration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _value(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    """Return `mapping[key]`. Raise a ValueError that gives the full key if it is missing."""
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise ValueError(f"The settings do not have the key '{path}{key}'.")
    return mapping[key]


def _check_count(name: str, value: int) -> None:
    """Raise a ValueError if a count is less than 1."""
    if value < 1:
        raise ValueError(f"{name} must be 1 or more, but it is {value}.")


@dataclass(frozen=True)
class SimulationSettings:
    """Settings of the Sionna RT simulation.

    Attributes:
        frequency_hz: Carrier frequency, in Hz.
        subcarrier_spacing_hz: Subcarrier spacing, in Hz. The scene bandwidth is one
            resource element. The bandwidth changes only the thermal noise, not the RSRP.
        receiver: Arguments of the Sionna RT `PlanarArray` of the UE.
        solver: Arguments of a call of the Sionna RT `PathSolver`.
        deterministic: Argument of the `PathSolver` constructor.
        rx_batch_size: Number of UE positions in one call of the path solver.

    Raises:
        ValueError: If `rx_batch_size` is less than 1.
    """

    frequency_hz: float
    subcarrier_spacing_hz: float
    receiver: Mapping[str, Any]
    solver: Mapping[str, Any]
    deterministic: bool
    rx_batch_size: int

    def __post_init__(self) -> None:
        _check_count("rx_batch_size", self.rx_batch_size)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SimulationSettings":
        """Make the settings from a mapping with the structure of `simulation.yaml`.

        The function ignores keys that it does not use, for example `scene`.

        Args:
            mapping: The keys `radio` (`frequency_hz`, `subcarrier_spacing_hz`), `receiver`,
                `solver`, `deterministic`, and `rx_batch_size`.

        Returns:
            The settings.

        Raises:
            ValueError: If a key is missing, or if `rx_batch_size` is less than 1.
        """
        radio = _value(mapping, "radio", "")
        return cls(
            frequency_hz=float(_value(radio, "frequency_hz", "radio.")),
            subcarrier_spacing_hz=float(_value(radio, "subcarrier_spacing_hz", "radio.")),
            receiver=dict(_value(mapping, "receiver", "")),
            solver=dict(_value(mapping, "solver", "")),
            deterministic=bool(_value(mapping, "deterministic", "")),
            rx_batch_size=int(_value(mapping, "rx_batch_size", "")),
        )


@dataclass(frozen=True)
class CalibrationSettings:
    """Settings of the calibration of a neural material (arXiv:2311.18558).

    Attributes:
        encoding_octaves: Number of frequencies of the positional encoding (Eq. 36).
        hidden_layers: Number of hidden layers of the network.
        hidden_units: Number of neurons in each hidden layer.
        learning_rate: Learning rate of the AdamW optimizer.
        weight_decay: Weight decay of the AdamW optimizer.
        num_steps: Number of gradient steps.
        batch_size: Number of records in one gradient step.
        validation_interval: Number of steps between two calculations of the validation loss.
        scale_decay: Decay of the moving average of the power scale (Eq. 42).

    Raises:
        ValueError: If `num_steps`, `batch_size`, or `validation_interval` is less than 1.
    """

    encoding_octaves: int
    hidden_layers: int
    hidden_units: int
    learning_rate: float
    weight_decay: float
    num_steps: int
    batch_size: int
    validation_interval: int
    scale_decay: float

    def __post_init__(self) -> None:
        for name in ("num_steps", "batch_size", "validation_interval"):
            _check_count(name, getattr(self, name))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CalibrationSettings":
        """Make the settings from a mapping with the structure of `calibration.yaml`.

        The function ignores keys that it does not use, for example `split`.

        Args:
            mapping: The keys `network` (`encoding_octaves`, `hidden_layers`,
                `hidden_units`), `learning_rate`, `weight_decay`, `num_steps`,
                `batch_size`, `validation_interval`, and `scale_decay`.

        Returns:
            The settings.

        Raises:
            ValueError: If a key is missing, or if a count is less than 1.
        """
        network = _value(mapping, "network", "")
        return cls(
            encoding_octaves=int(_value(network, "encoding_octaves", "network.")),
            hidden_layers=int(_value(network, "hidden_layers", "network.")),
            hidden_units=int(_value(network, "hidden_units", "network.")),
            learning_rate=float(_value(mapping, "learning_rate", "")),
            weight_decay=float(_value(mapping, "weight_decay", "")),
            num_steps=int(_value(mapping, "num_steps", "")),
            batch_size=int(_value(mapping, "batch_size", "")),
            validation_interval=int(_value(mapping, "validation_interval", "")),
            scale_decay=float(_value(mapping, "scale_decay", "")),
        )
