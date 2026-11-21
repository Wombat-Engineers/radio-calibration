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
        radio_map: Arguments of a call of the Sionna RT `RadioMapSolver`. The geometry
            arguments (`center`, `orientation`, `size`, `cell_size`) come from the grid,
            not from this mapping.
        radio_map_cell_size_m: Length of one side of a radio-map cell, in meters.
        radio_map_tx_batch_size: Maximum number of transmitters in one solver call.

    Raises:
        ValueError: If `radio_map_tx_batch_size` is less than 1, or if
            `radio_map_cell_size_m` is not more than 0.
    """

    frequency_hz: float
    radio_map: Mapping[str, Any]
    radio_map_cell_size_m: float
    radio_map_tx_batch_size: int

    def __post_init__(self) -> None:
        _check_count("radio_map_tx_batch_size", self.radio_map_tx_batch_size)
        if not self.radio_map_cell_size_m > 0:
            raise ValueError(
                f"radio_map_cell_size_m must be more than 0, but it is "
                f"{self.radio_map_cell_size_m}."
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SimulationSettings":
        """Make the settings from a mapping with the structure of `simulation.yaml`.

        Args:
            mapping: The keys `radio` (`frequency_hz`), `radio_map`,
                `radio_map_cell_size_m`, and `radio_map_tx_batch_size`.

        Returns:
            The settings.

        Raises:
            ValueError: If a key is missing, or if a value is out of its range.
        """
        radio = _value(mapping, "radio", "")
        return cls(
            frequency_hz=float(_value(radio, "frequency_hz", "radio.")),
            radio_map=dict(_value(mapping, "radio_map", "")),
            radio_map_cell_size_m=float(_value(mapping, "radio_map_cell_size_m", "")),
            radio_map_tx_batch_size=int(_value(mapping, "radio_map_tx_batch_size", "")),
        )


@dataclass(frozen=True)
class TrainingSettings:
    """Settings of one training stage.

    Attributes:
        learning_rate: Learning rate of the Adam optimizer.
        batch_size: Number of samples in one gradient step. The last batch of an epoch can
            have fewer samples.
        num_epochs: Number of epochs. In one epoch, each train sample is in one batch.

    Raises:
        ValueError: If `batch_size` or `num_epochs` is less than 1, or if `learning_rate` is
            not more than 0.
    """

    learning_rate: float
    batch_size: int
    num_epochs: int

    def __post_init__(self) -> None:
        _check_count("batch_size", self.batch_size)
        _check_count("num_epochs", self.num_epochs)
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be more than 0, but it is {self.learning_rate}.")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], path: str) -> "TrainingSettings":
        """Make the settings from a mapping with the keys of one training stage.

        Args:
            mapping: The keys `learning_rate`, `batch_size`, and `num_epochs`.
            path: Prefix of the keys in an error message, for example `"pretraining."`.

        Returns:
            The settings.

        Raises:
            ValueError: If a key is missing, or if a value is out of its range.
        """
        return cls(
            learning_rate=float(_value(mapping, "learning_rate", path)),
            batch_size=int(_value(mapping, "batch_size", path)),
            num_epochs=int(_value(mapping, "num_epochs", path)),
        )


@dataclass(frozen=True)
class CalibrationSettings:
    """Settings of the U-Net that corrects the simulated path gain map.

    Attributes:
        base_channels: Number of feature channels of the first level of the U-Net.
        depth: Number of down-sampling steps of the U-Net.
        groups: Number of groups of each GroupNorm layer.
        validation_fraction: Fraction of the pretraining samples for validation.
        pretraining: Settings of the pretraining on the simulated scene variants.
        fine_tuning: Settings of the fine-tuning on the MDT records.

    Raises:
        ValueError: If a count is less than 1, or if `validation_fraction` is not more than 0
            and less than 1.
    """

    base_channels: int
    depth: int
    groups: int
    validation_fraction: float
    pretraining: TrainingSettings
    fine_tuning: TrainingSettings

    def __post_init__(self) -> None:
        for name in ("base_channels", "depth", "groups"):
            _check_count(name, getattr(self, name))
        if not 0 < self.validation_fraction < 1:
            raise ValueError(
                f"validation_fraction must be more than 0 and less than 1, but it is "
                f"{self.validation_fraction}."
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CalibrationSettings":
        """Make the settings from a mapping with the structure of `calibration.yaml`.

        Args:
            mapping: The keys `network` (`base_channels`, `depth`, `groups`),
                `validation_fraction`, `pretraining`, and `fine_tuning`.

        Returns:
            The settings.

        Raises:
            ValueError: If a key is missing, or if a value is out of its range.
        """
        network = _value(mapping, "network", "")
        return cls(
            base_channels=int(_value(network, "base_channels", "network.")),
            depth=int(_value(network, "depth", "network.")),
            groups=int(_value(network, "groups", "network.")),
            validation_fraction=float(_value(mapping, "validation_fraction", "")),
            pretraining=TrainingSettings.from_mapping(
                _value(mapping, "pretraining", ""), "pretraining."
            ),
            fine_tuning=TrainingSettings.from_mapping(
                _value(mapping, "fine_tuning", ""), "fine_tuning."
            ),
        )
