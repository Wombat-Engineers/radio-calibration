"""Composition of the configuration in `configs/`."""

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.utils.paths import CONFIG_DIR


def load_config(*overrides: str) -> DictConfig:
    """Compose `configs/config.yaml` as `task scenario` composes it.

    The result has the keys `seed`, `scenario`, `simulation`, and `calibration`. It has no
    `hydra` key.

    Args:
        overrides: Hydra overrides, for example `"scenario.sites.isd_m=60"`.

    Returns:
        The configuration.

    Raises:
        hydra.errors.ConfigCompositionException: If an override names a key that is not in
            the configuration.
    """
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))
