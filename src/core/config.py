"""
Route Resilience — src/core/config.py

YAML configuration loader.
Single source of truth for all pipeline config access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from src.core.exceptions import ConfigError

# Default config path, relative to project root
DEFAULT_CONFIG_PATH = "configs/config.yaml"


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load and return the YAML configuration as a nested dict.

    Args:
        config_path: Path to the YAML config file.
                     Defaults to configs/config.yaml relative to CWD.

    Returns:
        Parsed config dict.

    Raises:
        ConfigError: If the file is missing or the YAML is malformed/empty.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(
            f"Configuration file not found: {config_path}\n"
            "Please ensure configs/config.yaml exists in the project root."
        )
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {config_path}: {exc}") from exc

    if cfg is None:
        raise ConfigError(f"Empty configuration file: {config_path}")

    return cfg
