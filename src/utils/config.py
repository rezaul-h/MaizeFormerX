"""
Configuration loading and recursive merge utilities.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.utils.io import read_yaml


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values from `override` take precedence over `base`.
    """
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def _resolve_inherits_path(config_path: Path, inherits_value: str) -> Path:
    """Resolve an inherited config path relative to repository root or current file."""
    inherits_path = Path(inherits_value)

    if inherits_path.is_absolute():
        return inherits_path

    # First try relative to current config file.
    candidate_local = (config_path.parent / inherits_path).resolve()
    if candidate_local.exists():
        return candidate_local

    # Then try relative to project root.
    candidate_root = Path.cwd() / inherits_path
    return candidate_root.resolve()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config with optional single-file inheritance support.

    Expected pattern:
        inherits: "configs/train/default.yaml"
    """
    path = Path(path).resolve()
    cfg = read_yaml(path)

    inherits = cfg.pop("inherits", None)
    if inherits is None:
        return cfg

    parent_path = _resolve_inherits_path(path, inherits)
    parent_cfg = load_yaml_config(parent_path)
    return deep_merge_dicts(parent_cfg, cfg)


def load_and_merge_configs(paths: list[str | Path]) -> dict[str, Any]:
    """Load and recursively merge multiple YAML config files in order."""
    merged: dict[str, Any] = {}
    for path in paths:
        cfg = load_yaml_config(path)
        merged = deep_merge_dicts(merged, cfg)
    return merged


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """
    Set a nested config value using dot notation.

    Example
    -------
    set_by_dotted_key(cfg, "training.optimizer.lr", 1e-4)
    """
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value
    return config


def apply_overrides(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply dotted-key overrides to a config dictionary."""
    if not overrides:
        return config

    updated = deepcopy(config)
    for key, value in overrides.items():
        set_by_dotted_key(updated, key, value)
    return updated