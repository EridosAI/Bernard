"""
Configuration loader — single source of truth for all paths and settings.
"""
import yaml
from pathlib import Path
from typing import Any, Optional

_config: Optional[dict] = None
_project_root: Optional[Path] = None


def get_project_root() -> Path:
    """Get the project root directory."""
    global _project_root
    if _project_root is None:
        # Walk up from this file to find project root (contains config/)
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "config").is_dir():
                _project_root = current
                break
            current = current.parent
        else:
            _project_root = Path(__file__).parent.parent
    return _project_root


def load_config(machine: str = "default") -> dict:
    """Load configuration, optionally with machine-specific overrides.

    Args:
        machine: "default", "office_pc", or "workshop_pc"

    Returns:
        Merged configuration dictionary
    """
    global _config

    root = get_project_root()
    config_dir = root / "config"

    # Load base config
    base_path = config_dir / "default.yaml"
    if base_path.exists():
        with open(base_path) as f:
            _config = yaml.safe_load(f) or {}
    else:
        _config = {}

    # Apply machine-specific overrides
    if machine != "default":
        override_path = config_dir / f"{machine}.yaml"
        if override_path.exists():
            with open(override_path) as f:
                overrides = yaml.safe_load(f) or {}
            _config = _deep_merge(_config, overrides)

    return _config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get(key: str, default: Any = None) -> Any:
    """Get a config value by key. Supports dot notation for nested keys.

    Examples:
        get("device")  # "cuda"
        get("internal_jepa.d_internal")  # 512
        get("nonexistent", "fallback")  # "fallback"
    """
    if _config is None:
        load_config()

    keys = key.split(".")
    value = _config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def get_path(key: str) -> Path:
    """Get a config value as an absolute Path, resolved relative to project root."""
    value = get(key)
    if value is None:
        raise KeyError(f"Config key not found: {key}")

    path = Path(value)
    if not path.is_absolute():
        path = get_project_root() / path

    return path
