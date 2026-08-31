from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SYSTEM_ROOT / "configs" / "system.yaml"


def resolve_system_path(value: str | Path) -> Path:
    """Resolve application-owned paths relative to this package, not the research repo."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (SYSTEM_ROOT / path).resolve()


resolve_repo_path = resolve_system_path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = resolve_system_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"System config must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def nested(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    return value.get(key, default) if isinstance(value, dict) else default
