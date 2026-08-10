"""Load baseline, experiment, and CLI configuration in precedence order."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Three parents up from this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"
MODEL_REGISTRY_PATH = CONFIGS_DIR / "models.yaml"

_MISSING = object()


class ConfigError(Exception):
    """Raised when a config file is malformed or a requested key is absent."""


@dataclass
class Config:
    """A nested config tree with dotted-path access and no central schema."""

    _data: dict[str, Any] = field(default_factory=dict)

    # Reading

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Read ``a.b.c``; raise when missing unless a default is supplied."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"Missing config key: {path!r}")
                return default
            node = node[part]
        return node

    def __getitem__(self, path: str) -> Any:
        return self.get(path)

    def section(self, path: str) -> dict[str, Any]:
        """Read a subtree, asserting that it is a mapping."""
        value = self.get(path)
        if not isinstance(value, dict):
            raise ConfigError(f"Config key {path!r} is not a section (got {type(value).__name__})")
        return value

    # Writing

    def set(self, path: str, value: Any) -> None:
        """Write ``a.b.c``, creating intermediate dicts as needed."""
        if not path or any(not part for part in path.split(".")):
            raise ConfigError("Config path must contain non-empty dotted keys")
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            existing = node.get(part, _MISSING)
            if existing is _MISSING:
                existing = {}
                node[part] = existing
            elif not isinstance(existing, dict):
                raise ConfigError(
                    f"Cannot set {path!r}: intermediate key {part!r} is "
                    f"{type(existing).__name__}, not a section"
                )
            node = existing
        node[parts[-1]] = value

    def apply_overrides(self, overrides: list[str] | None) -> None:
        """Apply ``key.path=value`` strings, parsing values as YAML."""
        for item in overrides or []:
            if "=" not in item:
                raise ConfigError(f"Override {item!r} is not of the form key.path=value")
            path, raw = item.split("=", 1)
            self.set(path.strip(), yaml.safe_load(raw))

    # Paths

    def path(self, key: str) -> Path:
        """Resolve a config path relative to the repository root."""
        value = self.get(key)
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    def to_dict(self) -> dict[str, Any]:
        """A deep copy, safe to serialise into a results file."""
        return copy.deepcopy(self._data)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge nested mappings; replace all other values, including lists."""
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at the top level")
    return data


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
    load_env: bool = True,
) -> Config:
    """Load defaults, then an experiment file, then CLI overrides."""
    if load_env:
        load_dotenv(PROJECT_ROOT / ".env", override=False)

    data = _read_yaml(DEFAULT_CONFIG_PATH)

    if config_path is not None:
        experiment_path = Path(config_path)
        if not experiment_path.is_absolute():
            # Accept both `configs/foo.yaml` and a bare `foo.yaml`.
            experiment_path = (
                PROJECT_ROOT / experiment_path
                if (PROJECT_ROOT / experiment_path).exists()
                else CONFIGS_DIR / experiment_path
            )
        data = _deep_merge(data, _read_yaml(experiment_path))

    config = Config(data)
    config.apply_overrides(overrides)
    return config


def load_model_registry() -> dict[str, Any]:
    """Load ``configs/models.yaml``."""
    registry = _read_yaml(MODEL_REGISTRY_PATH)
    if "providers" not in registry:
        raise ConfigError(f"{MODEL_REGISTRY_PATH} must define a top-level `providers` mapping")
    return registry


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, treating blank values as unset."""
    value = os.environ.get(name, default)
    if value is not None and not value.strip():
        return default
    return value
