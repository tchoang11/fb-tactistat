"""Configuration loading for TactiStat.

Three sources feed into a run, in increasing order of precedence:

1. ``configs/default.yaml``  - the baseline system definition
2. an optional experiment config passed with ``--config``
3. ``--set key.path=value`` overrides from the command line

Keeping all three in one place is what makes the ablation study in Section 6.3
tractable: an experiment is a config diff, not a code branch, so a result can
always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# src/tactistat/config.py -> src/tactistat -> src -> <repo root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"
MODEL_REGISTRY_PATH = CONFIGS_DIR / "models.yaml"

_MISSING = object()


class ConfigError(Exception):
    """Raised when a config file is malformed or a requested key is absent."""


@dataclass
class Config:
    """A nested config tree with dotted-path access.

    ``Config`` deliberately does not validate against a schema. The consumers
    (router, stats tool, RAG tool) each read the handful of keys they care
    about and fail loudly on a missing one, which keeps the config format open
    for experiments without a central registry of every knob.
    """

    _data: dict[str, Any] = field(default_factory=dict)

    # -- reading ---------------------------------------------------------------

    def get(self, path: str, default: Any = _MISSING) -> Any:
        """Read ``a.b.c`` from the tree.

        Raises ``ConfigError`` when the path is absent and no default is given,
        rather than returning ``None``. A typo in a config key should stop the
        run, not silently disable a feature and quietly change the numbers in
        the report.
        """
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

    # -- writing ---------------------------------------------------------------

    def set(self, path: str, value: Any) -> None:
        """Write ``a.b.c``, creating intermediate dicts as needed."""
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            existing = node.get(part)
            if not isinstance(existing, dict):
                existing = {}
                node[part] = existing
            node = existing
        node[parts[-1]] = value

    def apply_overrides(self, overrides: list[str] | None) -> None:
        """Apply ``key.path=value`` strings from the command line.

        Values are parsed as YAML scalars, so ``top_k=10`` becomes an int,
        ``rerank.enabled=true`` a bool, and ``labels=[A,B]`` a list -- without
        a hand-rolled type coercion ladder.
        """
        for item in overrides or []:
            if "=" not in item:
                raise ConfigError(f"Override {item!r} is not of the form key.path=value")
            path, raw = item.split("=", 1)
            self.set(path.strip(), yaml.safe_load(raw))

    # -- paths -----------------------------------------------------------------

    def path(self, key: str) -> Path:
        """Read a config value as a filesystem path, anchored at the repo root.

        Config files store repo-relative paths so a checkout works from any
        working directory and the values stay readable in a diff.
        """
        value = self.get(key)
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    def to_dict(self) -> dict[str, Any]:
        """A deep copy, safe to serialise into a results file."""
        return copy.deepcopy(self._data)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``patch`` into a copy of ``base``.

    Nested dicts merge key-by-key; every other type (including lists) is
    replaced wholesale. That means an experiment config only has to state the
    keys it actually changes.
    """
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
    """Build the effective config for a run.

    Args:
        config_path: Optional experiment config layered on top of the baseline.
        overrides: ``key.path=value`` strings, applied last.
        load_env: Read ``.env`` into the process environment. Disabled in tests
            so a developer's real keys cannot leak into a test run.
    """
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
    """Read an environment variable, treating empty strings as unset.

    ``.env.example`` ships every provider key as ``NAME=`` so users can see the
    full list. Without this, an untouched line would register as a present but
    empty key and produce a 401 instead of "you have not set this key".
    """
    value = os.environ.get(name, default)
    if value is not None and not value.strip():
        return default
    return value
