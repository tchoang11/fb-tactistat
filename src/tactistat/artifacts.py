"""Helpers for reproducible, atomic data artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

ARTIFACT_SCHEMA_VERSION = 1


def stable_hash(value: Any) -> str:
    """Return a stable short hash for JSON-compatible data."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def manifest_matches(manifest: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    """Check the fields that define artifact compatibility."""
    return manifest is not None and all(
        manifest.get(key) == value for key, value in expected.items()
    )


def write_json_atomic(path: Path, value: Any, *, indent: int = 2) -> None:
    """Write JSON through a temporary file, then replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_text_atomic(path: Path, value: str) -> None:
    """Write text through a temporary file, then replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(value)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write Parquet through a temporary file, then replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temp_path = Path(f.name)
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
