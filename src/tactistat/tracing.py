"""Trace model calls to LangSmith or local JSONL."""

from __future__ import annotations

import json
import threading
import time
import warnings
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from tactistat.config import Config, env

LANGSMITH_KEY_ENV = "LANGSMITH_API_KEY"
# These enable process-wide tracing outside this module's control.
GLOBAL_TRACING_ENV = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
_warned = False


class JsonlTraceHandler(BaseCallbackHandler):
    """Append auditable call/result pairs when LangSmith is unavailable."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started: dict[str, float] = {}
        self._lock = threading.Lock()

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _begin(self, run_id: Any) -> None:
        with self._lock:
            self._started[str(run_id)] = time.perf_counter()

    def _elapsed_ms(self, run_id: Any) -> int:
        with self._lock:
            started = self._started.pop(str(run_id), None)
        return 0 if started is None else int((time.perf_counter() - started) * 1000)

    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id=None,
        parent_run_id=None,
        metadata=None,
        **kwargs,
    ) -> None:
        self._begin(run_id)
        prompt = [
            {"role": getattr(m, "type", "?"), "content": getattr(m, "content", "")}
            for batch in messages
            for m in batch
        ]
        self._write(
            {
                "event": "call",
                "run_id": str(run_id),
                "parent_run_id": None if parent_run_id is None else str(parent_run_id),
                "pipeline_run_id": (metadata or {}).get("tactistat_run_id"),
                "evaluation_item_id": (metadata or {}).get("tactistat_eval_item_id"),
                "evaluation_call_id": (metadata or {}).get("tactistat_eval_call_id"),
                "node": (metadata or {}).get("langgraph_node"),
                "timestamp": time.time(),
                "model": (serialized or {}).get("name"),
                "prompt": prompt,
            }
        )

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:
        generations = [g.text for batch in getattr(response, "generations", []) for g in batch]
        self._write(
            {
                "event": "result",
                "run_id": str(run_id),
                "timestamp": time.time(),
                "latency_ms": self._elapsed_ms(run_id),
                "output": generations,
                "usage": (getattr(response, "llm_output", None) or {}).get("token_usage"),
            }
        )

    def on_llm_error(self, error, *, run_id=None, **kwargs) -> None:
        self._write(
            {
                "event": "error",
                "run_id": str(run_id),
                "timestamp": time.time(),
                "latency_ms": self._elapsed_ms(run_id),
                "error": f"{type(error).__name__}: {error}",
            }
        )


def _warn_if_environment_overrides(backend: str) -> None:
    """Warn when process-wide tracing overrides this pipeline's config."""
    global _warned
    if _warned or backend == "langsmith":
        return
    forced = [name for name in GLOBAL_TRACING_ENV if (env(name) or "").lower() == "true"]
    if forced:
        _warned = True
        warnings.warn(
            f"{', '.join(forced)} is set in the environment, so LangChain traces every run "
            f"regardless of tracing.enabled (config selected {backend!r}). Unset it in .env.",
            RuntimeWarning,
            stacklevel=2,
        )


def tracing_backend(config: Config) -> str:
    """Which backend this config and environment select: langsmith | file | off."""
    if not config.get("tracing.enabled", True):
        return "off"
    return "langsmith" if env(LANGSMITH_KEY_ENV) else "file"


def configure_tracing(config: Config) -> list[BaseCallbackHandler]:
    """Build per-run callbacks; without a LangSmith key, use local JSONL."""
    backend = tracing_backend(config)
    _warn_if_environment_overrides(backend)
    if backend == "off":
        return []
    if backend == "file":
        return [JsonlTraceHandler(config.path("tracing.file_dir") / "trace.jsonl")]

    from langchain_core.tracers import LangChainTracer

    return [LangChainTracer(project_name=config.get("tracing.project", None))]
