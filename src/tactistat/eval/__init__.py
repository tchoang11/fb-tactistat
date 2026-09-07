"""Labelled evaluation, deterministic metrics, and an independent judge."""

from __future__ import annotations

from typing import Any

__all__ = ["EvaluationRunner", "EvaluationSet", "load_test_set"]


def __getattr__(name: str) -> Any:
    """Keep schema-only validation from importing the full pipeline."""
    if name == "EvaluationRunner":
        from tactistat.eval.runner import EvaluationRunner

        return EvaluationRunner
    if name in {"EvaluationSet", "load_test_set"}:
        from tactistat.eval.schema import EvaluationSet, load_test_set

        return {"EvaluationSet": EvaluationSet, "load_test_set": load_test_set}[name]
    raise AttributeError(name)
