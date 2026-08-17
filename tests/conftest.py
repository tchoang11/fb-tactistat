"""Keep tests from reading keys or enabling process-wide tracing."""

from __future__ import annotations

import os

import pytest

_TRACING_ENV = {
    "LANGSMITH_API_KEY": None,
    "LANGCHAIN_API_KEY": None,
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
}


@pytest.fixture(scope="session", autouse=True)
def offline_tracing():
    for name, value in _TRACING_ENV.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    yield
