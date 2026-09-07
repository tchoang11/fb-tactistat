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


@pytest.fixture(autouse=True)
def no_repo_writes():
    """Fail a test that leaves files in the tracked evaluation directories.

    Two separate bugs have shipped this way: an audit path derived from the
    config instead of its report, and a runner call that defaulted its output
    into `eval/results/raw/`. Both wrote real artifacts from the test suite and
    were only noticed by eye, so the suite now notices for us.
    """
    from pathlib import Path

    watched = Path(__file__).resolve().parents[1] / "eval" / "results"

    def snapshot() -> set[Path]:
        return set(watched.rglob("*")) if watched.exists() else set()

    before = snapshot()
    yield
    created = sorted(path for path in snapshot() - before if path.is_file())
    for path in created:
        path.unlink()
    assert not created, (
        "test wrote into the tracked evaluation directory: "
        + ", ".join(str(path.relative_to(watched.parent.parent)) for path in created)
        + " — pass an explicit output_path under tmp_path"
    )
