"""Regression tests for config overrides and model-probe parsing."""

from __future__ import annotations

import pytest

from scripts.probe_models import _extract_label
from tactistat.config import Config, ConfigError


def test_config_set_does_not_replace_an_intermediate_scalar():
    config = Config({"rag": {"top_k": 5}})
    with pytest.raises(ConfigError, match="not a section"):
        config.set("rag.top_k.value", 10)
    assert config["rag.top_k"] == 5


@pytest.mark.parametrize("path", ["", ".rag", "rag.", "rag..top_k"])
def test_config_set_rejects_empty_path_parts(path):
    with pytest.raises(ConfigError):
        Config().set(path, 1)


def test_probe_accepts_only_router_labels():
    assert _extract_label('{"label":"STAT"}') == "STAT"
    assert _extract_label('answer: {"label":"HYBRID"}') == "HYBRID"
    assert _extract_label('{"label":"BANANA"}') is None
    assert _extract_label('{"label":3}') is None
