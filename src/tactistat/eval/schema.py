"""Schema and validation for the labelled evaluation set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tactistat.artifacts import stable_hash

SCHEMA_VERSION = 1


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class StatsArgsExpectation(_StrictModel):
    operation: Literal["player", "ranking", "compare", "total"]
    metric: str
    players: list[str] = Field(default_factory=list)
    per90: bool = False
    top_n: int = Field(default=5, ge=1, le=50)
    team: str | None = None
    stage: str | None = None
    opponent: str | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> StatsArgsExpectation:
        count = len(self.players)
        if self.operation == "player" and count != 1:
            raise ValueError("player expects exactly one name")
        if self.operation == "compare" and count < 2:
            raise ValueError("compare expects at least two names")
        if self.operation in ("ranking", "total") and count:
            raise ValueError(f"{self.operation} expects no player names")
        if self.operation == "total" and self.per90:
            raise ValueError("a total cannot be per 90")
        return self


class RouteExpectation(_StrictModel):
    label: Literal["STAT", "TACTICAL", "HYBRID"]
    stats_args: StatsArgsExpectation | None = None

    @model_validator(mode="after")
    def validate_tools(self) -> RouteExpectation:
        needs_stats = self.label in ("STAT", "HYBRID")
        if needs_stats != (self.stats_args is not None):
            raise ValueError(f"{self.label} has an inconsistent stats_args value")
        return self


class ExpectedValue(_StrictModel):
    label: str = Field(min_length=1)
    value: float
    tolerance: float = Field(default=1e-6, ge=0)


class StatsExpectation(_StrictModel):
    values: list[ExpectedValue] = Field(min_length=1)


class RetrievalTarget(_StrictModel):
    """A relevant information unit independent of the chunking strategy."""

    title: str = Field(min_length=1)
    heading: str | None = None
    text_contains: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_anchors(self) -> RetrievalTarget:
        anchors = [anchor.strip().casefold() for anchor in self.text_contains]
        if any(not anchor for anchor in anchors):
            raise ValueError("retrieval anchors cannot be blank")
        if len(anchors) != len(set(anchors)):
            raise ValueError("retrieval anchors must be unique")
        return self


class RetrievalExpectation(_StrictModel):
    relevant: list[RetrievalTarget] = Field(min_length=1)


class EvaluationItem(_StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=3)
    language: Literal["en", "vi"]
    route: RouteExpectation
    stats: StatsExpectation | None = None
    retrieval: RetrievalExpectation | None = None
    reference_answer: str = Field(min_length=3)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ground_truth(self) -> EvaluationItem:
        needs_stats = self.route.label in ("STAT", "HYBRID")
        needs_rag = self.route.label in ("TACTICAL", "HYBRID")
        if needs_stats != (self.stats is not None):
            raise ValueError(f"{self.route.label} has inconsistent stats ground truth")
        if needs_rag != (self.retrieval is not None):
            raise ValueError(f"{self.route.label} has inconsistent retrieval ground truth")
        if self.stats is not None and self.route.stats_args is not None:
            expected = len(self.stats.values)
            args = self.route.stats_args
            required = {
                "player": 1,
                "compare": len(args.players),
                "ranking": args.top_n,
                "total": 1,
            }[args.operation]
            if expected != required:
                raise ValueError(
                    f"{args.operation} expects {required} labelled value(s), got {expected}"
                )
            labels = [value.label.casefold() for value in self.stats.values]
            if len(labels) != len(set(labels)):
                raise ValueError("stats ground truth contains duplicate labels")
        if self.retrieval is not None:
            targets = [
                (
                    target.title.casefold(),
                    (target.heading or "").casefold(),
                    tuple(anchor.casefold() for anchor in target.text_contains),
                )
                for target in self.retrieval.relevant
            ]
            if len(targets) != len(set(targets)):
                raise ValueError("retrieval ground truth contains duplicate targets")
        if len(self.tags) != len(set(self.tags)):
            raise ValueError("tags must be unique within an item")
        return self


class EvaluationSet(_StrictModel):
    schema_version: Literal[1]
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    items: list[EvaluationItem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> EvaluationSet:
        ids = [item.id for item in self.items]
        duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate evaluation ids: {duplicates}")
        return self

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


def load_test_set(path: str | Path) -> EvaluationSet:
    """Read and fully validate a test set before any model call is made."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation set not found: {source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc
    return EvaluationSet.model_validate(payload)
