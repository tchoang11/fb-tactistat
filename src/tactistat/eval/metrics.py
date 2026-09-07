"""Deterministic stage metrics for one evaluation result."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from tactistat.eval.schema import EvaluationItem, RetrievalTarget, StatsArgsExpectation
from tactistat.pipeline import PipelineResult
from tactistat.rag_tool.retrieve import Passage
from tactistat.stats_tool.query import StatsAnswer

JUDGE_METRICS = ("judge_correctness", "judge_completeness", "judge_faithfulness")


def _normalise(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    folded = "".join(char for char in text if not unicodedata.combining(char)).casefold()
    return " ".join(re.findall(r"\w+", folded))


def _same_name(actual: str, expected: str) -> bool:
    left, right = _normalise(actual), _normalise(expected)
    if left == right:
        return True
    # Ground truth uses common names while StatsBomb stores legal names.
    left_words, right_words = set(left.split()), set(right.split())
    return left_words <= right_words or right_words <= left_words


def _route_slots_match(actual: dict[str, Any] | None, expected: StatsArgsExpectation) -> bool:
    if actual is None:
        return False
    expected_data = expected.model_dump()
    keys = ["operation", "metric", "players", "per90", "team", "stage", "opponent"]
    if expected.operation == "ranking":
        keys.append("top_n")
    for key in keys:
        got, wanted = actual.get(key), expected_data[key]
        if key == "players":
            if len(got or []) != len(wanted):
                return False
            unmatched = list(got or [])
            for name in wanted:
                match = next(
                    (candidate for candidate in unmatched if _same_name(candidate, name)), None
                )
                if match is None:
                    return False
                unmatched.remove(match)
        elif isinstance(wanted, str) or isinstance(got, str):
            if _normalise(got) != _normalise(wanted):
                return False
        elif got != wanted:
            return False
    return True


def route_scores(item: EvaluationItem, result: PipelineResult) -> dict[str, float]:
    label = float(result.route.label == item.route.label)
    if item.route.stats_args is None:
        slots = float(result.route.stats_args is None and result.route.rag_query is not None)
    else:
        slots = float(_route_slots_match(result.route.stats_args, item.route.stats_args))
        if item.route.label == "HYBRID":
            slots *= float(bool(result.route.rag_query))
    return {"route_label_accuracy": label, "route_slots_accuracy": slots}


def _actual_values(answer: StatsAnswer, operation: str) -> list[tuple[str, float]]:
    if operation == "total":
        return [] if answer.total is None else [(answer.total.label, float(answer.total.value))]
    return [(row.player_name, float(row.value)) for row in answer.rows]


def stats_exact_match(item: EvaluationItem, answer: StatsAnswer | None) -> float | None:
    expected = item.stats
    if expected is None:
        return None
    if answer is None or not answer.ok or item.route.stats_args is None:
        return 0.0
    if answer.metric != item.route.stats_args.metric or answer.per90 != item.route.stats_args.per90:
        return 0.0

    operation = item.route.stats_args.operation
    actual = _actual_values(answer, operation)
    if len(actual) != len(expected.values):
        return 0.0
    for wanted in expected.values:
        matches = [value for label, value in actual if _same_name(label, wanted.label)]
        if not matches or not any(
            math.isclose(value, wanted.value, abs_tol=wanted.tolerance, rel_tol=0)
            for value in matches
        ):
            return 0.0
    # A leaderboard is an ordered answer: "who scored the most" is answered by
    # position, so a set-equal check would score a reversed table 1.0. Compared
    # by value in returned order, which lets tied players permute freely.
    if operation == "ranking":
        for (_, got), wanted in zip(actual, expected.values, strict=True):
            if not math.isclose(got, wanted.value, abs_tol=wanted.tolerance, rel_tol=0):
                return 0.0
    return 1.0


def _target_matches(target: RetrievalTarget, passage: Passage) -> bool:
    if _normalise(target.title) != _normalise(passage.title):
        return False
    body = _normalise(passage.text)
    if any(_normalise(anchor) not in body for anchor in target.text_contains):
        return False
    # Fixed windows lose heading metadata, and a label may deliberately name no
    # heading; either way the content anchor already identifies the unit.
    if target.heading is None or passage.heading is None:
        return True
    return _normalise(target.heading) == _normalise(passage.heading)


def retrieval_scores(
    item: EvaluationItem, passages: list[Passage], ks: Iterable[int]
) -> dict[str, float | None]:
    expected = item.retrieval
    keys = [int(k) for k in ks]
    if expected is None:
        return {**{f"retrieval_recall_at_{k}": None for k in keys}, "retrieval_mrr": None}

    ranks: list[int | None] = []
    for target in expected.relevant:
        rank = next(
            (passage.rank for passage in passages if _target_matches(target, passage)), None
        )
        ranks.append(rank)
    recalls = {
        f"retrieval_recall_at_{k}": sum(rank is not None and rank <= k for rank in ranks)
        / len(ranks)
        for k in keys
    }
    found = [rank for rank in ranks if rank is not None]
    return {**recalls, "retrieval_mrr": 0.0 if not found else 1.0 / min(found)}


# Which metrics stop being measurements when a stage could not run. A stage
# that ran and found nothing still scores; a stage that never ran has no score,
# and averaging a zero in its place reports an outage as a bad answer.
STAGE_METRICS: dict[str, tuple[str, ...]] = {
    "translation": ("translation_language_accuracy",),
    "router": ("route_label_accuracy", "route_slots_accuracy"),
    "stats": ("stats_exact_match",),
    "rag": ("retrieval_mrr", "retrieval_recall_at_"),
    "synthesis": ("answer_evidence_valid", *JUDGE_METRICS),
}


def _blank_unusable(
    scores: dict[str, float | None], stage_errors: Iterable[str]
) -> dict[str, float | None]:
    """Drop the metrics that a broken stage would otherwise score as zero."""
    stages = list(stage_errors)
    if not stages:
        return scores
    for stage in stages:
        for prefix in STAGE_METRICS.get(stage, ()):
            for name in list(scores):
                if name == prefix or name.startswith(prefix):
                    scores[name] = None
    # Downstream of any broken stage the whole answer is unmeasured, not failed.
    scores["pipeline_success"] = None
    return scores


def score_result(
    item: EvaluationItem,
    result: PipelineResult,
    recall_at_k: Iterable[int],
    judge: dict[str, float | None] | None = None,
) -> dict[str, float | None]:
    scores: dict[str, float | None] = route_scores(item, result)
    scores["translation_language_accuracy"] = (
        None
        if result.translation.status == "disabled"
        else float(result.translation.source_language == item.language)
    )
    scores["stats_exact_match"] = stats_exact_match(item, result.stats)
    passages = result.rag.passages if result.rag is not None and result.rag.ok else []
    scores.update(retrieval_scores(item, passages, recall_at_k))
    scores["answer_evidence_valid"] = float(result.answer.ok)
    # Every labelled item is answerable; a safe abstention is valid but not correct.
    scores["pipeline_success"] = float(result.ok and not result.answer.abstained)
    scores.update(judge or {name: None for name in JUDGE_METRICS})
    # An abstention asserts nothing, so a judge scores it perfectly faithful.
    # Averaging that in would let a system that refuses every question report
    # faithfulness 1.0; correctness and completeness still score it as the
    # miss it is.
    if result.answer.abstained:
        scores["judge_faithfulness"] = None
    return _blank_unusable(scores, result.stage_errors)


def failure_scores(
    item: EvaluationItem, recall_at_k: Iterable[int], *, translation_scored: bool = True
) -> dict[str, float | None]:
    """Scores for an item whose pipeline raised, so no stage produced anything.

    `translation_scored` mirrors what `score_result` would have done: with
    translation disabled it returns None there, and an errored item must not be
    the only row in that metric's denominator.
    """
    scores: dict[str, float | None] = {
        "translation_language_accuracy": 0.0 if translation_scored else None,
        "route_label_accuracy": 0.0,
        "route_slots_accuracy": 0.0,
        "stats_exact_match": 0.0 if item.stats else None,
        "answer_evidence_valid": 0.0,
        "pipeline_success": 0.0,
    }
    scores.update(retrieval_scores(item, [], recall_at_k))
    scores.update({name: None for name in JUDGE_METRICS})
    return scores


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    metric_names = sorted({name for row in rows for name in row["scores"]})
    metrics: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        values = [row["scores"].get(name) for row in rows]
        usable = [float(value) for value in values if value is not None]
        metrics[name] = {
            "mean": sum(usable) / len(usable) if usable else None,
            "n": len(usable),
        }
    return metrics


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average only applicable metrics and report their denominator."""
    metrics = _metric_summary(rows)

    timings: dict[str, dict[str, float | int]] = {}
    stages = sorted({stage for row in rows for stage in row.get("timings_ms", {})})
    for stage in stages:
        values = sorted(
            float(row["timings_ms"][stage]) for row in rows if stage in row.get("timings_ms", {})
        )
        if values:
            timings[stage] = {
                "mean": sum(values) / len(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "n": len(values),
            }
    route_groups = {
        label: [row for row in rows if row["expected_route"] == label]
        for label in ("STAT", "TACTICAL", "HYBRID")
    }
    languages = sorted({row.get("language") for row in rows if row.get("language")})
    language_groups = {
        language: [row for row in rows if row.get("language") == language] for language in languages
    }
    route_labels = ("STAT", "TACTICAL", "HYBRID", "ERROR")
    route_confusion = {
        expected: {
            actual: sum(
                row["expected_route"] == expected
                and (((row.get("result") or {}).get("route") or {}).get("label") or "ERROR")
                == actual
                for row in rows
            )
            for actual in route_labels
        }
        for expected in route_labels[:-1]
    }
    # A row whose every model call was served by the cache never reached a
    # provider, so its latencies describe a SQLite lookup. Reported next to the
    # timings because nothing else separates a warm rerun from a cold run.
    measured = [row for row in rows if row.get("model_calls") is not None]
    cached_items = sum(
        row["model_calls"] > 0 and row.get("cache_hits") == row["model_calls"] for row in measured
    )
    # One hit is enough to make a latency unusable: a route served from cache
    # and a synthesis that really ran sum to a number describing neither.
    contaminated = sum((row.get("cache_hits") or 0) > 0 for row in measured)
    return {
        "items": len(rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "cached_items": cached_items,
        "items_with_cache_hits": contaminated,
        # Every row measured, and not one of them touched the cache.
        "timings_trustworthy": bool(rows) and len(measured) == len(rows) and contaminated == 0,
        "judge_errors": sum(
            row.get("judge") is not None and not row["judge"].get("ok", False) for row in rows
        ),
        # A row nobody has graded yet is not a graded row that failed, so it
        # never reached judge_errors — and a limited rejudge could finish
        # reporting a clean mean over the handful it did get to. A row that
        # errored is not waiting for a judge; `--rejudge` skips it on purpose,
        # so counting it here would ask for a pass that can never clear it.
        "ungraded": sum(row.get("judge") is None and not row.get("error") for row in rows),
        "partial_results": sum(
            (row.get("result") or {}).get("status") == "partial" for row in rows
        ),
        "abstentions": sum(
            ((row.get("result") or {}).get("answer") or {}).get("abstained", False) for row in rows
        ),
        "items_with_route_repairs": sum(
            bool(((row.get("result") or {}).get("route") or {}).get("repairs")) for row in rows
        ),
        "translation_degradations": sum(
            ((row.get("result") or {}).get("translation") or {}).get("status")
            in {"degraded", "failed", "empty"}
            for row in rows
        ),
        "by_route": {label: len(group) for label, group in route_groups.items()},
        "route_confusion": route_confusion,
        "metrics": metrics,
        "slices": {
            "route": {
                label: {"items": len(group), "metrics": _metric_summary(group)}
                for label, group in route_groups.items()
            },
            "language": {
                language: {"items": len(group), "metrics": _metric_summary(group)}
                for language, group in language_groups.items()
            },
        },
        "timings_ms": timings,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
