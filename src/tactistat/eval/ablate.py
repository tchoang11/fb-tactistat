"""Sweep one configuration axis at a time and compare the arms.

Two run modes, because the axes cost very different amounts to measure.

`retrieval` runs translate -> route -> retrieve and scores only the retrieval
metrics. It answers what a chunking, embedding, retrieval-mode or reranking
change does to recall, without paying for synthesis or a judge. Holding the
upstream stages fixed is what makes the comparison an ablation rather than two
unrelated runs, and the LLM cache is doing that work: the translate and route
prompts are identical across arms of a retrieval axis, so every arm after the
first receives exactly the same query. For the translation axis the prompt
differs by construction, so nothing is shared and nothing should be.

`pipeline` runs the whole labelled evaluation per arm. It is the expensive mode
and belongs to axes that reach synthesis.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from tactistat import __version__
from tactistat.artifacts import project_relative, stable_hash, write_json_atomic
from tactistat.config import Config
from tactistat.eval.metrics import retrieval_scores
from tactistat.eval.runner import EvaluationRunner, evaluation_settings
from tactistat.eval.schema import EvaluationItem, EvaluationSet, load_test_set

MODES = ("retrieval", "pipeline")

# A translator that refused the call produced no measurement of the strategy.
# "degraded" is not here on purpose: a strategy that returned fewer variants
# than it promised is the strategy's own behaviour, and belongs in the mean.
TRANSLATION_FAILURES = frozenset({"failed", "empty"})


@dataclass(frozen=True)
class Arm:
    """One point on an axis: a name and the config it overrides."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Axis:
    """One question the study asks, and the arms that answer it."""

    name: str
    mode: str
    question: str
    arms: tuple[Arm, ...]
    # Arms that change how documents are chunked or embedded need their own
    # index; sharing one directory would silently compare an arm against
    # another arm's vectors.
    rebuilds_index: bool = False


AXES: dict[str, Axis] = {
    "chunking": Axis(
        name="chunking",
        mode="retrieval",
        question="Do Wikipedia section boundaries retrieve better than fixed windows?",
        arms=(
            Arm("section", {"rag.chunking.strategy": "section"}),
            Arm("fixed", {"rag.chunking.strategy": "fixed"}),
        ),
        rebuilds_index=True,
    ),
    "retrieval_mode": Axis(
        name="retrieval_mode",
        mode="retrieval",
        question="Does hybrid retrieval beat dense or BM25 alone?",
        arms=(
            Arm("dense", {"rag.retrieval.mode": "dense"}),
            Arm("bm25", {"rag.retrieval.mode": "bm25"}),
            Arm("hybrid", {"rag.retrieval.mode": "hybrid"}),
        ),
    ),
    "rerank": Axis(
        name="rerank",
        mode="retrieval",
        question="Does a cross-encoder reranker earn its latency?",
        arms=(
            Arm("off", {"rag.rerank.enabled": False}),
            Arm("on", {"rag.rerank.enabled": True}),
        ),
    ),
    "translation": Axis(
        name="translation",
        mode="retrieval",
        question="Which query-translation strategy retrieves best?",
        arms=(
            Arm("disabled", {"query_translation.enabled": False}),
            Arm("off", {"query_translation.strategy": "off"}),
            Arm("rewrite", {"query_translation.strategy": "rewrite"}),
            Arm("multi_query", {"query_translation.strategy": "multi_query"}),
            Arm("hyde", {"query_translation.strategy": "hyde"}),
        ),
    ),
    # The translation axis above runs against the LLM cache, which is what holds
    # the upstream stages fixed between arms. That also means every number in it
    # is conditioned on one draw of the translator. These two axes take a second
    # draw with the cache off, to measure how much of the axis is the strategy
    # and how much is the provider. Split in two because 120 uncached calls does
    # not fit inside the translator's free tier in one sitting.
    "translation_uncached": Axis(
        name="translation_uncached",
        mode="retrieval",
        question="Do the generative translation arms reproduce against a live provider?",
        arms=(
            Arm(
                "multi_query",
                {"query_translation.strategy": "multi_query", "llm.cache_enabled": False},
            ),
            Arm("hyde", {"query_translation.strategy": "hyde", "llm.cache_enabled": False}),
        ),
    ),
    "translation_uncached_single": Axis(
        name="translation_uncached_single",
        mode="retrieval",
        question="Do the single-query translation arms reproduce against a live provider?",
        arms=(
            Arm("off", {"query_translation.strategy": "off", "llm.cache_enabled": False}),
            Arm("rewrite", {"query_translation.strategy": "rewrite", "llm.cache_enabled": False}),
        ),
    ),
    "router": Axis(
        name="router",
        mode="pipeline",
        question="Does the few-shot router beat the keyword baseline end to end?",
        arms=(
            Arm("few_shot", {"router.strategy": "few_shot"}),
            Arm("keyword", {"router.strategy": "keyword"}),
        ),
    ),
    "baseline": Axis(
        name="baseline",
        mode="pipeline",
        question="What does the shipped configuration score?",
        arms=(Arm("shipped", {}),),
    ),
}


def arm_config(base: Config, arm: Arm, *, index_root: Path | None = None) -> Config:
    """The shipped config with this arm's overrides, and its own index if needed."""
    config = Config(base.to_dict())
    for path, value in arm.overrides.items():
        config.set(path, value)
    if index_root is not None:
        # Repo-relative, because this value is written into the artifact: an
        # absolute path names a directory on the machine that ran the sweep.
        # Config.path() resolves a relative value against the project root.
        config.set("rag.index_dir", project_relative(index_root / arm.name))
    return config


def config_diff(base: dict[str, Any], arm: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Every dotted path where the arm's effective config differs from the base.

    Declared overrides are what the axis asked for; this is what it got. An
    index-rebuilding arm also has its own `rag.index_dir`, which no override
    mentions, so reconstructing an arm from base plus declared overrides alone
    would quietly rebuild it against the wrong vectors.
    """
    changed: dict[str, Any] = {}
    for key in sorted(set(base) | set(arm)):
        path = f"{prefix}{key}"
        left, right = base.get(key), arm.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            changed.update(config_diff(left, right, f"{path}."))
        elif left != right:
            changed[path] = right
    return changed


def retrieval_items(suite: EvaluationSet) -> list[EvaluationItem]:
    """Only items with labelled relevant passages can score retrieval."""
    return [item for item in suite.items if item.retrieval is not None]


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _percentile(values: list[float], percent: float) -> float | None:
    """Nearest-rank percentile; thirty items is too few to interpolate over."""
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(percent / 100.0 * len(ordered))
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]


def _passage_refs(passages: list[Any]) -> list[dict[str, Any]]:
    """What a reader needs to recheck a recall score without rerunning the arm."""
    return [
        {
            "rank": passage.rank,
            "title": passage.title,
            "heading": passage.heading,
            "chunk_id": passage.chunk_id,
        }
        for passage in passages
    ]


def score_retrieval_arm(
    config: Config,
    items: list[EvaluationItem],
    recall_at_k: list[int],
    *,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """Run translate -> route -> retrieve for each item and average the metrics.

    Two passes, because only the second one is timed. Translation and routing
    are model calls served from the LLM cache across arms of an axis, so their
    wall clock says how warm the cache was, not how fast the arm is. Retrieval
    is local compute — FAISS, BM25, the cross-encoder — and the cache never
    touches it, which is what makes `retrieval_ms` comparable between arms and
    lets the rerank axis answer its own question about latency.
    """
    from tactistat.pipeline import TactiStatPipeline

    pipeline = pipeline or TactiStatPipeline(config)

    prepared: list[dict[str, Any]] = []
    for item in items:
        row: dict[str, Any] = {"item": item, "queries": [], "error": None}
        try:
            translation = pipeline.translate(item.question)
            row["translation_status"] = translation.status
            row["translation_note"] = translation.note
            if translation.status in TRANSLATION_FAILURES:
                # The translator swallows a provider error and hands back the
                # untranslated question, which on a Vietnamese item retrieves
                # nothing. Scoring that is scoring an outage: a rate-limited
                # sweep would publish as "this strategy retrieves badly".
                detail = translation.note or "no detail recorded"
                row["error"] = (
                    detail
                    if detail.startswith("translation ")
                    else f"translation {translation.status}: {detail}"
                )
            else:
                route = pipeline.route(translation.query, intent_hint=item.question)
                row["queries"] = pipeline.retrieval_queries(route, translation)
                # Both are model calls. Without them in the artifact a rerun
                # that scores differently cannot be attributed to the stage that
                # actually changed, and the axis silently measures both.
                row["translated_query"] = translation.query
                row["route_label"] = route.label
        except Exception as failure:  # noqa: BLE001 - one item must not end the arm
            row["error"] = f"{type(failure).__name__}: {failure}"
        prepared.append(row)

    # The first retrieval of an arm also pays for lazily loading the embedding
    # model and the cross-encoder. Spend that once, before the clock starts.
    for row in prepared:
        if row["queries"]:
            try:
                pipeline.run_rag(row["queries"])
            except Exception:  # noqa: BLE001 - a real failure resurfaces below
                pass
            break

    rows: list[dict[str, Any]] = []
    for prep in prepared:
        item = prep["item"]
        scored: dict[str, Any] = {
            "id": item.id,
            "language": item.language,
            "queries": prep["queries"],
            "translation_status": prep.get("translation_status"),
            "translation_note": prep.get("translation_note"),
            "translated_query": prep.get("translated_query"),
            "route_label": prep.get("route_label"),
            "error": prep["error"],
            "retrieval_ms": None,
            "retrieved": 0,
            "passages": [],
            # An item that failed outright scores nothing. Zero would be a
            # measurement; this was an absence of one, and the two must not
            # average together.
            "scores": {},
        }
        if prep["error"] is None:
            try:
                started = perf_counter()
                # A routing miss is a real outcome of a translation arm, not an
                # error: scoring it as zero recall is what makes arms comparable.
                answer = pipeline.run_rag(prep["queries"]) if prep["queries"] else None
                elapsed = (perf_counter() - started) * 1000.0
                if answer is not None and answer.failed:
                    # The tool did not run. Its elapsed time is how long the
                    # failure took, which would drag the arm's p50 down.
                    scored["error"] = f"rag tool failed: {answer.note}"
                else:
                    scored["retrieval_ms"] = elapsed
                    passages = answer.passages if answer is not None and answer.ok else []
                    scored["retrieved"] = len(passages)
                    scored["passages"] = _passage_refs(passages)
                    scored["scores"] = retrieval_scores(item, passages, recall_at_k)
            except Exception as failure:  # noqa: BLE001 - one item must not end the arm
                scored["error"] = f"{type(failure).__name__}: {failure}"
                scored["retrieval_ms"] = None
        rows.append(scored)

    names = sorted({name for row in rows for name in row["scores"]})
    latencies = [row["retrieval_ms"] for row in rows if row["retrieval_ms"] is not None]
    return {
        "items": len(rows),
        "metrics": {
            name: {
                "mean": _mean([row["scores"].get(name) for row in rows]),
                "n": sum(row["scores"].get(name) is not None for row in rows),
            }
            for name in names
        },
        "retrieval_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "n": len(latencies),
        },
        # "degraded" and "failed" are not the same event and must not be one
        # count: a strategy that produced nothing is a property of the strategy,
        # while a provider that refused the call is weather. Reading one number
        # as the other is how a rate-limited sweep gets published as a finding.
        "translation_status_counts": dict(
            Counter(row["translation_status"] for row in rows if row["translation_status"])
        ),
        "translation_degradations": sum(
            row["translation_status"] in {"degraded", "failed", "empty"} for row in rows
        ),
        "errors": sum(row["error"] is not None for row in rows),
        # A route that sent the question away from RAG is a real outcome, but it
        # is the router's outcome, not the translation strategy's. Counting it
        # separately is what keeps a rerun's delta attributable.
        "routed_away_from_rag": sum(row["error"] is None and not row["queries"] for row in rows),
        "rows": rows,
    }


def run_axis(
    config: Config,
    axis: Axis,
    *,
    suite: EvaluationSet | None = None,
    output_path: str | Path | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Run every arm of one axis and write a comparable report."""
    suite = suite or load_test_set(config.path("eval.test_set"))
    recall_at_k, _ = evaluation_settings(config)
    started = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility
    # Under the configured index dir, which is already git-ignored.
    index_root = config.path("rag.index_dir") / "ablation" if axis.rebuilds_index else None

    report: dict[str, Any] = {
        "schema_version": 2,
        "axis": axis.name,
        "mode": axis.mode,
        "question": axis.question,
        # An interrupted sweep and a finished one otherwise differ only by a
        # missing key, which is not a difference a reader notices.
        "status": "running",
        "started_at": started.isoformat(),
        "package_version": __version__,
        "test_set": {"name": suite.name, "fingerprint": suite.fingerprint},
        "base_config_fingerprint": stable_hash(config.to_dict()),
        # Every arm is this config plus its own overrides, so recording it once
        # makes each arm's effective settings reconstructible from the artifact
        # rather than from whichever configs/ file happens to be checked out.
        "base_config": config.to_dict(),
        "recall_at_k": recall_at_k,
        "arms": [],
    }
    destination = _destination(config, axis, output_path)
    write_json_atomic(destination, report)

    items = retrieval_items(suite)
    for position, arm in enumerate(axis.arms, start=1):
        if progress:
            progress(position, len(axis.arms), arm)
        arm_cfg = arm_config(config, arm, index_root=index_root)
        entry: dict[str, Any] = {
            "name": arm.name,
            "overrides": arm.overrides,
            # What the arm asked for, and what it actually differs by. They are
            # the same on most axes and are not on the ones that build an index.
            "effective_overrides": config_diff(config.to_dict(), arm_cfg.to_dict()),
            "config_fingerprint": stable_hash(arm_cfg.to_dict()),
            "error": None,
        }
        try:
            entry.update(_run_arm(arm_cfg, axis, arm, items, suite, recall_at_k))
        except Exception as failure:  # noqa: BLE001 - one arm must not end the sweep
            # A comparison missing an arm is still a usable comparison; a
            # comparison that silently lost one is not.
            entry["error"] = f"{type(failure).__name__}: {failure}"
            entry.setdefault("metrics", {})
        report["arms"].append(entry)
        write_json_atomic(destination, report)

    completed = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility
    report["status"] = "complete"
    report["completed_at"] = completed.isoformat()
    report["duration_ms"] = int((completed - started).total_seconds() * 1000)
    report["arm_errors"] = sum(arm["error"] is not None for arm in report["arms"])
    report["output_path"] = project_relative(destination)
    write_json_atomic(destination, report)
    return report


def _run_arm(
    arm_cfg: Config,
    axis: Axis,
    arm: Arm,
    items: list[EvaluationItem],
    suite: EvaluationSet,
    recall_at_k: list[int],
) -> dict[str, Any]:
    """One arm's measurements, whichever mode the axis runs in."""
    if axis.rebuilds_index:
        from tactistat.rag_tool.index import build_index

        build_index(arm_cfg)
    if axis.mode == "retrieval":
        return score_retrieval_arm(arm_cfg, items, recall_at_k)

    run = EvaluationRunner(arm_cfg).run(suite, output_path=None)
    return {
        "items": run["summary"]["items"],
        "metrics": run["summary"]["metrics"],
        "errors": run["summary"]["errors"],
        # A judge that ran out of quota leaves the deterministic scores intact
        # and the graded ones on a denominator of almost nothing. Carried up so
        # the caller can say so instead of printing a mean over six items as if
        # it were a result.
        "judge_errors": run["summary"]["judge_errors"],
        "abstentions": run["summary"]["abstentions"],
        "timings_trustworthy": run["summary"]["timings_trustworthy"],
        "timings_ms": run["summary"]["timings_ms"],
        # Both tracked, and both relative: an absolute path names a directory
        # on the machine that produced the run.
        "report_path": run.get("audit_path"),
        "raw_report_path": run["output_path"],
    }


def _destination(config: Config, axis: Axis, output_path: str | Path | None) -> Path:
    """Where the axis report goes; a bare filename joins the other reports."""
    ablations = config.path("eval.results_dir") / "ablations"
    if output_path is None:
        return ablations / f"{axis.name}.json"
    path = Path(output_path)
    if path.is_absolute():
        return path
    # A bare name means "one of the reports"; a path with a directory in it
    # means the caller chose a location. Dropping `foo.json` in the repo root
    # is neither, and it is what this did.
    if path.parent == Path("."):
        return ablations / path
    from tactistat.config import PROJECT_ROOT

    return PROJECT_ROOT / path


def comparison_table(report: dict[str, Any], metrics: list[str] | None = None) -> str:
    """Render one axis as a Markdown table, best arm first on the lead metric."""
    arms = report["arms"]
    if not arms:
        return "_no arms_"
    names = metrics or sorted({name for arm in arms for name in arm.get("metrics") or {}})
    if not names:
        # Every arm failed. The table has nothing to rank, and the reason each
        # one has nothing is the only useful thing left to print.
        lines = [f"**{report['question']}**", "", "_no arm produced a metric._", ""]
        lines += [f"- `{arm['name']}`: {arm.get('error') or 'no metrics'}" for arm in arms]
        return "\n".join(lines)
    lead = names[0]

    def value(arm: dict[str, Any], name: str) -> float | None:
        return ((arm.get("metrics") or {}).get(name) or {}).get("mean")

    ordered = sorted(arms, key=lambda arm: (value(arm, lead) is None, -(value(arm, lead) or 0)))
    header = "| arm | " + " | ".join(names) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in names) + " |"
    lines = [f"**{report['question']}**", "", header, divider]
    for arm in ordered:
        cells = ["n/a" if (shown := value(arm, name)) is None else f"{shown:.3f}" for name in names]
        lines.append(f"| `{arm['name']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)
