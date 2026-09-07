"""Run the labelled set and checkpoint an auditable raw report."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.callbacks import BaseCallbackHandler

from tactistat import __version__
from tactistat.artifacts import project_relative, stable_hash, write_json_atomic
from tactistat.config import Config
from tactistat.eval.judge import JudgeResult, LLMJudge, quota_exhausted
from tactistat.eval.metrics import (
    JUDGE_METRICS,
    aggregate,
    failure_scores,
    score_result,
    stats_exact_match,
)
from tactistat.eval.schema import EvaluationItem, EvaluationSet, load_test_set
from tactistat.pipeline import PipelineResult, TactiStatPipeline

# Every trailing rejudge stamp, so a third pass names itself after the original
# run rather than after the second pass.
REJUDGE_SUFFIX_RE = re.compile(r"(?:-rejudge-\d{8}T\d+Z)+$")

Progress = Callable[[int, int, EvaluationItem], None]
RejudgeProgress = Callable[[int, int, str], None]


class ModelCallCounter(BaseCallbackHandler):
    """Count model invocations for one item."""

    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> int:
        previous, self.calls = self.calls, 0
        return previous

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.calls += 1

    def on_llm_start(self, *args, **kwargs) -> None:
        self.calls += 1


class CountingCache(BaseCache):
    """Wrap the process LLM cache to tell a served answer from a computed one.

    The cache is on by default and a hit never reaches the provider, so a
    re-run of the same questions reports latencies that measure a SQLite
    lookup. The callbacks cannot see this: LangChain fires
    `on_chat_model_start` before it consults the cache, so a fully cached item
    still reports its model calls. Only the cache knows, so the count is taken
    here and every row carries it.
    """

    def __init__(self, inner: BaseCache):
        self.inner = inner
        self.hits = 0

    def reset(self) -> int:
        previous, self.hits = self.hits, 0
        return previous

    def lookup(self, prompt: str, llm_string: str) -> Any:
        value = self.inner.lookup(prompt, llm_string)
        if value is not None:
            self.hits += 1
        return value

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        self.inner.update(prompt, llm_string, return_val)

    def clear(self, **kwargs: Any) -> None:
        self.inner.clear(**kwargs)


def counting_cache() -> CountingCache | None:
    """Install the wrapper over whatever cache is currently configured.

    Called before every item because the registry installs its cache lazily,
    when a role's model is first built, which would otherwise replace this.
    """
    from langchain_core.globals import get_llm_cache, set_llm_cache

    cache = get_llm_cache()
    if cache is None:
        return None
    if not isinstance(cache, CountingCache):
        cache = CountingCache(cache)
        set_llm_cache(cache)
    return cache


def evaluation_settings(config: Config) -> tuple[list[int], bool]:
    depths = config["eval.recall_at_k"]
    if (
        not isinstance(depths, list)
        or not depths
        or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in depths)
        or len(set(depths)) != len(depths)
    ):
        raise ValueError("eval.recall_at_k must contain unique positive integers")
    top_k = config["rag.retrieval.top_k"]
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"rag.retrieval.top_k must be a positive integer, got {top_k!r}")
    if max(depths) > top_k:
        raise ValueError(
            f"eval.recall_at_k reaches {max(depths)}, but retrieval returns only top_k={top_k}"
        )
    return sorted(depths), bool(config.get("eval.judge.enabled", True))


def default_output_path(config: Config, run_id: str) -> Path:
    return config.path("eval.results_dir") / "raw" / f"{run_id}.json"


def audit_output_path(destination: Path) -> Path:
    """Where the tracked slice of a raw report goes: beside it, under `runs/`.

    Derived from the raw report's own location rather than from the configured
    results directory, so a run told to write somewhere else — a test's tmp
    directory, a scratch path — keeps both halves there instead of dropping one
    into the repository.
    """
    parent = destination.parent
    root = parent.parent if parent.name == "raw" else parent
    return root / "runs" / destination.name


# The audit slice replaces tool text with this, and anything that regrades an
# answer has to be able to recognise it in a file written by an older version
# or by hand — a flag alone only protects reports this build wrote.
AUDIT_PLACEHOLDER_RE = re.compile(r"^<\d+ characters; see the raw report>$")


def is_audit_slice(report: dict[str, Any]) -> bool:
    """Whether this report's tool evidence has been replaced by placeholders."""
    if report.get("audit_slice"):
        return True
    for row in report.get("rows") or []:
        result = (row or {}).get("result")
        if not isinstance(result, dict):
            continue
        for rendered in ("stats_context", "rag_context"):
            body = result.get(rendered)
            if isinstance(body, str) and AUDIT_PLACEHOLDER_RE.match(body):
                return True
        rag = result.get("rag_artifact")
        if isinstance(rag, dict):
            for passage in rag.get("passages") or []:
                if "text" not in passage and "chars" in passage:
                    return True
    return False


def _audit_passage(passage: dict[str, Any]) -> dict[str, Any]:
    text = passage.get("text") or ""
    return {**{key: value for key, value in passage.items() if key != "text"}, "chars": len(text)}


def _audit_row(row: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(row)
    result = row.get("result")
    if not isinstance(result, dict):
        return row
    for rendered in ("stats_context", "rag_context"):
        body = result.get(rendered)
        if isinstance(body, str):
            result[rendered] = f"<{len(body)} characters; see the raw report>"
    rag = result.get("rag_artifact")
    if isinstance(rag, dict):
        rag["passages"] = [_audit_passage(passage) for passage in rag.get("passages") or []]
    return row


def audit_slice(report: dict[str, Any]) -> dict[str, Any]:
    """The tracked half of a run: every score, note and judgement, no bodies.

    What makes a raw report too large to track is the retrieved text and the
    rendered tool context — the same Wikipedia paragraphs repeated across items.
    Everything needed to recheck a published table is small: the per-item
    scores, the guard's notes, the judge's rationale, and which passage came
    back at which rank. So that half is tracked rather than described, and the
    raw report stays a local convenience rather than the only evidence.
    """
    slim = {key: value for key, value in report.items() if key != "rows"}
    # Named, because the tool text in here is a placeholder. Anything that
    # regrades an answer has to read the raw report or it will grade the
    # placeholder and never notice.
    slim["audit_slice"] = True
    slim["rows"] = [_audit_row(row) for row in report.get("rows", [])]
    return slim


def write_audit_slice(report: dict[str, Any], destination: Path) -> Path:
    """Write the tracked slice next to a finished raw report."""
    audit_path = audit_output_path(destination)
    report["audit_path"] = project_relative(audit_path)
    write_json_atomic(audit_path, audit_slice(report))
    return audit_path


def validate_suite_config(config: Config, suite: EvaluationSet) -> None:
    """Reject ground truth that the configured stats tool cannot express."""
    configured_metrics = set(config["stats_tool.metrics"])
    labelled_metrics = {
        item.route.stats_args.metric for item in suite.items if item.route.stats_args is not None
    }
    unknown_metrics = sorted(labelled_metrics - configured_metrics)
    if unknown_metrics:
        raise ValueError(f"test set uses metrics disabled by config: {unknown_metrics}")


def validate_ground_truth_artifacts(config: Config, suite: EvaluationSet) -> dict[str, int]:
    """Check every label against the committed stats tables and corpus."""
    from tactistat.data.wikipedia import load_corpus
    from tactistat.stats_tool.query import StatsQueryEngine, run_stats_operation

    validate_suite_config(config, suite)
    engine = StatsQueryEngine(config)
    bad_stats = []
    stats_items = 0
    for item in suite.items:
        if item.stats is None or item.route.stats_args is None:
            continue
        stats_items += 1
        answer = run_stats_operation(engine, **item.route.stats_args.model_dump())
        if stats_exact_match(item, answer) != 1:
            bad_stats.append(item.id)

    corpus_sections: dict[str, dict[str | None, str]] = {}
    for page in load_corpus(config):
        for section in page.sections:
            corpus_sections.setdefault(page.title, {})[section.heading] = section.text
    labelled_targets = [
        target
        for item in suite.items
        for target in (item.retrieval.relevant if item.retrieval else [])
    ]
    missing_targets = []
    for target in labelled_targets:
        sections = corpus_sections.get(target.title, {})
        # A label may name no heading, which means any section of that page.
        candidates = (
            list(sections.values()) if target.heading is None else [sections.get(target.heading)]
        )
        wanted = [" ".join(anchor.casefold().split()) for anchor in target.text_contains]
        if not any(
            text is not None
            and all(anchor in " ".join(text.casefold().split()) for anchor in wanted)
            for text in candidates
        ):
            missing_targets.append((target.title, target.heading, target.text_contains))
    if bad_stats or missing_targets:
        details = []
        if bad_stats:
            details.append(f"numeric labels disagree for {bad_stats}")
        if missing_targets:
            details.append(f"corpus targets do not exist: {missing_targets}")
        raise ValueError("; ".join(details))
    unique_targets = {
        (target.title, target.heading, tuple(target.text_contains)) for target in labelled_targets
    }
    return {"stats_items": stats_items, "retrieval_targets": len(unique_targets)}


def _result_artifact(result: PipelineResult) -> dict[str, Any]:
    artifact = result.to_dict()
    artifact["stats_context"] = (
        result.stats.to_context() if result.stats is not None and result.stats.ok else None
    )
    artifact["stats_artifact"] = asdict(result.stats) if result.stats else None
    artifact["rag_artifact"] = asdict(result.rag) if result.rag else None
    artifact["rag_context"] = (
        result.rag.to_context() if result.rag is not None and result.rag.ok else None
    )
    return artifact


def _checkpointed_evidence(row: dict[str, Any]) -> str:
    """Recover exactly the tool text needed to judge an existing raw row."""
    result = row.get("result") or {}
    blocks = [result.get("stats_context"), result.get("rag_context")]
    if not blocks[1] and (artifact := result.get("rag_artifact")) and artifact.get("ok"):
        # Schema-v1 reports written before rag_context was added still contain
        # every Passage, so they remain rejudgeable.
        from tactistat.rag_tool.retrieve import Passage, RagAnswer

        rag = RagAnswer(
            query=artifact.get("query", ""),
            mode=artifact.get("mode", "unknown"),
            reranked=bool(artifact.get("reranked", False)),
            passages=[Passage(**passage) for passage in artifact.get("passages", [])],
            note=artifact.get("note"),
            ok=bool(artifact.get("ok", False)),
        )
        blocks[1] = rag.to_context()
    return "\n\n".join(block for block in blocks if block and block.strip()) or "NO TOOL EVIDENCE"


def _apply_judge(row: dict[str, Any], judged: JudgeResult, model: str) -> None:
    row["judge"] = {**judged.to_dict(), "model": model}
    metrics = judged.metrics()
    if ((row.get("result") or {}).get("answer") or {}).get("abstained", False):
        metrics["judge_faithfulness"] = None
    row["scores"].update(metrics)


def _abstention_judge() -> JudgeResult:
    """An answerable labelled item that was refused is a deterministic miss."""
    return JudgeResult(
        ok=True,
        correctness=0.0,
        completeness=0.0,
        faithfulness=None,
        rationale="deterministic: the system abstained on an answerable evaluation item",
    )


def rejudge_report(
    config: Config,
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
    item_ids: list[str] | None = None,
    limit: int | None = None,
    judge: LLMJudge | None = None,
    progress: RejudgeProgress | None = None,
) -> dict[str, Any]:
    """Fill missing/failed judge scores without rerunning the pipeline."""
    from tactistat.config import PROJECT_ROOT

    source = Path(source_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    try:
        original = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation report not found: {source}") from None
    if not isinstance(original.get("rows"), list):
        raise ValueError(f"Evaluation report has no rows: {source}")
    if is_audit_slice(original):
        # The slice keeps every score and rationale but replaces the retrieved
        # text with a character count, so judging it would score the placeholder
        # and report a number rather than an error.
        raw = original.get("output_path") or "the raw report it was written from"
        raise ValueError(
            f"{source} is a tracked audit slice, not the evidence: its tool context was "
            f"replaced by a character count. Rejudge {raw} instead."
        )
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")

    report = copy.deepcopy(original)
    rows = report["rows"]
    known = {row.get("id") for row in rows}
    wanted = set(item_ids or [])
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"unknown evaluation ids: {unknown}")
    target_model = config["models.judge"]
    original_model = ((report.get("config") or {}).get("models") or {}).get("judge")
    eligible = [row for row in rows if row.get("error") is None and row.get("result") is not None]
    for row in eligible:
        current = (row.get("judge") or {}).get("model") or original_model
        abstained = ((row.get("result") or {}).get("answer") or {}).get("abstained", False)
        if abstained and current != "deterministic":
            if row.get("judge") is not None:
                row.setdefault("judge_history", []).append(row["judge"])
            row["judge"] = {
                "ok": False,
                "error": "pending deterministic abstention grade",
                "model": "deterministic",
            }
            row["scores"].update({name: None for name in JUDGE_METRICS})
            continue
        if current in {target_model, "deterministic"}:
            continue
        if row.get("judge") is not None:
            row.setdefault("judge_history", []).append(row["judge"])
        row["judge"] = {
            "ok": False,
            "error": f"pending rejudge with {target_model}",
            "model": target_model,
        }
        row["scores"].update({name: None for name in JUDGE_METRICS})

    pending = [row for row in eligible if not ((row.get("judge") or {}).get("ok", False))]
    # Do untouched rows before retrying a row that already timed out. Otherwise
    # one provider-specific poison prompt can block every later resume pass.
    pending.sort(
        key=lambda row: (
            not any(
                marker in ((row.get("judge") or {}).get("error") or "")
                for marker in (
                    "pending rejudge",
                    "pending deterministic",
                    "skipped after quota",
                )
            )
        )
    )
    candidates = [row for row in pending if not wanted or row.get("id") in wanted][:limit]

    started = datetime.now(timezone.utc)  # noqa: UP017
    stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
    if output_path is None:
        # Rejudging a rejudge would otherwise chain suffixes until the name is
        # unreadable; lineage lives in `parent_report`, not in the filename.
        stem = REJUDGE_SUFFIX_RE.sub("", source.stem)
        destination = source.with_name(f"{stem}-rejudge-{stamp}.json")
    else:
        destination = Path(output_path)
        if not destination.is_absolute():
            destination = config_relative_output(config, destination)
    if destination.resolve() == source.resolve():
        raise ValueError("rejudge output must differ from its source report")

    run = {
        "started_at": started.isoformat(),
        "completed_at": None,
        "status": "running",
        "source_report": project_relative(source),
        "judge_model": target_model,
        "judge_config_fingerprint": stable_hash(
            {"model": config["models.judge"], "llm": config.section("llm")}
        ),
        "selected_rows": len(candidates),
        "attempted": 0,
        "succeeded": 0,
        "remaining": len(pending),
    }
    report.setdefault("rejudge_runs", []).append(run)
    report["judge_enabled"] = True
    report["parent_report"] = project_relative(source)
    report["output_path"] = project_relative(destination)
    report["status"] = "judging"
    write_json_atomic(destination, report)

    grader = judge or LLMJudge(config)
    for position, row in enumerate(candidates, start=1):
        if progress:
            progress(position, len(candidates), row["id"])
        result = row["result"]
        judged = (
            _abstention_judge()
            if result["answer"].get("abstained", False)
            else grader.score_evidence(
                row["question"],
                row["ground_truth"]["reference_answer"],
                result["answer"]["text"],
                _checkpointed_evidence(row),
                item_id=row["id"],
            )
        )
        method = "deterministic" if result["answer"].get("abstained", False) else target_model
        _apply_judge(row, judged, method)
        run["attempted"] += 1
        run["succeeded"] += int(judged.ok)
        run["remaining"] = sum(
            not ((candidate.get("judge") or {}).get("ok", False)) for candidate in eligible
        )
        report["summary"] = aggregate(rows)
        write_json_atomic(destination, report)
        if quota_exhausted(judged):
            run["status"] = "quota_exhausted"
            break

    if run["status"] == "running":
        run["status"] = "complete"
    run["completed_at"] = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    report["status"] = "complete"
    report["summary"] = aggregate(rows)
    write_audit_slice(report, destination)
    write_json_atomic(destination, report)
    return report


class EvaluationRunner:
    def __init__(
        self,
        config: Config,
        *,
        pipeline: TactiStatPipeline | None = None,
        judge: LLMJudge | None = None,
    ):
        self.config = config
        self.recall_at_k, self.judge_default = evaluation_settings(config)
        self.pipeline = pipeline or TactiStatPipeline(config)
        self._judge = judge
        self._judge_quota_error: str | None = None
        self.translation_scored = bool(config.get("query_translation.enabled", True))
        self.counter = ModelCallCounter()
        # Install and wrap the cache before any model exists. The registry
        # installs it lazily when a role's model is first built, which happens
        # *inside* the first item's run — wrapping only afterwards would leave
        # that item's hits uncounted and report a cached run as a measured one.
        # `configure_cache` returns early once the path matches, so every later
        # model build leaves the wrapper in place.
        from tactistat.llm.registry import configure_cache

        configure_cache(config)
        counting_cache()
        callbacks = getattr(self.pipeline, "callbacks", None)
        # A pipeline stand-in has nothing to attach to; then the count is
        # unknown rather than zero, which would read as "served from cache".
        self.counting = isinstance(callbacks, list)
        if self.counting:
            callbacks.append(self.counter)

    def _call_counts(self) -> tuple[int | None, int | None]:
        """Model invocations for this item, and how many the cache served."""
        cache = counting_cache()
        hits = None if cache is None else cache.reset()
        calls = self.counter.reset()
        return (calls, hits or 0) if self.counting else (None, None)

    @property
    def judge(self) -> LLMJudge:
        if self._judge is None:
            self._judge = LLMJudge(self.config)
        return self._judge

    def _score_with_judge(self, item: EvaluationItem, result: PipelineResult) -> JudgeResult:
        if result.answer.abstained:
            return _abstention_judge()
        if self._judge_quota_error is not None:
            return JudgeResult(
                ok=False,
                error="judge skipped after quota exhaustion earlier in this run",
            )
        judged = self.judge.score(item.question, item.reference_answer, result, item_id=item.id)
        if quota_exhausted(judged):
            self._judge_quota_error = judged.error
        return judged

    def run(
        self,
        test_set: EvaluationSet | None = None,
        *,
        limit: int | None = None,
        item_ids: list[str] | None = None,
        judge_enabled: bool | None = None,
        output_path: str | Path | None = None,
        progress: Progress | None = None,
    ) -> dict[str, Any]:
        suite = test_set or load_test_set(self.config.path("eval.test_set"))
        validate_suite_config(self.config, suite)
        if limit is not None and limit < 1:
            raise ValueError("limit must be a positive integer")
        items = suite.items
        if item_ids:
            wanted = set(item_ids)
            known = {item.id for item in suite.items}
            unknown = sorted(wanted - known)
            if unknown:
                raise ValueError(f"unknown evaluation ids: {unknown}")
            items = [item for item in items if item.id in wanted]
        items = items[:limit]
        use_judge = self.judge_default if judge_enabled is None else judge_enabled
        # timezone.utc keeps the package compatible with Python 3.10.
        started = datetime.now(timezone.utc)  # noqa: UP017
        run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + f"-{suite.fingerprint[:8]}"
        destination = Path(output_path) if output_path else default_output_path(self.config, run_id)
        if not destination.is_absolute():
            destination = config_relative_output(self.config, destination)

        report: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "started_at": started.isoformat(),
            "completed_at": None,
            "duration_ms": None,
            "package_version": __version__,
            "test_set": {
                "name": suite.name,
                "fingerprint": suite.fingerprint,
                "selected_items": len(items),
                "total_items": len(suite.items),
            },
            "config_fingerprint": stable_hash(self.config.to_dict()),
            "config": self.config.to_dict(),
            "judge_enabled": use_judge,
            "llm_cache_enabled": bool(self.config.get("llm.cache_enabled", False)),
            "recall_at_k": self.recall_at_k,
            "rows": [],
            "summary": None,
        }
        write_json_atomic(destination, report)

        for position, item in enumerate(items, start=1):
            if progress:
                progress(position, len(items), item)
            row = self._run_one(item, use_judge)
            report["rows"].append(row)
            report["summary"] = aggregate(report["rows"])
            write_json_atomic(destination, report)

        report["status"] = "complete"
        completed = datetime.now(timezone.utc)  # noqa: UP017
        report["completed_at"] = completed.isoformat()
        report["duration_ms"] = int((completed - started).total_seconds() * 1000)
        report["summary"] = aggregate(report["rows"])
        report["output_path"] = project_relative(destination)
        write_audit_slice(report, destination)
        write_json_atomic(destination, report)
        return report

    def _run_one(self, item: EvaluationItem, use_judge: bool) -> dict[str, Any]:
        base = {
            "id": item.id,
            "question": item.question,
            "language": item.language,
            "expected_route": item.route.label,
            "tags": item.tags,
            "ground_truth": {
                "route": item.route.model_dump(mode="json"),
                "stats": item.stats.model_dump(mode="json") if item.stats else None,
                "retrieval": (item.retrieval.model_dump(mode="json") if item.retrieval else None),
                "reference_answer": item.reference_answer,
            },
        }
        self.counter.reset()
        cache = counting_cache()
        if cache is not None:
            cache.reset()
        try:
            result = self.pipeline.run(item.question)
        except Exception as exc:  # noqa: BLE001 - preserve the rest of the run
            return {
                **base,
                "error": f"{type(exc).__name__}: {exc}",
                "scores": failure_scores(
                    item, self.recall_at_k, translation_scored=self.translation_scored
                ),
                "judge": None,
                **dict(zip(("model_calls", "cache_hits"), self._call_counts(), strict=True)),
                "timings_ms": {},
                "result": None,
            }

        model_calls, cache_hits = self._call_counts()
        # A stage that swallowed its own failure produced an answer, so nothing
        # raised; the row still has to say the system was not measured here.
        stage_errors = result.stage_errors
        judged = self._score_with_judge(item, result) if use_judge and not stage_errors else None
        metrics = judged.metrics() if judged is not None else None
        return {
            **base,
            "error": (
                "; ".join(f"{stage}: {detail}" for stage, detail in stage_errors.items())
                if stage_errors
                else None
            ),
            "stage_errors": stage_errors,
            "scores": score_result(item, result, self.recall_at_k, metrics),
            "judge": (
                {
                    **judged.to_dict(),
                    "model": (
                        "deterministic" if result.answer.abstained else self.config["models.judge"]
                    ),
                }
                if judged is not None
                else None
            ),
            # cache_hits == model_calls means nothing reached a provider, so
            # this row's timings measure the cache and not the system.
            "model_calls": model_calls,
            "cache_hits": cache_hits,
            "timings_ms": result.timings_ms,
            "result": _result_artifact(result),
        }


def config_relative_output(config: Config, path: Path) -> Path:
    """Put a bare filename under results_dir; preserve explicit subdirectories."""
    if path.parent == Path("."):
        return config.path("eval.results_dir") / "raw" / path
    from tactistat.config import PROJECT_ROOT

    return PROJECT_ROOT / path
