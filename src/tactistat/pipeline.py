"""Implement the stages wrapped by the LangGraph pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from tactistat.config import Config
from tactistat.query_translation.translate import QueryTranslator, TranslatedQuery
from tactistat.rag_tool.retrieve import RagAnswer, RagRetriever
from tactistat.router.route import Route, Router
from tactistat.stats_tool.query import (
    StatsAnswer,
    StatsQueryEngine,
    claimable_values,
    run_stats_operation,
)
from tactistat.synthesis.synthesize import Answer, Synthesizer
from tactistat.tracing import configure_tracing


@dataclass
class PipelineResult:
    """Every stage's output, so an evaluation can score them separately."""

    question: str
    translation: TranslatedQuery
    route: Route
    answer: Answer
    stats: StatsAnswer | None = None
    rag: RagAnswer | None = None
    retrieval_queries: list[str] = field(default_factory=list)
    tool_failures: list[str] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)
    trace_run_id: str | None = None

    @property
    def status(self) -> str:
        """Return partial when a valid answer used only part of its routed evidence."""
        if not self.answer.ok:
            return "failed"
        return "partial" if self.tool_failures else "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def stage_errors(self) -> dict[str, str]:
        """Stages that could not run, as opposed to running and finding nothing.

        Every stage here swallows its own exception so one bad question cannot
        end a sweep, which is right for answering and wrong for scoring: a
        retriever whose index is gone returns no passages, and scoring that as
        zero recall reports an outage as a weak retriever. Anything that
        averages these rows has to drop them instead.
        """
        broken: dict[str, str] = {}
        if self.translation.status in {"failed", "empty"}:
            broken["translation"] = self.translation.note or self.translation.status
        model_failure = next(
            (repair for repair in self.route.repairs if repair.startswith("router model failed")),
            None,
        )
        if model_failure:
            broken["router"] = model_failure
        if self.stats is not None and self.stats.failed:
            broken["stats"] = self.stats.note or "stats tool failed"
        if self.rag is not None and self.rag.failed:
            broken["rag"] = self.rag.note or "rag tool failed"
        if self.answer.failed:
            broken["synthesis"] = self.answer.note or "synthesis failed"
        return broken

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "translation": self.translation.to_dict(),
            "route": self.route.to_dict(),
            "retrieval_queries": self.retrieval_queries,
            "status": self.status,
            "stats_ok": None if self.stats is None else self.stats.ok,
            "rag_passages": 0 if self.rag is None else len(self.rag.passages),
            "tool_failures": self.tool_failures,
            "answer": self.answer.to_dict(),
            "timings_ms": self.timings_ms,
            "trace_run_id": self.trace_run_id,
        }


def retrieval_queries_for(route: Route, translation: TranslatedQuery) -> list[str]:
    """Preserve the translation ablation when choosing RAG queries.

    Disabled translation uses the raw question; failures may use the router's
    fallback. Otherwise the selected strategy owns retrieval.
    """
    if route.rag_query is None:
        return []
    if any("promoted STAT to HYBRID" in repair for repair in route.repairs):
        # Translation lost the prose half; searching its variants would repeat
        # the same loss. The router preserved the original intent as rag_query.
        return [route.rag_query]
    if translation.status == "disabled":
        return [translation.original]
    if not translation.translated:
        return [route.rag_query]
    return translation.retrieval_queries or [route.rag_query]


class TactiStatPipeline:
    """Question in, cited answer out, with every intermediate result kept."""

    def __init__(
        self,
        config: Config,
        *,
        translator: QueryTranslator | None = None,
        router: Router | None = None,
        engine: StatsQueryEngine | None = None,
        retriever: RagRetriever | None = None,
        synthesizer: Synthesizer | None = None,
    ):
        self.config = config
        self.translator = translator or QueryTranslator(config)
        self.engine = engine
        self.retriever = retriever
        self.synthesizer = synthesizer or Synthesizer(config)
        # The keyword baseline needs team names to avoid calling a country a player.
        self.router = router or Router(config, known_teams=self._team_names())
        self.callbacks = configure_tracing(config)
        self._graph = None
        self._conversation = None

    def _team_names(self) -> set[str]:
        try:
            return set(self.stats_engine.player_totals["team"].astype(str).unique())
        except Exception:  # noqa: BLE001 - routing must not require built stats
            return set()

    @property
    def stats_engine(self) -> StatsQueryEngine:
        if self.engine is None:
            self.engine = StatsQueryEngine(self.config)
        return self.engine

    @property
    def rag_retriever(self) -> RagRetriever:
        if self.retriever is None:
            self.retriever = RagRetriever(self.config)
        return self.retriever

    def _run_stats(self, route: Route) -> StatsAnswer:
        """A tool that raises becomes a refusal, not a crashed run."""
        try:
            return run_stats_operation(self.stats_engine, **route.stats_args)
        except Exception as exc:  # noqa: BLE001 - one bad question must not end a sweep
            metric = (route.stats_args or {}).get("metric") or "unknown"
            note = f"stats tool raised {type(exc).__name__}: {exc}"
            return StatsAnswer(metric, False, [], ok=False, note=note, failed=True)

    def _run_rag(self, queries: list[str]) -> RagAnswer:
        try:
            retriever = self.rag_retriever
            if len(queries) > 1:
                return retriever.search_multi(queries)
            return retriever.search(queries[0])
        except Exception as exc:  # noqa: BLE001 - as above
            note = f"rag tool raised {type(exc).__name__}: {exc}"
            # failed=True so evaluation can tell an index outage from a query
            # that legitimately matched nothing. Synthesis needs neither.
            return RagAnswer(queries[0], "unknown", False, ok=False, failed=True, note=note)

    def translate(self, question: str, history: list[dict] | None = None):
        return self.translator.translate(question, history=history or None)

    def route(self, query: str, intent_hint: str | None = None) -> Route:
        return self.router.route(query, intent_hint=intent_hint)

    def retrieval_queries(self, route: Route, translation: TranslatedQuery) -> list[str]:
        return retrieval_queries_for(route, translation)

    def run_stats(self, route: Route) -> StatsAnswer:
        return self._run_stats(route)

    def run_rag(self, queries: list[str]) -> RagAnswer:
        return self._run_rag(queries)

    def synthesize(
        self,
        question: str,
        translation: TranslatedQuery,
        route: Route,
        stats: StatsAnswer | None,
        rag: RagAnswer | None,
    ) -> Answer:
        """Exclude refused tool output from synthesis evidence."""
        return self.synthesizer.answer(
            question,
            stats_context=stats.to_context() if stats is not None and stats.ok else None,
            rag_context=rag.to_context() if rag is not None and rag.ok else None,
            n_passages=len(rag.passages) if rag is not None and rag.ok else 0,
            language=translation.source_language,
            stats_claims=claimable_values(stats) if stats is not None else None,
        )

    @property
    def history_limit(self) -> int:
        """Never keep fewer turns than the translator is configured to read."""
        from tactistat.graph import HISTORY_LIMIT

        return max(HISTORY_LIMIT, self.translator.history_turns)

    def _compiled(self, thread_id: str | None):
        """Use an isolated graph unless the caller explicitly names a conversation."""
        from tactistat.graph import build_graph, new_checkpointer

        invoke_config: dict[str, Any] = {
            "callbacks": self.callbacks,
            "run_name": "tactistat.ask",
            "metadata": {"tactistat_run_id": uuid4().hex},
        }
        if thread_id is None:
            if self._graph is None:
                self._graph = build_graph(self, history_limit=self.history_limit)
            return self._graph, invoke_config

        if self._conversation is None:
            self._conversation = build_graph(
                self, checkpointer=new_checkpointer(), history_limit=self.history_limit
            )
        invoke_config["configurable"] = {"thread_id": thread_id}
        invoke_config["metadata"]["thread_id"] = thread_id
        return self._conversation, invoke_config

    def run(self, question: str, thread_id: str | None = None) -> PipelineResult:
        """Answer one question. Pass a thread_id to continue a conversation."""
        graph, invoke_config = self._compiled(thread_id)
        state = graph.invoke({"question": question}, invoke_config)
        return self._result(
            question, state, trace_run_id=invoke_config["metadata"]["tactistat_run_id"]
        )

    def stream(self, question: str, thread_id: str | None = None):
        """Yield (node, state) as each stage finishes, then the final result."""
        graph, invoke_config = self._compiled(thread_id)
        # Isolated graphs have no checkpoint to read after streaming.
        final: dict[str, Any] = {}
        for update in graph.stream({"question": question}, invoke_config, stream_mode="updates"):
            for node, payload in update.items():
                final.update(payload)
                yield node, payload
        yield (
            "result",
            self._result(
                question, final, trace_run_id=invoke_config["metadata"]["tactistat_run_id"]
            ),
        )

    def _result(
        self, question: str, state: dict[str, Any], *, trace_run_id: str | None = None
    ) -> PipelineResult:
        from tactistat.graph import timings_from

        stats, rag = state.get("stats"), state.get("rag")
        failures = [
            f"{name}: {result.note}"
            for name, result in (("stats", stats), ("rag", rag))
            if result is not None and not result.ok
        ]
        return PipelineResult(
            question=question,
            translation=state["translation"],
            route=state["route"],
            answer=state["answer"],
            stats=stats,
            rag=rag,
            retrieval_queries=state.get("retrieval_queries") or [],
            tool_failures=failures,
            timings_ms=timings_from(state),
            trace_run_id=trace_run_id,
        )
