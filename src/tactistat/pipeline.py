"""Wire translation, routing, the two tools and synthesis into one answer.

Deliberately a plain function pipeline: each stage is separately testable, and
the LangGraph graph wraps these same stages as nodes rather than reimplementing
them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

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

    @property
    def status(self) -> str:
        """ok | partial | failed.

        A HYBRID question that lost one branch was answered from half the
        evidence it asked for; reporting that as success would let the
        evaluation count it alongside answers that had everything.
        """
        if not self.answer.ok:
            return "failed"
        return "partial" if self.tool_failures else "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"

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
        }


def retrieval_queries_for(route: Route, translation: TranslatedQuery) -> list[str]:
    """Decide what the RAG tool actually searches for.

    The translation strategy is an ablation axis, so it owns this decision: if
    the router's own phrasing replaced the `off` arm's literal translation, the
    arms would retrieve identically and the axis would measure nothing.

    When translation is *disabled* the raw question is used verbatim, because
    the router's rewrite is itself an LLM translation and would quietly restore
    the step the baseline arm exists to remove. A translation *failure* may fall
    back to the router, and `translation.status` records which happened.
    """
    if route.rag_query is None:
        return []
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
            return StatsAnswer(metric, False, [], ok=False, note=note)

    def _run_rag(self, queries: list[str]) -> RagAnswer:
        try:
            retriever = self.rag_retriever
            if len(queries) > 1:
                return retriever.search_multi(queries)
            return retriever.search(queries[0])
        except Exception as exc:  # noqa: BLE001 - as above
            note = f"rag tool raised {type(exc).__name__}: {exc}"
            return RagAnswer(queries[0], "unknown", False, ok=False, note=note)

    def run(self, question: str) -> PipelineResult:
        """Answer one question, recording what each stage decided."""
        timings: dict[str, int] = {}

        def timed(name: str, work):
            started = time.perf_counter()
            try:
                return work()
            finally:
                timings[name] = int((time.perf_counter() - started) * 1000)

        translation = timed("translate", lambda: self.translator.translate(question))
        route = timed("route", lambda: self.router.route(translation.query))

        stats: StatsAnswer | None = None
        if route.needs_stats:
            stats = timed("stats", lambda: self._run_stats(route))

        rag: RagAnswer | None = None
        queries = retrieval_queries_for(route, translation)
        if queries:
            rag = timed("retrieve", lambda: self._run_rag(queries))

        # A tool that refused is not evidence: its "no result" line would
        # otherwise be summarised into a confident answer.
        failures = [
            f"{name}: {result.note}"
            for name, result in (("stats", stats), ("rag", rag))
            if result is not None and not result.ok
        ]
        # Answer the question as it was asked, in the language it was asked in.
        answer = timed(
            "synthesize",
            lambda: self.synthesizer.answer(
                question,
                stats_context=stats.to_context() if stats is not None and stats.ok else None,
                rag_context=rag.to_context() if rag is not None and rag.ok else None,
                n_passages=len(rag.passages) if rag is not None and rag.ok else 0,
                language=translation.source_language,
                stats_claims=claimable_values(stats) if stats is not None else None,
            ),
        )
        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        return PipelineResult(
            question=question,
            translation=translation,
            route=route,
            answer=answer,
            stats=stats,
            rag=rag,
            retrieval_queries=queries,
            tool_failures=failures,
            timings_ms=timings,
        )
