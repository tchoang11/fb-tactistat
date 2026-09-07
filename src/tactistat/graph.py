"""LangGraph orchestration for parallel tools, memory, and streaming."""

from __future__ import annotations

import time
from typing import Any, Protocol, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

NODES = ("translate", "route", "stats", "retrieve", "synthesize")

# Bounds visible history, not checkpoint versions; the in-process store still
# grows with conversation length and is discarded when the command exits.
HISTORY_LIMIT = 8


class Stages(Protocol):
    """What the graph needs from the pipeline; the contract lives there."""

    def translate(self, question: str, history: list[dict]) -> Any: ...
    def route(self, query: str, intent_hint: str | None = None) -> Any: ...
    def run_stats(self, route: Any) -> Any: ...
    def run_rag(self, queries: list[str]) -> Any: ...
    def retrieval_queries(self, route: Any, translation: Any) -> list[str]: ...
    def synthesize(self, question: str, translation, route, stats, rag) -> Any: ...


class GraphState(TypedDict, total=False):
    """One turn's working state; `history` is the only field that outlives it."""

    question: str
    history: list[dict]
    translation: Any
    route: Any
    stats: Any
    rag: Any
    retrieval_queries: list[str]
    answer: Any
    # Separate keys avoid concurrent writes; None means the branch did not run.
    t_translate: int | None
    t_route: int | None
    t_stats: int | None
    t_retrieve: int | None
    t_synthesize: int | None


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _all_evidence_arrived(state: GraphState) -> bool:
    """Did every tool the route asked for return usable evidence?"""
    stats, rag = state.get("stats"), state.get("rag")
    if state["route"].needs_stats and not (stats is not None and stats.ok):
        return False
    return not (state.get("retrieval_queries") and not (rag is not None and rag.ok))


def build_graph(
    stages: Stages,
    checkpointer: Any | None = None,
    history_limit: int = HISTORY_LIMIT,
):
    """Compile the graph. Pass a checkpointer to keep conversations."""

    def translate_node(state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        translation = stages.translate(state["question"], state.get("history") or [])
        return {"translation": translation, "t_translate": _elapsed_ms(started)}

    def route_node(state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        route = stages.route(state["translation"].query, intent_hint=state["question"])
        queries = stages.retrieval_queries(route, state["translation"])
        # Never carry tool evidence into the next turn.
        return {
            "route": route,
            "retrieval_queries": queries,
            "stats": None,
            "rag": None,
            "t_stats": None,
            "t_retrieve": None,
            "t_route": _elapsed_ms(started),
        }

    def stats_node(state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        return {"stats": stages.run_stats(state["route"]), "t_stats": _elapsed_ms(started)}

    def retrieve_node(state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        answer = stages.run_rag(state["retrieval_queries"])
        return {"rag": answer, "t_retrieve": _elapsed_ms(started)}

    def synthesize_node(state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        translation = state["translation"]
        answer = stages.synthesize(
            state["question"], translation, state["route"], state.get("stats"), state.get("rag")
        )
        history = state.get("history") or []
        # Failed or partial answers must not become follow-up context.
        if answer.ok and _all_evidence_arrived(state):
            turn = {
                "question": state["question"],
                "query": translation.query,
                "answer": answer.text,
            }
            history = [*history, turn][-history_limit:]
        return {
            "answer": answer,
            "history": history,
            "t_synthesize": _elapsed_ms(started),
        }

    def branches(state: GraphState) -> list[str]:
        """Both tools for HYBRID, and they run in the same superstep."""
        route = state["route"]
        targets = []
        if route.needs_stats:
            targets.append("stats")
        if state.get("retrieval_queries"):
            targets.append("retrieve")
        return targets or ["synthesize"]

    graph = StateGraph(GraphState)
    graph.add_node("translate", translate_node)
    graph.add_node("route", route_node)
    graph.add_node("stats", stats_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "translate")
    graph.add_edge("translate", "route")
    graph.add_conditional_edges("route", branches, ["stats", "retrieve", "synthesize"])
    graph.add_edge("stats", "synthesize")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer)


# Allow only the result types stored in checkpoints.
CHECKPOINT_TYPES = (
    ("tactistat.query_translation.translate", "TranslatedQuery"),
    ("tactistat.router.route", "Route"),
    ("tactistat.stats_tool.query", "StatsAnswer"),
    ("tactistat.stats_tool.query", "PlayerRow"),
    ("tactistat.stats_tool.query", "TotalRow"),
    ("tactistat.stats_tool.query", "MatchRef"),
    ("tactistat.stats_tool.bootstrap", "Interval"),
    ("tactistat.rag_tool.retrieve", "RagAnswer"),
    ("tactistat.rag_tool.retrieve", "Passage"),
    ("tactistat.synthesis.synthesize", "Answer"),
)


def new_checkpointer():
    """In-process conversation memory; a durable store is not needed yet."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


def timings_from(state: dict[str, Any]) -> dict[str, int]:
    """Per-node timings, omitting the branches this question did not take."""
    timings = {node: int(value) for node in NODES if (value := state.get(f"t_{node}")) is not None}
    # Wall clock, not the sum: for HYBRID the two tools share a superstep.
    concurrent = max(timings.get("stats", 0), timings.get("retrieve", 0))
    serial = sum(v for k, v in timings.items() if k not in ("stats", "retrieve"))
    return {**timings, "total": serial + concurrent}
