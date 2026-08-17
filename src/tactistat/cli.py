"""Command-line entry point for data builds and structured stats queries."""

from __future__ import annotations

import argparse

from tactistat.config import load_config
from tactistat.data.statsbomb import build_raw_dataset
from tactistat.data.wikipedia import build_corpus
from tactistat.pipeline import TactiStatPipeline
from tactistat.query_translation.translate import STRATEGIES
from tactistat.rag_tool.index import build_index
from tactistat.rag_tool.retrieve import MODES, RagRetriever
from tactistat.stats_tool.aggregate import build_player_matches
from tactistat.stats_tool.minutes import match_length
from tactistat.stats_tool.query import StatsQueryEngine, build_stats_tables


def _check_minutes(config, player_matches) -> bool:
    """Return whether aggregate player-minutes stay within match capacity."""
    difference = player_matches.groupby("match_id")["minutes"].sum() - match_length(config) * 22
    over = difference[difference > 0.01]
    print(f"matches checked: {len(difference)}; over-counted: {len(over)}")
    return over.empty


# User-facing labels for streamed graph nodes.
NODE_LABEL = {
    "translate": "understanding the question",
    "route": "choosing a tool",
    "stats": "computing statistics",
    "retrieve": "searching Wikipedia",
    "synthesize": "writing the answer",
}


def _report(result, trace: bool) -> int:
    """Print an answer with its caveats, its sources, and an honest exit code."""
    if trace:
        translation = result.translation
        print(f"[{translation.source_language}/{translation.status}] {translation.query}")
        if translation.note:
            print(f"  translation: {translation.note}")
        print(f"{result.route.label} {result.route.stats_args or ''}")
        for query in result.retrieval_queries:
            print(f"  search: {query[:100]}")
        for repair in result.route.repairs:
            print(f"  repair: {repair}")
        print(f"  {result.timings_ms}\n")

    for failure in result.tool_failures:
        print(f"! tool failed — {failure}")
    # Never present rejected text as a verified answer.
    if not result.answer.ok:
        print(f"! UNVERIFIED — {result.answer.note}")
        print("! the text below did not pass the evidence checks\n")
    elif result.status == "partial":
        print("! partial — one tool failed; the answer uses the rest\n")

    print(result.answer.text)

    # Resolve bracket ranks into usable sources.
    if result.answer.cited and result.rag is not None:
        print("\nSources:")
        by_rank = {passage.rank: passage for passage in result.rag.passages}
        for rank in result.answer.cited:
            passage = by_rank.get(rank)
            if passage is not None:
                print(f"  [{rank}] {passage.cite()}\n      {passage.url}")
    return 0 if result.ok else 1


# `chat` keeps one in-process thread; `ask` is isolated.
CHAT_THREAD = "chat"


def _chat(config, args) -> int:
    """Hold a conversation: one thread, so a follow-up resolves against it."""
    pipeline = TactiStatPipeline(config)
    print("Ask in Vietnamese or English. Ctrl-D or an empty line to finish.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0
        result = _run(pipeline, question, args, thread_id=CHAT_THREAD)
        _report(result, args.trace)
        print()


def _run(pipeline, question: str, args, thread_id: str | None = None):
    """Run one question, streaming node progress when asked."""
    if not getattr(args, "stream", False):
        return pipeline.run(question, thread_id=thread_id)
    result = None
    for node, payload in pipeline.stream(question, thread_id=thread_id):
        if node == "result":
            result = payload
        else:
            print(f"  … {NODE_LABEL.get(node, node)}")
    print()
    return result


def _ask(config, args) -> int:
    """Answer one question on its own; `chat` is where follow-ups belong."""
    result = _run(TactiStatPipeline(config), args.question, args)
    return _report(result, args.trace)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="experiment config layered over configs/default.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override a config value; repeatable",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("build-data", help="download the configured StatsBomb dataset")
    data.add_argument("--force", action="store_true")

    corpus = commands.add_parser("build-corpus", help="build the Wikipedia corpus")
    corpus.add_argument("--force", action="store_true")

    stats_build = commands.add_parser("build-stats", help="build processed player statistics")
    stats_build.add_argument("--check", action="store_true", help="validate without writing")

    index_build = commands.add_parser("build-index", help="chunk and embed the Wikipedia corpus")
    index_build.add_argument("--force", action="store_true")

    search = commands.add_parser("search", help="retrieve passages from the Wikipedia corpus")
    search.add_argument("query")
    search.add_argument("--top-k", type=int)
    # Shorthands still become config overrides, keeping runs reproducible.
    search.add_argument("--mode", choices=MODES, help="overrides rag.retrieval.mode")
    search.add_argument("--rerank", action="store_true", help="overrides rag.rerank.enabled")
    search.add_argument("--full", action="store_true", help="print whole passages")

    ask = commands.add_parser("ask", help="answer a question end to end")
    ask.add_argument("question")
    # Reject strategy typos before building models.
    ask.add_argument("--strategy", choices=STRATEGIES, help="overrides query_translation.strategy")
    ask.add_argument("--trace", action="store_true", help="print each stage's decision")
    ask.add_argument("--stream", action="store_true", help="show stages as they finish")

    chat = commands.add_parser("chat", help="ask follow-up questions in one conversation")
    chat.add_argument("--strategy", choices=STRATEGIES, help="overrides query_translation.strategy")
    chat.add_argument("--trace", action="store_true", help="print each stage's decision")
    chat.add_argument("--stream", action="store_true", help="show stages as they finish")

    stats = commands.add_parser("stats", help="query one structured player statistic")
    stats.add_argument("metric")
    stats.add_argument("players", nargs="*")
    stats.add_argument("--per90", action="store_true")
    stats.add_argument("--top-n", type=int, default=5)
    stats.add_argument("--team")
    stats.add_argument("--match-id", type=int, action="append", dest="match_ids")
    stats.add_argument("--stage")
    stats.add_argument("--date-from")
    stats.add_argument("--date-to")
    return parser


def main() -> int:
    args = _parser().parse_args()
    overrides = list(args.overrides or [])
    if getattr(args, "mode", None):
        overrides.append(f"rag.retrieval.mode={args.mode}")
    if getattr(args, "rerank", False):
        overrides.append("rag.rerank.enabled=true")
    if getattr(args, "strategy", None):
        overrides.append(f"query_translation.strategy={args.strategy}")
    config = load_config(args.config, overrides)

    if args.command == "build-data":
        build_raw_dataset(config, force=args.force)
        return 0
    if args.command == "build-corpus":
        build_corpus(config, force=args.force)
        return 0
    if args.command == "build-stats":
        if args.check:
            return 0 if _check_minutes(config, build_player_matches(config)) else 1
        result = build_stats_tables(config)
        player_matches, player_totals = result["frames"]
        print(f"wrote {len(player_matches)} player-match rows and {len(player_totals)} players")
        return 0 if _check_minutes(config, player_matches) else 1
    if args.command == "build-index":
        result = build_index(config, force=args.force)
        state = "built" if result["rebuilt"] else "already current"
        print(f"index {state}: {result['chunks']} chunks in {result['path']}")
        return 0
    if args.command == "ask":
        return _ask(config, args)
    if args.command == "chat":
        return _chat(config, args)
    if args.command == "search":
        answer = RagRetriever(config).search(args.query, top_k=args.top_k)
        print(answer.to_context(max_chars=10_000 if args.full else 400))
        return 0 if answer.ok else 1

    engine = StatsQueryEngine(config)
    scope = {
        "match_ids": args.match_ids,
        "stage": args.stage,
        "date_from": args.date_from,
        "date_to": args.date_to,
    }
    if not args.players:
        answer = engine.leaderboard(
            args.metric,
            top_n=args.top_n,
            per90=args.per90,
            team=args.team,
            **scope,
        )
    elif len(args.players) == 1:
        answer = engine.player_metric(args.players[0], args.metric, per90=args.per90, **scope)
    else:
        answer = engine.compare(args.players, args.metric, per90=args.per90, **scope)
    print(answer.to_context())
    return 0 if answer.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
