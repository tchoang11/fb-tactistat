"""Command-line entry point for data builds and structured stats queries."""

from __future__ import annotations

import argparse
from pathlib import Path

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


# Counts that a metric mean cannot express, printed after every run.
EVAL_COUNTERS = (
    ("items", "items"),
    ("errors", "errors"),
    ("judge errors", "judge_errors"),
    ("ungraded rows", "ungraded"),
    ("partial results", "partial_results"),
    ("abstentions", "abstentions"),
    ("items with route repairs", "items_with_route_repairs"),
    ("translation degradations", "translation_degradations"),
    ("items served from cache", "cached_items"),
    ("items touching the cache", "items_with_cache_hits"),
)


# Named here so `--help` does not import the evaluation stack. Duplicating the
# registry is what let two axes ship documented and unrunnable, so a test pins
# this tuple to AXES.
_AXIS_NAMES = (
    "baseline",
    "chunking",
    "embedding",
    "retrieval_mode",
    "rerank",
    "router",
    "translation",
    "translation_uncached",
    "translation_uncached_single",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _evaluate(config, args) -> int:
    """Validate or run the labelled set without conversation state."""
    from tactistat.eval.runner import (
        EvaluationRunner,
        evaluation_settings,
        rejudge_report,
        validate_ground_truth_artifacts,
    )
    from tactistat.eval.schema import load_test_set

    if getattr(args, "rejudge", None):
        if args.no_judge or args.validate_only:
            print("! --rejudge cannot be combined with --no-judge or --validate-only")
            return 2

        def rejudge_progress(position, total, item_id):
            print(f"[{position}/{total}] judge {item_id}", flush=True)

        report = rejudge_report(
            config,
            args.rejudge,
            limit=args.limit,
            item_ids=args.item_ids,
            output_path=args.output,
            progress=rejudge_progress,
        )
        return _print_eval_report(report)

    suite = load_test_set(config.path("eval.test_set"))
    if args.item_ids:
        unknown = sorted(set(args.item_ids) - {item.id for item in suite.items})
        if unknown:
            print(f"! unknown evaluation id(s): {', '.join(unknown)}")
            return 2
    recall_at_k, _ = evaluation_settings(config)
    checked = validate_ground_truth_artifacts(config, suite)
    if args.validate_only:
        counts = {
            label: sum(item.route.label == label for item in suite.items)
            for label in ("STAT", "TACTICAL", "HYBRID")
        }
        print(f"{suite.name}: {len(suite.items)} items, fingerprint {suite.fingerprint}")
        print("  " + ", ".join(f"{label}={count}" for label, count in counts.items()))
        print(
            f"  checked {checked['stats_items']} stats items and "
            f"{checked['retrieval_targets']} retrieval targets"
        )
        print(f"  Recall@k={recall_at_k}")
        return 0

    def progress(position, total, item):
        print(f"[{position}/{total}] {item.id}: {item.question}", flush=True)

    judge_enabled = True if args.judge else (False if args.no_judge else None)
    report = EvaluationRunner(config).run(
        suite,
        limit=args.limit,
        item_ids=args.item_ids,
        judge_enabled=judge_enabled,
        output_path=args.output,
        progress=progress,
    )
    return _print_eval_report(report)


def _print_eval_report(report) -> int:
    """Render the common summary for a pipeline run or a rejudge pass."""
    summary = report["summary"]
    print("\nMetrics:")
    for name, metric in summary["metrics"].items():
        mean = metric["mean"]
        shown = "n/a" if mean is None else f"{mean:.3f}"
        print(f"  {name}: {shown} (n={metric['n']})")

    # A metric table alone cannot distinguish a system that answered badly from
    # one that did not run: an item that raised scores 0 and looks like a wrong
    # answer. These counts are what separates them.
    print("\nRun:")
    for label, key in EVAL_COUNTERS:
        print(f"  {label}: {summary[key]}")
    if not summary["timings_trustworthy"]:
        print(
            f"  ! timings are not a latency measurement: "
            f"{summary['items_with_cache_hits']} item(s) took at least one answer "
            "from the LLM cache — rerun with --set llm.cache_enabled=false"
        )
    ungraded = summary.get("ungraded", 0) if report.get("judge_enabled") else 0
    if ungraded:
        print(
            f"  ! {ungraded} row(s) still have no judge result; the judge_* means above "
            "stand on the rest — rerun --rejudge without --limit to finish them"
        )
    print(f"\nReport: {report['output_path']}")
    # Metrics can legitimately be low; an item that never ran is a broken run.
    return 1 if summary["errors"] or summary["judge_errors"] or ungraded else 0


def _bootstrap(config, args) -> int:
    """Write every interval behind a published comparison, with its inputs."""
    from tactistat.artifacts import project_relative, write_json_atomic
    from tactistat.config import PROJECT_ROOT
    from tactistat.stats_tool.query import bootstrap_artifact

    try:
        report = bootstrap_artifact(config, args.players, args.metric)
    except ValueError as exc:
        print(f"! {exc}")
        return 2
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    report["output_path"] = project_relative(destination)
    write_json_atomic(destination, report)

    settings = report["settings"]
    print(
        f"{report['metric']} per 90 — {settings['n_resamples']} resamples of "
        f"{settings['resampling_unit']}es, seed {settings['seed']}\n"
    )
    for player in report["players"]:
        interval = player["interval"]
        if interval is None:
            print(f"  {player['player_name']}: no rate — {player['ineligible_reason']}")
            continue
        print(
            f"  {player['player_name']}: {interval['point']:.2f} "
            f"[{interval['low']:.2f}, {interval['high']:.2f}] "
            f"over {interval['n_matches']} matches"
        )
    if not report["pairs"]:
        print("\n  no pair to compare")
        return 0
    print("\n  paired vs independent difference:")
    for pair in report["pairs"]:
        paired, independent = pair["paired"], pair["independent"]
        shared = len(pair["shared_match_ids"])
        print(
            f"  {pair['left']} − {pair['right']}: {paired['point']:+.2f}\n"
            f"      paired      [{paired['low']:+.2f}, {paired['high']:+.2f}] "
            f"width {paired['high'] - paired['low']:.2f} "
            f"({paired['n_matches']} fixtures, {shared} shared)\n"
            f"      independent [{independent['low']:+.2f}, {independent['high']:+.2f}] "
            f"width {independent['high'] - independent['low']:.2f} "
            f"({independent['n_matches']} player-matches)"
        )
    print(f"\nReport: {report['output_path']}")
    return 0


def _ablate(config, args) -> int:
    """Sweep one axis and print the arms side by side."""
    from tactistat.eval.ablate import AXES, comparison_table, run_axis

    axis = AXES[args.axis]

    def progress(position, total, arm):
        print(f"[{position}/{total}] {axis.name}={arm.name} {arm.overrides or ''}")

    report = run_axis(config, axis, output_path=args.output, progress=progress)
    print("\n" + comparison_table(report, metrics=args.metrics))
    # An arm where every item errored has the key and no timings under it.
    latencies = [
        (arm["name"], arm["retrieval_ms"])
        for arm in report["arms"]
        if (arm.get("retrieval_ms") or {}).get("p50") is not None
    ]
    if latencies:
        print("\nRetrieval latency (local compute; the LLM cache does not serve it):")
        for name, timing in latencies:
            print(f"  {name}: p50 {timing['p50']:.0f} ms, p95 {timing['p95']:.0f} ms")
    print(f"\nReport: {report['output_path']}")

    # An arm that never ran is not a low score, and neither is an item that
    # raised. Both have to reach the exit code or a broken sweep reads as a
    # finished one.
    lost_arms = report.get("arm_errors") or 0
    for arm in report["arms"]:
        if arm.get("error"):
            print(f"! arm {arm['name']} failed: {arm['error']}")
    broken = sum(arm.get("errors") or 0 for arm in report["arms"])
    ungraded = sum(arm.get("judge_errors") or 0 for arm in report["arms"])
    if broken:
        print(f"! {broken} item(s) errored across the sweep")
    if ungraded:
        print(
            f"! {ungraded} judge call(s) failed; the judge_* means above stand on "
            "whatever was graded — check the n before quoting them"
        )
    return 1 if lost_arms or broken or ungraded else 0


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

    evaluate = commands.add_parser("evaluate", help="run the labelled evaluation set")
    evaluate.add_argument("--limit", type=_positive_int, help="run only the first N selected items")
    evaluate.add_argument(
        "--id", dest="item_ids", action="append", help="run one item id; repeatable"
    )
    judging = evaluate.add_mutually_exclusive_group()
    judging.add_argument("--no-judge", action="store_true", help="skip LLM answer grading")
    judging.add_argument(
        "--judge", action="store_true", help="force LLM answer grading on for this run"
    )
    evaluate.add_argument("--output", help="raw report path or filename under eval/results/raw")
    mode = evaluate.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate schema and artifact labels without model calls",
    )
    mode.add_argument(
        "--rejudge",
        metavar="REPORT",
        help="retry missing/failed judge rows without rerunning the pipeline",
    )

    ablate = commands.add_parser("ablate", help="sweep one configuration axis")
    ablate.add_argument("axis", choices=sorted(_AXIS_NAMES))
    ablate.add_argument(
        "--output", help="report path; a bare filename lands under eval/results/ablations"
    )
    ablate.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        help="metric column to show; repeatable, first one orders the table",
    )

    bootstrap = commands.add_parser(
        "bootstrap", help="resampled per-90 intervals for a set of players, and every pair"
    )
    bootstrap.add_argument("metric")
    bootstrap.add_argument("players", nargs="+")
    bootstrap.add_argument(
        "--output",
        default="eval/results/bootstrap.json",
        help="artifact path, relative to the repository root",
    )

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
    if args.command == "evaluate":
        return _evaluate(config, args)
    if args.command == "ablate":
        return _ablate(config, args)
    if args.command == "bootstrap":
        return _bootstrap(config, args)
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
