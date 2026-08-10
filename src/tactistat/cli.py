"""Command-line entry point for data builds and structured stats queries."""

from __future__ import annotations

import argparse

from tactistat.config import load_config
from tactistat.data.statsbomb import build_raw_dataset
from tactistat.data.wikipedia import build_corpus
from tactistat.stats_tool.aggregate import build_player_matches
from tactistat.stats_tool.minutes import match_length
from tactistat.stats_tool.query import StatsQueryEngine, build_stats_tables


def _check_minutes(config, player_matches) -> bool:
    """Return whether aggregate player-minutes stay within match capacity."""
    difference = player_matches.groupby("match_id")["minutes"].sum() - match_length(config) * 22
    over = difference[difference > 0.01]
    print(f"matches checked: {len(difference)}; over-counted: {len(over)}")
    return over.empty


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
    config = load_config(args.config, args.overrides)

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
