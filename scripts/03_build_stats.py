#!/usr/bin/env python
"""Build player-stat tables; ``--check`` validates minutes without writing."""

from __future__ import annotations

import argparse

from tactistat.config import load_config
from tactistat.stats_tool.aggregate import build_player_matches
from tactistat.stats_tool.minutes import match_length
from tactistat.stats_tool.query import build_stats_tables


def report_reconciliation(config, player_matches) -> None:
    """Check that player-minutes never exceed 22 x match length."""
    lengths = match_length(config)
    totals = player_matches.groupby("match_id")["minutes"].sum()
    expected = lengths * 22
    difference = (totals - expected).round(2)

    over = difference[difference > 0.01]
    short = difference[difference < -0.01]

    print("\nMinutes reconciliation (22 x nominal length per match)")
    print(f"  exact                {len(difference) - len(over) - len(short)} / {len(difference)}")
    print(f"  short-handed         {len(short)}  (red cards and off-pitch spells)")
    print(f"  OVER-COUNTED         {len(over)}  <- must be 0")
    if len(short):
        print(f"  largest shortfall    {short.min():.2f} min  (match {short.idxmin()})")
    if len(over):
        print("\n  matches over-counting minutes:")
        for match_id, delta in over.items():
            print(f"    {match_id}  +{delta:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="experiment config layered over configs/default.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override a config value, repeatable",
    )
    parser.add_argument(
        "--check", action="store_true", help="run the reconciliation without writing parquet"
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)

    if args.check:
        report_reconciliation(config, build_player_matches(config))
        return 0

    print("Aggregating event data into per-player tables...")
    result = build_stats_tables(config)
    player_matches, player_totals = result["frames"]

    report_reconciliation(config, player_matches)

    threshold = config["dataset.min_minutes_for_per90"]
    eligible = player_totals["per90_eligible"].sum()
    print("\nTables")
    print(f"  player_matches       {len(player_matches):,} rows")
    print(f"  player_totals        {len(player_totals):,} players")
    print(f"  per-90 eligible      {eligible} (>= {threshold} min)")

    print(f"\nTop scorers (of {len(player_totals)} players)")
    for row in player_totals.nlargest(5, "goals").itertuples():
        stat = f"{row.goals:>2} goals  {row.xg:5.2f} xG  {row.minutes:>6.1f} min"
        print(f"  {stat}  {row.player_name}")

    print("\nBest goals per 90 among eligible players")
    for row in player_totals.dropna(subset=["goals_per90"]).nlargest(5, "goals_per90").itertuples():
        stat = f"{row.goals_per90:.2f}/90  ({row.goals:g} in {row.minutes:.0f} min)"
        print(f"  {stat}  {row.player_name}")

    print(f"\n  wrote {result['player_matches']}")
    print(f"  wrote {result['player_totals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
