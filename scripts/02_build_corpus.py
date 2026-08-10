#!/usr/bin/env python
"""Build the competition-scoped Wikipedia corpus from the raw dataset."""

from __future__ import annotations

import argparse
from collections import Counter

from tactistat.config import load_config
from tactistat.data.wikipedia import build_corpus, build_page_specs, select_players


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
    parser.add_argument("--force", action="store_true", help="refetch even if the corpus exists")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the page list and player selection, fetch nothing",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)

    if args.dry_run:
        players = select_players(config)
        specs = build_page_specs(config)
        counts = Counter(spec.entity_type for spec in specs)

        print(
            f"Player selection: apps >= {config['wikipedia.min_appearances']}"
            f"{' or scored' if config['wikipedia.include_all_scorers'] else ''}"
            f"  ->  {len(players)} players"
        )
        print("\n  top 10 by goals:")
        for _, row in players.head(10).iterrows():
            arrow = (
                "" if row["search_name"] == row["player_name"] else f"  ->  {row['search_name']}"
            )
            print(f"    {row['goals']}g {row['apps']}app  {row['player_name']}{arrow}")

        print(f"\nPages to fetch: {sum(counts.values())}")
        for entity_type, count in sorted(counts.items()):
            print(f"  {entity_type:12} {count}")
        return 0

    build_corpus(config, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
