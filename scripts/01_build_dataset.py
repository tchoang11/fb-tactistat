#!/usr/bin/env python
"""Download the configured competition from StatsBomb open data.

Run once before anything else:

    python scripts/01_build_dataset.py
    python scripts/01_build_dataset.py --force        # refetch even if cached
    python scripts/01_build_dataset.py --set dataset.competition_id=55 \
                                       --set dataset.season_id=282   # Euro 2024

Takes about a minute on a normal connection and writes ~50 MB of parquet into
``data/raw/``. That directory is gitignored: it is large and this script
regenerates it exactly.
"""

from __future__ import annotations

import argparse

from tactistat.config import load_config
from tactistat.data.statsbomb import build_raw_dataset, load_matches


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
    parser.add_argument("--force", action="store_true", help="refetch even if the cache exists")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    build_raw_dataset(config, force=args.force)

    # Sanity summary. A silent success on a dataset build is not reassuring --
    # print enough that an obviously wrong competition or a truncated download
    # is visible immediately rather than three steps later.
    matches = load_matches(config)
    print()
    print(f"{config['dataset.competition_name']} {config['dataset.season_name']}")
    print(f"  matches      {len(matches)}")
    print(f"  date range   {matches['match_date'].min()} .. {matches['match_date'].max()}")
    print(f"  teams        {matches['home_team'].nunique()} distinct home sides")
    print(f"  stages       {', '.join(matches['competition_stage'].unique())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
