"""STAT pipeline: events -> minutes -> aggregates -> query results."""

from tactistat.stats_tool.aggregate import (
    METRICS,
    available_metrics,
    build_player_matches,
    build_player_totals,
)
from tactistat.stats_tool.minutes import compute_minutes, match_length, nominal_clock
from tactistat.stats_tool.query import (
    MatchRef,
    PlayerResolution,
    PlayerRow,
    StatsAnswer,
    StatsQueryEngine,
    build_stats_tables,
)

__all__ = [
    "METRICS",
    "MatchRef",
    "PlayerResolution",
    "PlayerRow",
    "StatsAnswer",
    "StatsQueryEngine",
    "available_metrics",
    "build_player_matches",
    "build_player_totals",
    "build_stats_tables",
    "compute_minutes",
    "match_length",
    "nominal_clock",
]
