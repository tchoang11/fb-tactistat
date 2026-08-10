"""Aggregate in-play events into per-match, tournament, and per-90 stats."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tactistat.config import Config, ConfigError
from tactistat.data.statsbomb import load_events
from tactistat.stats_tool.minutes import IN_PLAY_PERIODS, compute_minutes


@dataclass(frozen=True)
class Metric:
    """A count or sum derived from event rows."""

    name: str
    description: str
    mask: Callable[[pd.DataFrame], pd.Series]
    # Sum this column when set; otherwise count matching rows.
    value_column: str | None = None


def _is(frame: pd.DataFrame, event_type: str) -> pd.Series:
    return frame["type"] == event_type


METRICS: tuple[Metric, ...] = (
    Metric(
        "goals",
        "Shots resulting in a goal, excluding penalty-shootout kicks.",
        lambda f: _is(f, "Shot") & (f["shot_outcome"] == "Goal"),
    ),
    Metric(
        "xg",
        "Sum of StatsBomb expected-goals values over all shots taken.",
        lambda f: _is(f, "Shot"),
        value_column="shot_statsbomb_xg",
    ),
    Metric("shots", "Shots attempted.", lambda f: _is(f, "Shot")),
    Metric("passes_attempted", "Passes attempted.", lambda f: _is(f, "Pass")),
    Metric(
        "passes_completed",
        "Passes that reached a team-mate. StatsBomb leaves pass_outcome null "
        "on a completed pass and fills it with the failure reason otherwise.",
        lambda f: _is(f, "Pass") & f["pass_outcome"].isna(),
    ),
    Metric(
        "key_passes",
        "Passes that directly created a shot.",
        lambda f: _is(f, "Pass") & (f["pass_shot_assist"] == True),  # noqa: E712
    ),
    Metric(
        "assists",
        "Passes that directly created a goal.",
        lambda f: _is(f, "Pass") & (f["pass_goal_assist"] == True),  # noqa: E712
    ),
    Metric(
        "dribbles_completed",
        "Take-ons completed.",
        lambda f: _is(f, "Dribble") & (f["dribble_outcome"] == "Complete"),
    ),
    Metric(
        "tackles",
        "Tackle duels contested.",
        lambda f: _is(f, "Duel") & (f["duel_type"] == "Tackle"),
    ),
    Metric("interceptions", "Interceptions made.", lambda f: _is(f, "Interception")),
)

METRICS_BY_NAME = {metric.name: metric for metric in METRICS}

# Ratios derived after event metrics are aggregated.
DERIVED_METRICS = {"pass_accuracy": ("passes_completed", "passes_attempted")}


def available_metrics() -> list[str]:
    return [*METRICS_BY_NAME, *DERIVED_METRICS]


def check_configured_metrics(config: Config) -> None:
    """Fail if config names a metric this module cannot produce."""
    unknown = sorted(set(config["stats_tool.metrics"]) - set(available_metrics()))
    if unknown:
        raise ConfigError(
            f"stats_tool.metrics names unimplemented metric(s): {unknown}. "
            f"Available: {sorted(available_metrics())}"
        )
    unknown_per90 = sorted(set(config["stats_tool.per90_metrics"]) - set(available_metrics()))
    if unknown_per90:
        raise ConfigError(f"stats_tool.per90_metrics names unknown metric(s): {unknown_per90}")


# Per player, per match


def build_player_matches(config: Config) -> pd.DataFrame:
    """Build one row per player per match, keeping players with no event."""
    check_configured_metrics(config)

    events = load_events(config)
    in_play = events[events["period"] <= IN_PLAY_PERIODS]

    counts = in_play.groupby(["match_id", "team", "player"], dropna=True)
    table = pd.DataFrame(index=counts.size().index)

    for metric in METRICS:
        masked = in_play[metric.mask(in_play)]
        if metric.value_column:
            series = masked.groupby(["match_id", "team", "player"])[metric.value_column].sum()
        else:
            series = masked.groupby(["match_id", "team", "player"]).size()
        table[metric.name] = series

    table = table.fillna(0.0).reset_index().rename(columns={"player": "player_name"})

    minutes = compute_minutes(config, events).rename(columns={"player": "player_name"})
    merged = table.merge(minutes, on=["match_id", "team", "player_name"], how="outer")

    metric_columns = [m.name for m in METRICS]
    merged[metric_columns] = merged[metric_columns].fillna(0.0)
    merged["minutes"] = merged["minutes"].fillna(0.0)

    integer_columns = [m.name for m in METRICS if m.value_column is None]
    merged[integer_columns] = merged[integer_columns].astype(int)

    merged["pass_accuracy"] = _ratio(merged["passes_completed"], merged["passes_attempted"])
    return merged.sort_values(["match_id", "team", "player_name"]).reset_index(drop=True)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide element-wise; return ``NaN`` when the denominator is zero."""
    return numerator.astype(float) / denominator.replace(0, np.nan).astype(float)


# Per-player tournament totals


def build_player_totals(config: Config, player_matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build tournament totals and per-90 rates for eligible players."""
    matches = build_player_matches(config) if player_matches is None else player_matches

    metric_columns = [m.name for m in METRICS]
    grouped = matches.groupby("player_name")

    totals = grouped[[*metric_columns, "minutes"]].sum()
    totals["team"] = grouped["team"].first()
    totals["matches_played"] = grouped["match_id"].nunique()
    totals["appearances"] = grouped.apply(
        lambda g: int((g["minutes"] > 0).sum()), include_groups=False
    )

    totals["pass_accuracy"] = _ratio(totals["passes_completed"], totals["passes_attempted"])

    threshold = config["dataset.min_minutes_for_per90"]
    eligible = totals["minutes"] >= threshold
    for name in config["stats_tool.per90_metrics"]:
        per90 = totals[name] / totals["minutes"] * 90.0
        totals[f"{name}_per90"] = per90.where(eligible)

    totals["per90_eligible"] = eligible
    return totals.reset_index().sort_values("goals", ascending=False).reset_index(drop=True)
