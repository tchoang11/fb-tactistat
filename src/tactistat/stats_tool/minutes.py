"""Reconstruct 90/120-minute playing time from presence-changing events."""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any

import pandas as pd

from tactistat.config import Config
from tactistat.data.statsbomb import load_events

# Period 5 is the shootout and is not playing time.
NOMINAL_PERIOD_END = {1: 45.0, 2: 90.0, 3: 105.0, 4: 120.0}

IN_PLAY_PERIODS = 4

# Both values dismiss a player.
RED_CARDS = {"Red Card", "Second Yellow"}

# Events that can change whether a player is on the pitch.
PRESENCE_EVENTS = [
    "Starting XI",
    "Substitution",
    "Player Off",
    "Player On",
    "Foul Committed",
    "Bad Behaviour",
]


def nominal_clock(period: Any, minute: Any, second: Any) -> float:
    """Convert a timestamp to nominal minutes, excluding stoppage time."""
    return min(float(minute) + float(second) / 60.0, NOMINAL_PERIOD_END[int(period)])


def match_length(config: Config, events: pd.DataFrame | None = None) -> pd.Series:
    """Return 90 or 120 minutes from the highest period played."""
    events = load_events(config) if events is None else events
    in_play = events[events["period"] <= IN_PLAY_PERIODS]
    return in_play.groupby("match_id")["period"].max().map(NOMINAL_PERIOD_END)


def _sorted_presence_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return presence events in order; ``index`` breaks timestamp ties."""
    frame = events[
        (events["period"] <= IN_PLAY_PERIODS) & events["type"].isin(PRESENCE_EVENTS)
    ].copy()
    keys = ["match_id", "period", "minute", "second"]
    if "index" in frame.columns:
        keys.append("index")
    return frame.sort_values(keys, kind="stable")


def _red_card(row: Any) -> str | None:
    """The card on a foul or misconduct event, if there is one."""
    card = row.foul_committed_card if row.type == "Foul Committed" else row.bad_behaviour_card
    return card if isinstance(card, str) else None


Spells = dict[str, list[list[float | None]]]


def _close_spell(spells: Spells, player: str, at: float) -> None:
    """Close the player's current on-pitch interval, if open."""
    if spells.get(player) and spells[player][-1][1] is None:
        spells[player][-1][1] = at


def compute_minutes(config: Config, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return ``match_id``, ``team``, ``player``, and minutes played."""
    events = load_events(config) if events is None else events
    lengths = match_length(config, events)

    records: list[dict[str, Any]] = []

    for match_id, group in _sorted_presence_events(events).groupby("match_id", sort=False):
        final_whistle = lengths[match_id]
        # A player may have multiple [start, end] intervals.
        spells: Spells = defaultdict(list)
        team_of: dict[str, str] = {}

        for row in group.itertuples():
            at = nominal_clock(row.period, row.minute, row.second)

            if row.type == "Starting XI":
                # Parquet stores this nested value as a repr string.
                lineup = ast.literal_eval(row.tactics)["lineup"]
                for entry in lineup:
                    name = entry["player"]["name"]
                    team_of[name] = row.team
                    spells[name].append([0.0, None])

            elif row.type == "Substitution":
                _close_spell(spells, row.player, at)
                replacement = row.substitution_replacement
                team_of[replacement] = row.team
                spells[replacement].append([at, None])

            elif row.type == "Player Off":
                _close_spell(spells, row.player, at)

            elif row.type == "Player On":
                team_of.setdefault(row.player, row.team)
                spells[row.player].append([at, None])

            elif _red_card(row) in RED_CARDS:
                _close_spell(spells, row.player, at)

        for player, intervals in spells.items():
            total = sum(
                (end if end is not None else final_whistle) - start for start, end in intervals
            )
            records.append(
                {
                    "match_id": match_id,
                    "team": team_of.get(player),
                    "player": player,
                    "minutes": round(total, 4),
                }
            )

    return pd.DataFrame.from_records(
        records, columns=["match_id", "team", "player", "minutes"]
    ).sort_values(["match_id", "team", "minutes"], ascending=[True, True, False])
