"""Golden tests pinning the raw dataset against externally known facts.

Every number downstream -- per-90 rates, the numeric-accuracy metric, the
bootstrap confidence intervals -- inherits whatever the event table says. A
loading bug does not announce itself: it produces a table that is the right
shape, full of plausible numbers, and quietly wrong. The evaluation harness
cannot catch that, because it grades the system against ground truth derived
from this same table.

So the dataset is checked against facts sourced from outside it: the official
World Cup 2022 goal total, the official top scorers, and the scoreboard
recorded on each match row.

The shootout case is the one that actually bit during development. StatsBomb
records penalty-shootout kicks as ordinary ``Shot`` events in ``period == 5``.
Counting them inflates the tournament total from 172 to 195 and turns several
players into phantom scorers -- while every table still looks entirely normal.
"""

from __future__ import annotations

import pytest

from tactistat.config import load_config
from tactistat.data.statsbomb import (
    DataError,
    load_events,
    load_lineups,
    load_matches,
)

# Sourced from FIFA's official tournament report, not from the event data.
OFFICIAL_TOTAL_GOALS = 172
OFFICIAL_TOP_SCORERS = {
    "Kylian Mbappé Lottin": 8,  # Golden Boot
    "Lionel Andrés Messi Cuccittini": 7,
    "Olivier Giroud": 4,
    "Julián Álvarez": 4,
}
N_MATCHES = 64
N_TEAMS = 32

# Regulation and extra time occupy periods 1-4. Period 5 is the shootout, which
# does not count towards a player's goal tally or the tournament total.
IN_PLAY_PERIODS = 4


@pytest.fixture(scope="module")
def config():
    return load_config(load_env=False)


@pytest.fixture(scope="module")
def events(config):
    try:
        return load_events(config)
    except DataError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def matches(config):
    try:
        return load_matches(config)
    except DataError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def in_play_goals(events):
    """Goals scored in regulation or extra time, excluding shootout kicks."""
    return events[
        (events["type"] == "Shot")
        & (events["shot_outcome"] == "Goal")
        & (events["period"] <= IN_PLAY_PERIODS)
    ]


# --------------------------------------------------------------------------- #
# Shape                                                                        #
# --------------------------------------------------------------------------- #


def test_match_count(matches):
    assert len(matches) == N_MATCHES


def test_all_teams_present(matches):
    teams = set(matches["home_team"]) | set(matches["away_team"])
    assert len(teams) == N_TEAMS


def test_every_match_has_two_starting_elevens(events):
    """128 Starting XI events, one per team per match.

    A missing one means a match failed to download and was concatenated in
    silently -- which would skew every per-90 rate for that squad.
    """
    assert (events["type"] == "Starting XI").sum() == N_MATCHES * 2


def test_extra_time_is_present(events):
    """Knockout matches ran past 90 minutes; the loader must not truncate them."""
    assert events["minute"].max() >= 120


# --------------------------------------------------------------------------- #
# Goals: the numbers everything else is graded against                         #
# --------------------------------------------------------------------------- #


def test_tournament_goal_total_matches_official_record(events, in_play_goals):
    """169 open-play + set-piece goals plus 3 own goals equals the official 172."""
    own_goals = (events["type"] == "Own Goal Against").sum()
    assert len(in_play_goals) + own_goals == OFFICIAL_TOTAL_GOALS


def test_shootout_goals_are_excluded(events, in_play_goals):
    """The shootout kicks exist in the data and are deliberately not counted.

    Asserting the gap is non-zero keeps this test honest: if StatsBomb ever
    stopped emitting period-5 shots, an equality-only check would still pass
    while quietly testing nothing.
    """
    all_shot_goals = ((events["type"] == "Shot") & (events["shot_outcome"] == "Goal")).sum()
    assert all_shot_goals > len(in_play_goals), "no period-5 shots found; filter may be dead"


def test_top_scorers_match_official_record(in_play_goals):
    tally = in_play_goals.groupby("player").size()
    for player, expected in OFFICIAL_TOP_SCORERS.items():
        assert player in tally.index, f"{player} absent from the event data"
        assert tally[player] == expected, (
            f"{player}: event data says {tally[player]}, official record says {expected}"
        )


def test_goals_reconcile_with_every_scoreboard(matches, events, in_play_goals):
    """Per-team goal counts must equal the score on the match row, 128/128 times.

    This is the strongest available check: it cross-validates the event stream
    against an independent field in the same dataset, for every team in every
    match, and it is what catches an own-goal attribution error.
    """
    scored = in_play_goals.groupby(["match_id", "team"]).size()
    own_goals = events[events["type"] == "Own Goal Against"].groupby(["match_id", "team"]).size()

    mismatches = []
    for _, match in matches.iterrows():
        for side, opponent in (("home", "away"), ("away", "home")):
            team = match[f"{side}_team"]
            key = (match["match_id"], team)
            # StatsBomb charges an own goal to the team that conceded it, so it
            # credits the opponent's score.
            conceded = own_goals.get((match["match_id"], match[f"{opponent}_team"]), 0)
            total = scored.get(key, 0) + conceded
            if total != match[f"{side}_score"]:
                mismatches.append(
                    f"{match['home_team']} v {match['away_team']}: {team} "
                    f"events={total} scoreboard={match[f'{side}_score']}"
                )
    assert not mismatches, "goal reconciliation failed:\n" + "\n".join(mismatches)


# --------------------------------------------------------------------------- #
# Metric coverage                                                              #
# --------------------------------------------------------------------------- #


def test_every_shot_has_an_xg_value(events):
    """xG underpins the STAT half of the system; partial coverage would bias it.

    A shot without xG silently contributes 0 to a player's total, making
    low-volume shooters look more wasteful than they were.
    """
    shots = events[events["type"] == "Shot"]
    assert len(shots) > 0
    assert shots["shot_statsbomb_xg"].notna().all()


def test_substitutions_are_recorded(events):
    """Minutes played -- and therefore every per-90 rate -- depends on these."""
    assert (events["type"] == "Substitution").sum() > 0


def test_lineups_cover_every_match(config, matches):
    try:
        lineups = load_lineups(config)
    except DataError as exc:
        pytest.skip(str(exc))
    assert lineups["match_id"].nunique() == len(matches)
