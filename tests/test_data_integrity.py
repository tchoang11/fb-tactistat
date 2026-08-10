"""Check raw StatsBomb data against official tournament facts."""

from __future__ import annotations

import pytest

from tactistat.config import load_config
from tactistat.data.statsbomb import (
    DataError,
    load_events,
    load_lineups,
    load_matches,
)

# External FIFA reference values.
OFFICIAL_TOTAL_GOALS = 172
OFFICIAL_TOP_SCORERS = {
    "Kylian Mbappé Lottin": 8,  # Golden Boot
    "Lionel Andrés Messi Cuccittini": 7,
    "Olivier Giroud": 4,
    "Julián Álvarez": 4,
}
N_MATCHES = 64
N_TEAMS = 32

# Period 5 is the shootout and does not count toward goal totals.
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


# Dataset shape


def test_match_count(matches):
    assert len(matches) == N_MATCHES


def test_all_teams_present(matches):
    teams = set(matches["home_team"]) | set(matches["away_team"])
    assert len(teams) == N_TEAMS


def test_every_match_has_two_starting_elevens(events):
    """Each match has one Starting XI event per team."""
    counts = events[events["type"] == "Starting XI"].groupby("match_id").size()
    assert len(counts) == N_MATCHES
    assert counts.eq(2).all()


def test_extra_time_is_present(events):
    """Knockout matches ran past 90 minutes; the loader must not truncate them."""
    assert events["minute"].max() >= 120


# Goal totals


def test_tournament_goal_total_matches_official_record(events, in_play_goals):
    """169 open-play + set-piece goals plus 3 own goals equals the official 172."""
    own_goals = (events["type"] == "Own Goal Against").sum()
    assert len(in_play_goals) + own_goals == OFFICIAL_TOTAL_GOALS


def test_shootout_goals_are_excluded(events, in_play_goals):
    """Confirm shootout goals exist but stay outside in-play totals."""
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
    """Reconcile each team's event goals with the match scoreboard."""
    scored = in_play_goals.groupby(["match_id", "team"]).size()
    own_goals = events[events["type"] == "Own Goal Against"].groupby(["match_id", "team"]).size()

    mismatches = []
    for _, match in matches.iterrows():
        for side, opponent in (("home", "away"), ("away", "home")):
            team = match[f"{side}_team"]
            key = (match["match_id"], team)
            # Credit an own goal to the opponent's score.
            conceded = own_goals.get((match["match_id"], match[f"{opponent}_team"]), 0)
            total = scored.get(key, 0) + conceded
            if total != match[f"{side}_score"]:
                mismatches.append(
                    f"{match['home_team']} v {match['away_team']}: {team} "
                    f"events={total} scoreboard={match[f'{side}_score']}"
                )
    assert not mismatches, "goal reconciliation failed:\n" + "\n".join(mismatches)


# Metric coverage


def test_every_shot_has_an_xg_value(events):
    """Every shot must have xG to avoid biased totals."""
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
