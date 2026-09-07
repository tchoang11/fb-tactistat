"""Resampled confidence intervals for per-90 rates."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from tactistat.config import Config, load_config
from tactistat.stats_tool.bootstrap import (
    bootstrap_settings,
    difference_interval,
    per90_interval,
)


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config(load_env=False)


def matches(values, minutes=90.0, ids=None):
    """One row per match, with the metric under a `goals` column.

    `ids` names the fixtures, so two frames can be given an overlapping
    schedule the way two players at the same tournament have one.
    """
    values = list(values)
    return pd.DataFrame(
        {
            "goals": values,
            "minutes": [minutes] * len(values),
            "match_id": list(ids) if ids is not None else list(range(1, len(values) + 1)),
        }
    )


def test_the_point_estimate_is_the_ordinary_per_90_rate():
    """The interval must sit around the number the stats tool already reports."""
    interval = per90_interval(matches([1, 1, 0, 2]), "goals", n_resamples=2000)
    assert math.isclose(interval.point, 4 * 90 / 360)
    assert interval.low <= interval.point <= interval.high
    assert interval.n_matches == 4


def test_the_interval_narrows_as_matches_accumulate():
    """Sampling error is what the interval measures, so more games must shrink it."""
    few = per90_interval(matches([1, 0, 2]), "goals", n_resamples=4000, seed=7)
    many = per90_interval(matches([1, 0, 2] * 12), "goals", n_resamples=4000, seed=7)
    assert many.width < few.width


def test_a_player_who_never_scored_has_a_degenerate_interval():
    interval = per90_interval(matches([0, 0, 0]), "goals", n_resamples=1000)
    assert (interval.point, interval.low, interval.high) == (0.0, 0.0, 0.0)
    assert not interval.excludes_zero()


def test_resampling_is_reproducible_from_the_seed():
    """A reported interval has to be the same interval tomorrow."""
    args = {"n_resamples": 3000, "seed": 99}
    assert per90_interval(matches([1, 0, 2, 1]), "goals", **args) == per90_interval(
        matches([1, 0, 2, 1]), "goals", **args
    )


def test_the_seed_reaches_the_generator():
    """Asserted on the draws, not the percentiles.

    Resampling small integer counts gives a coarse distribution whose 2.5% and
    97.5% points often land on the same value for two seeds, so comparing
    intervals would test the granularity of the statistic rather than whether
    the seed was used at all.
    """
    import numpy as np

    from tactistat.stats_tool.bootstrap import _draw

    totals, minutes = np.array([1.0, 0.0, 2.0, 1.0]), np.array([90.0] * 4)
    first = _draw(totals, minutes, 500, seed=99)
    assert np.array_equal(first, _draw(totals, minutes, 500, seed=99))
    assert not np.array_equal(first, _draw(totals, minutes, 500, seed=100))


def test_minutes_weight_the_rate_not_the_match_count():
    """A substitute's goal in ten minutes is a high rate on thin evidence."""
    interval = per90_interval(
        pd.DataFrame({"goals": [1, 0], "minutes": [10.0, 90.0]}), "goals", n_resamples=2000
    )
    assert math.isclose(interval.point, 90 / 100)


def test_a_difference_interval_spanning_zero_refuses_to_order_the_players():
    """The whole point: a point estimate alone never admits it cannot decide."""
    # Two players from different sides who met once, as the leading scorers did.
    left = matches([2, 1, 1, 0, 1], ids=[1, 2, 3, 4, 50])
    right = matches([1, 1, 1, 0, 1], ids=[5, 6, 7, 8, 50])
    gap = difference_interval(left, right, "goals", n_resamples=4000, seed=3)
    assert gap.point > 0  # left scored more
    assert gap.low < 0 < gap.high and not gap.excludes_zero()


def test_a_large_separation_does_exclude_zero():
    prolific = matches([3, 2, 3, 2, 3])
    barren = matches([0, 0, 0, 0, 0])
    gap = difference_interval(prolific, barren, "goals", n_resamples=4000, seed=3)
    assert gap.excludes_zero() and gap.low > 0


def test_a_player_against_themselves_has_no_gap_at_all():
    """The sharpest test of pairing: every fixture is shared, so every draw cancels."""
    same = matches([1, 0, 2, 1])
    gap = difference_interval(same, same, "goals", n_resamples=4000, seed=11)
    assert math.isclose(gap.point, 0.0, abs_tol=1e-9)
    assert gap.width == 0.0
    assert gap.n_matches == 4 and gap.n_shared == 4


def test_players_who_never_met_are_drawn_independently():
    """With no shared fixture there is nothing to pair, and the gap must vary."""
    left = matches([1, 0, 2, 1], ids=[1, 2, 3, 4])
    right = matches([1, 0, 2, 1], ids=[5, 6, 7, 8])
    gap = difference_interval(left, right, "goals", n_resamples=4000, seed=11)
    assert math.isclose(gap.point, 0.0, abs_tol=1e-9)
    assert gap.width > 0
    assert gap.n_shared == 0


def test_a_shared_fixture_is_counted_once_and_named():
    """Two seven-match players who met once played thirteen fixtures, not fourteen."""
    left = matches([1, 1, 0, 1, 0, 2, 1], ids=[1, 2, 3, 4, 5, 6, 100])
    right = matches([0, 1, 1, 0, 2, 1, 3], ids=[7, 8, 9, 10, 11, 12, 100])
    gap = difference_interval(left, right, "goals", n_resamples=2000, seed=5)
    assert gap.n_matches == 13
    assert gap.n_shared == 1


def test_pairing_a_shared_fixture_narrows_the_interval():
    """The whole reason to pair: a common match moves both rates together."""
    overlap = [1, 1, 0, 1, 2]
    left = matches(overlap, ids=[1, 2, 3, 4, 99])
    shared = matches([0, 1, 1, 0, 2], ids=[5, 6, 7, 8, 99])
    apart = matches([0, 1, 1, 0, 2], ids=[5, 6, 7, 8, 10])
    paired = difference_interval(left, shared, "goals", n_resamples=8000, seed=7)
    unpaired = difference_interval(left, apart, "goals", n_resamples=8000, seed=7)
    assert paired.n_shared == 1 and unpaired.n_shared == 0
    assert paired.width < unpaired.width


def test_pairing_needs_fixture_ids_rather_than_row_order():
    """Without match ids the two schedules cannot be aligned, so refuse."""
    left = matches([1, 0, 2]).drop(columns=["match_id"])
    with pytest.raises(ValueError, match="match_id"):
        difference_interval(left, matches([1, 0, 2]), "goals", n_resamples=200)


def test_an_empty_frame_yields_no_interval_rather_than_a_crash():
    empty = matches([])
    assert math.isnan(per90_interval(empty, "goals").point)
    assert math.isnan(difference_interval(empty, matches([1]), "goals").point)


def test_an_unknown_metric_is_refused_by_name():
    with pytest.raises(ValueError, match="nutmegs"):
        per90_interval(matches([1]), "nutmegs")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("eval.bootstrap.n_resamples", 10),
        ("eval.bootstrap.n_resamples", True),
        ("eval.bootstrap.confidence_level", 1.0),
        ("eval.bootstrap.confidence_level", 0),
        ("eval.bootstrap.seed", "42"),
    ],
)
def test_bootstrap_settings_reject_unusable_values(config, key, value):
    broken = Config(config.to_dict())
    broken.set(key, value)
    with pytest.raises(ValueError):
        bootstrap_settings(broken)


def test_the_shipped_settings_are_valid(config):
    assert bootstrap_settings(config) == (10_000, 0.95, 42)


def test_a_per90_comparison_carries_its_intervals(config):
    """The engine attaches them, so synthesis can quote them as evidence."""
    from tactistat.stats_tool.query import StatsQueryEngine, claimable_values

    engine = StatsQueryEngine(config)
    answer = engine.compare(["Lionel Messi", "Kylian Mbappe"], "goals", per90=True)
    assert answer.ok and len(answer.intervals) == 2 and answer.difference is not None

    context = answer.to_context()
    assert "95% CI" in context and "cannot separate them" in context
    # Every bound the context prints must be quotable without tripping the guard.
    claimable = claimable_values(answer)["goals"]
    for interval in answer.intervals.values():
        assert round(interval.low, 2) in claimable and round(interval.high, 2) in claimable


def test_the_rendered_confidence_follows_the_configured_level(config):
    """A hard-coded 95% label would mislabel every non-default configuration."""
    from tactistat.stats_tool.query import StatsQueryEngine

    eighty = Config(config.to_dict())
    eighty.set("eval.bootstrap.confidence_level", 0.8)
    answer = StatsQueryEngine(eighty).compare(
        ["Lionel Messi", "Kylian Mbappe"], "goals", per90=True
    )
    context = answer.to_context()
    assert "80% CI" in context and "95% CI" not in context
    assert all(interval.confidence == 0.8 for interval in answer.intervals.values())
    assert answer.difference.confidence == 0.8


def test_a_shared_fixture_is_reported_as_shared(config):
    """Messi and Mbappe met in the final: thirteen fixtures between them, not fourteen."""
    from tactistat.stats_tool.query import StatsQueryEngine

    answer = StatsQueryEngine(config).compare(
        ["Lionel Messi", "Kylian Mbappe"], "goals", per90=True
    )
    gap = answer.difference
    assert gap.n_shared == 1
    assert gap.n_matches == sum(row.appearances for row in answer.rows) - gap.n_shared
    assert "13 matches, 1 of them shared" in answer.to_context()


def test_a_raw_total_comparison_gets_no_interval(config):
    """An interval around a season total would describe a tournament not played."""
    from tactistat.stats_tool.query import StatsQueryEngine

    engine = StatsQueryEngine(config)
    answer = engine.compare(["Lionel Messi", "Kylian Mbappe"], "goals")
    assert answer.ok and answer.intervals == {} and answer.difference is None


def test_the_confidence_label_does_not_round_into_a_different_interval():
    """A 95.5% interval printed as "96% CI" names an interval that was not computed."""
    from tactistat.stats_tool.bootstrap import Interval

    def label(confidence):
        return Interval(0.0, 0.0, 0.0, confidence, 1, 100).label

    assert label(0.95) == "95%"
    assert label(0.8) == "80%"
    assert label(0.955) == "95.5%"
    assert label(0.9999) == "99.99%"


def test_the_independent_estimator_is_still_reachable():
    """A claim that pairing changed the answer must be checkable against the old one."""
    left = matches([2, 1, 1, 0, 1], ids=[1, 2, 3, 4, 50])
    right = matches([1, 1, 1, 0, 1], ids=[5, 6, 7, 8, 50])
    paired = difference_interval(left, right, "goals", n_resamples=4000, seed=3)
    independent = difference_interval(left, right, "goals", n_resamples=4000, seed=3, paired=False)

    assert math.isclose(paired.point, independent.point)
    # Unpaired draws the shared fixture twice, which is what that estimand says.
    assert paired.n_matches == 9 and paired.n_shared == 1
    assert independent.n_matches == 10 and independent.n_shared == 0


def test_pairing_widens_when_the_two_players_trade_off():
    """Pairing keeps the covariance, and a negative one widens rather than narrows."""
    # Two forwards in the same XI who never score in the same match.
    fixtures = [1, 2, 3, 4, 5, 6]
    left = matches([2, 0, 2, 0, 2, 0], ids=fixtures)
    right = matches([0, 2, 0, 2, 0, 2], ids=fixtures)
    paired = difference_interval(left, right, "goals", n_resamples=8000, seed=5)
    independent = difference_interval(left, right, "goals", n_resamples=8000, seed=5, paired=False)

    assert paired.n_shared == 6
    assert paired.width > independent.width
    # And a fully shared schedule is not a zero-width interval unless the paired
    # differences themselves do not vary.
    assert paired.width > 0


def test_a_fully_shared_schedule_is_only_degenerate_when_the_gap_is_constant():
    fixtures = [1, 2, 3, 4]
    identical = matches([1, 0, 2, 1], ids=fixtures)
    assert difference_interval(identical, identical, "goals", n_resamples=2000).width == 0.0
    # Same fixtures, a constant one-goal gap in every one of them: the paired
    # difference is the same number in every resample.
    ahead = matches([2, 1, 3, 2], ids=fixtures)
    constant = difference_interval(ahead, identical, "goals", n_resamples=2000, seed=4)
    assert constant.n_shared == 4 and constant.width == 0.0
    assert math.isclose(constant.point, 1.0)


def test_the_artifact_carries_everything_needed_to_recompute_it(config, tmp_path):
    """An interval published without its seed and fixtures cannot be checked."""
    from tactistat.stats_tool.query import bootstrap_artifact

    report = bootstrap_artifact(config, ["Lionel Messi", "Kylian Mbappe", "Julian Alvarez"])

    assert report["settings"] == {
        "n_resamples": 10_000,
        "confidence_level": 0.95,
        "seed": 42,
        "resampling_unit": "match",
    }
    assert report["dataset_fingerprint"] and report["metric"] == "goals"
    assert len(report["players"]) == 3 and len(report["pairs"]) == 3

    messi = next(p for p in report["players"] if p["query"] == "Lionel Messi")
    assert len(messi["match_ids"]) == messi["interval"]["n_matches"] == 7
    assert messi["interval"]["low"] < messi["per90"] < messi["interval"]["high"]

    # Both estimators, so the comparison between them is reproducible.
    pair = next(p for p in report["pairs"] if "Mbapp" in p["right"] or "Mbapp" in p["left"])
    assert pair["paired"]["point"] == pair["independent"]["point"]
    assert pair["shared_match_ids"]
    assert pair["paired"]["n_shared"] == len(pair["shared_match_ids"])


def test_the_bootstrap_command_writes_its_artifact(config, tmp_path, capsys):
    from argparse import Namespace

    from tactistat.cli import _bootstrap

    output = tmp_path / "bootstrap.json"
    args = Namespace(metric="goals", players=["Lionel Messi", "Kylian Mbappe"], output=str(output))
    assert _bootstrap(config, args) == 0

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["artifact"] == "bootstrap_intervals"
    assert saved["pairs"][0]["shared_match_ids"] == [3869685]  # the final
    printed = capsys.readouterr().out
    assert "paired" in printed and "independent" in printed


def test_an_unresolved_player_refuses_rather_than_silently_dropping(config):
    from tactistat.stats_tool.query import bootstrap_artifact

    with pytest.raises(ValueError, match="Nobody At All"):
        bootstrap_artifact(config, ["Lionel Messi", "Nobody At All"])


def test_the_artifact_refuses_a_rate_the_engine_would_refuse(config):
    """An artifact holding a number the system declines to state is not an audit trail."""
    from tactistat.stats_tool.query import bootstrap_artifact

    report = bootstrap_artifact(config, ["Ao Tanaka", "Lionel Messi"])
    thin = next(p for p in report["players"] if p["query"] == "Ao Tanaka")

    assert thin["minutes"] < report["min_minutes_for_per90"]
    assert thin["eligible"] is False and thin["interval"] is None
    assert "threshold" in thin["ineligible_reason"]
    # And no pair is built from a player who has no rate.
    assert report["pairs"] == []


def test_two_names_for_one_player_are_refused(config):
    """Otherwise the artifact publishes a player compared against themselves."""
    from tactistat.stats_tool.query import bootstrap_artifact

    with pytest.raises(ValueError, match="same player"):
        bootstrap_artifact(config, ["Messi", "Lionel Messi"])


def test_the_bootstrap_command_survives_an_ineligible_player(config, tmp_path, capsys):
    from argparse import Namespace

    from tactistat.cli import _bootstrap

    output = tmp_path / "thin.json"
    args = Namespace(metric="goals", players=["Ao Tanaka", "Lionel Messi"], output=str(output))
    assert _bootstrap(config, args) == 0
    printed = capsys.readouterr().out
    assert "no rate" in printed and "no pair to compare" in printed


def test_a_metric_without_a_per_90_is_refused(config):
    """Summing per-match pass accuracy and dividing by minutes has no referent."""
    from tactistat.stats_tool.query import bootstrap_artifact

    with pytest.raises(ValueError, match="per-90 is not available"):
        bootstrap_artifact(config, ["Lionel Messi", "Kylian Mbappe"], "pass_accuracy")
    # And `minutes` would report 90.00 [90.00, 90.00].
    with pytest.raises(ValueError, match="per-90 is not available"):
        bootstrap_artifact(config, ["Lionel Messi", "Kylian Mbappe"], "minutes")
    # The metrics the engine does normalise still work.
    assert bootstrap_artifact(config, ["Lionel Messi"], "assists")["players"]


def test_a_negative_seed_is_refused_before_numpy_sees_it(config):
    broken = Config(config.to_dict())
    broken.set("eval.bootstrap.seed", -1)
    with pytest.raises(ValueError, match="non-negative"):
        bootstrap_settings(broken)


def test_a_lone_shared_fixture_is_held_fixed_rather_than_resampled():
    """Stratum sizes are conditioned on, so 1-of-1 draws the same match every time."""
    from tactistat.stats_tool.bootstrap import _picks

    assert set(_picks(1, 500, seed=42).ravel()) == {0}

    # So a pair whose only overlap is one fixture gets a shared contribution
    # with no variance: the narrowing is partly that, not only covariance.
    shared_only_left = matches([3], ids=[99])
    shared_only_right = matches([2], ids=[99])
    gap = difference_interval(shared_only_left, shared_only_right, "goals", n_resamples=500)
    assert gap.n_shared == 1 and gap.width == 0.0


def test_the_artifact_holds_the_numbers_it_resampled(config):
    """Match ids say which fixtures were drawn, not what they contributed."""
    from tactistat.stats_tool.bootstrap import per90_interval
    from tactistat.stats_tool.query import bootstrap_artifact

    report = bootstrap_artifact(config, ["Lionel Messi"], "goals")
    player = report["players"][0]

    assert [entry["match_id"] for entry in player["matches"]] == player["match_ids"]
    assert sum(entry["goals"] for entry in player["matches"]) == player["total"]
    assert sum(entry["minutes"] for entry in player["matches"]) == player["minutes"]

    # And the interval is recomputable from the artifact alone.
    frame = pd.DataFrame(player["matches"])
    settings = report["settings"]
    recomputed = per90_interval(
        frame,
        "goals",
        n_resamples=settings["n_resamples"],
        confidence=settings["confidence_level"],
        seed=settings["seed"],
    )
    assert recomputed.to_dict() == player["interval"]


def test_the_content_fingerprint_moves_when_the_numbers_do(config, monkeypatch):
    """The dataset manifest hashes the season, not the parquet behind it."""
    from tactistat.stats_tool.query import StatsQueryEngine, bootstrap_artifact

    engine = StatsQueryEngine(config)
    before = bootstrap_artifact(config, ["Lionel Messi"], "goals", engine=engine)

    messi = int(engine.resolve_player("Lionel Messi").player_id)
    edited = engine.player_matches.copy()
    row = edited.index[edited["player_id"] == messi][0]
    edited.loc[row, "goals"] = edited.loc[row, "goals"] + 1
    engine.player_matches = edited
    after = bootstrap_artifact(config, ["Lionel Messi"], "goals", engine=engine)

    assert before["dataset_fingerprint"] == after["dataset_fingerprint"]
    assert before["matches_fingerprint"] != after["matches_fingerprint"]
