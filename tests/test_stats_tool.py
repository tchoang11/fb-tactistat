"""Tests for minutes, aggregation, queries, and the LangChain stats tool."""

from __future__ import annotations

import pytest

from tactistat.config import ConfigError, load_config
from tactistat.data.statsbomb import DataError, load_events
from tactistat.stats_tool.aggregate import (
    build_player_matches,
    build_player_totals,
    check_configured_metrics,
)
from tactistat.stats_tool.minutes import (
    NOMINAL_PERIOD_END,
    compute_minutes,
    match_length,
    nominal_clock,
)
from tactistat.stats_tool.query import StatsQueryEngine, _strip_accents

# External reference values used to catch errors in the event pipeline.
MESSI = "Lionel Andrés Messi Cuccittini"
MESSI_MINUTES = 690.0
OFFICIAL_TOP_SCORERS = {
    "Kylian Mbappé Lottin": 8,
    MESSI: 7,
    "Olivier Giroud": 4,
    "Julián Álvarez": 4,
}
OFFICIAL_TOTAL_GOALS = 172
N_MATCHES = 64


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
def minutes(config, events):
    return compute_minutes(config, events)


@pytest.fixture(scope="module")
def player_matches(config, events):
    return build_player_matches(config)


@pytest.fixture(scope="module")
def totals(config, player_matches):
    return build_player_totals(config, player_matches)


# Nominal clock


class TestNominalClock:
    """Stoppage time is clipped from the per-90 denominator."""

    def test_a_reading_inside_the_period_is_unchanged(self):
        assert nominal_clock(1, 30, 19) == pytest.approx(30 + 19 / 60)
        assert nominal_clock(2, 73, 54) == pytest.approx(73 + 54 / 60)

    def test_first_half_stoppage_clips_to_forty_five(self):
        assert nominal_clock(1, 52, 4) == 45.0

    def test_second_half_stoppage_clips_to_ninety(self):
        assert nominal_clock(2, 99, 53) == 90.0

    def test_extra_time_periods_clip_to_their_own_ends(self):
        assert nominal_clock(3, 108, 6) == 105.0
        assert nominal_clock(4, 122, 3) == 120.0

    def test_periods_overlap_in_the_raw_clock(self):
        """The period number is required to interpret a raw timestamp."""
        assert nominal_clock(1, 47, 0) == 45.0
        assert nominal_clock(2, 47, 0) == pytest.approx(47.0)


def test_match_length_is_ninety_or_one_twenty(config, events):
    lengths = match_length(config, events)
    assert len(lengths) == N_MATCHES
    assert set(lengths.unique()) <= {90.0, 120.0}
    assert (lengths == 120.0).sum() == 5, "five 2022 World Cup ties went to extra time"


# Minutes played


def test_no_match_over_counts_minutes(config, minutes, events):
    """Player-minutes cannot exceed 22 x match length."""
    lengths = match_length(config, events)
    totals_per_match = minutes.groupby("match_id")["minutes"].sum()
    excess = (totals_per_match - lengths * 22).round(2)
    over = excess[excess > 0.01]
    assert over.empty, f"minutes over-counted in {len(over)} match(es):\n{over.to_string()}"


def test_shortfalls_are_small_and_explained(config, minutes, events):
    """Large shortfalls indicate a missed presence event."""
    lengths = match_length(config, events)
    totals_per_match = minutes.groupby("match_id")["minutes"].sum()
    shortfall = (lengths * 22 - totals_per_match).round(2)
    assert shortfall.max() < 15.0, (
        f"largest shortfall {shortfall.max():.2f} min in match {shortfall.idxmax()}; "
        "a whole missing substitution would look like this"
    )


def test_no_player_exceeds_the_match_length(config, minutes, events):
    lengths = match_length(config, events)
    limit = minutes["match_id"].map(lengths)
    over = minutes[minutes["minutes"] > limit + 1e-6]
    assert over.empty, f"players credited with more than the match length:\n{over}"


def test_messi_played_every_minute(minutes):
    """External check: Messi played about 690 tournament minutes."""
    total = minutes[minutes["player"] == MESSI]["minutes"].sum()
    assert total == pytest.approx(MESSI_MINUTES, abs=1.0)


def test_every_match_fields_at_least_twenty_two_players(minutes):
    counts = minutes[minutes["minutes"] > 0].groupby("match_id").size()
    assert len(counts) == N_MATCHES
    assert counts.min() >= 22


def test_extra_time_players_can_reach_one_twenty(minutes):
    """A guard against silently clipping every match at 90."""
    assert minutes["minutes"].max() == pytest.approx(NOMINAL_PERIOD_END[4])


# Aggregation


def test_goals_survive_aggregation_intact(totals):
    """Aggregation must preserve the official goal totals."""
    assert totals["goals"].sum() + 3 == OFFICIAL_TOTAL_GOALS, "172 = 169 scored + 3 own goals"
    indexed = totals.set_index("player_name")["goals"]
    for player, expected in OFFICIAL_TOP_SCORERS.items():
        assert indexed[player] == expected


def test_shootout_kicks_never_reach_the_tables(totals):
    """Counting period 5 inflates the total to 195 and invents scorers."""
    assert totals["goals"].sum() == 169
    assert totals["goals"].max() == 8, "Mbappé's Golden Boot, not a shootout tally"


def test_substitutes_with_no_events_still_get_a_row(player_matches):
    """The outer join keeps substitutes with minutes but no events."""
    silent = player_matches[
        (player_matches["minutes"] > 0) & (player_matches["passes_attempted"] == 0)
    ]
    assert not silent.empty, "expected at least one player with minutes but no passes"


def test_pass_accuracy_is_undefined_not_zero(player_matches):
    """No pass attempts means undefined accuracy, not 0%."""
    no_passes = player_matches[player_matches["passes_attempted"] == 0]
    assert not no_passes.empty
    assert no_passes["pass_accuracy"].isna().all()


def test_pass_accuracy_is_a_fraction(player_matches):
    defined = player_matches["pass_accuracy"].dropna()
    assert defined.between(0.0, 1.0).all()


def test_xg_is_present_wherever_shots_are(totals):
    shooters = totals[totals["shots"] > 0]
    assert not shooters.empty
    assert (shooters["xg"] > 0).all()


def test_player_id_keeps_name_variants_in_one_row(totals):
    """Phil and Philip Foden are aliases for the same StatsBomb player."""
    assert totals["player_id"].is_unique
    foden = totals[totals["player_id"] == 4354]
    assert len(foden) == 1
    assert foden.iloc[0]["player_name"] == "Phil Foden"
    assert foden.iloc[0]["goals"] == 1
    assert foden.iloc[0]["appearances"] == 4


# Per-90 normalization


def test_per90_is_suppressed_below_the_minutes_threshold(config, totals):
    """Eight minutes and one goal is 11.25 goals/90, and means nothing."""
    threshold = config["dataset.min_minutes_for_per90"]
    below = totals[totals["minutes"] < threshold]
    assert not below.empty
    assert below["goals_per90"].isna().all()

    above = totals[totals["minutes"] >= threshold]
    assert above["goals_per90"].notna().all()


def test_per90_arithmetic_is_right(config, totals):
    threshold = config["dataset.min_minutes_for_per90"]
    eligible = totals[totals["minutes"] >= threshold]
    expected = eligible["goals"] / eligible["minutes"] * 90.0
    assert eligible["goals_per90"].sub(expected).abs().max() < 1e-9


def test_per90_reproduces_a_hand_computed_case(totals):
    """Mbappé: 8 goals in 596.7 minutes is 1.21/90."""
    row = totals.set_index("player_name").loc["Kylian Mbappé Lottin"]
    assert row["goals_per90"] == pytest.approx(row["goals"] / row["minutes"] * 90.0)
    assert row["goals_per90"] == pytest.approx(1.21, abs=0.02)


def test_config_metrics_must_all_be_implemented(config):
    check_configured_metrics(config)  # the real config must pass

    broken = load_config(load_env=False)
    broken.set("stats_tool.metrics", ["goals", "nutmegs"])
    with pytest.raises(ConfigError, match="nutmegs"):
        check_configured_metrics(broken)


# Query interface


@pytest.fixture(scope="module")
def engine(config):
    try:
        return StatsQueryEngine(config)
    except DataError as exc:
        pytest.skip(str(exc))


class TestNameResolution:
    """Resolve common names without guessing ambiguous ones."""

    def test_common_name_resolves_to_the_legal_name(self, engine):
        assert engine.resolve_player("Messi").player_name == MESSI
        assert engine.resolve_player("Mbappe").player_name == "Kylian Mbappé Lottin"

    def test_accents_are_optional(self, engine):
        assert engine.resolve_player("Mbappé").player_name == "Kylian Mbappé Lottin"
        assert _strip_accents("Álvarez") == "alvarez"

    def test_a_shared_surname_is_ambiguous_not_a_guess(self, engine):
        resolution = engine.resolve_player("Alvarez")
        assert resolution.status == "ambiguous"
        assert len(resolution.candidates) > 1
        assert "Julián Álvarez" in resolution.candidates

    def test_a_full_name_disambiguates_it(self, engine):
        assert engine.resolve_player("Julian Alvarez").player_name == "Julián Álvarez"

    def test_a_player_who_was_not_there_is_not_found(self, engine):
        assert engine.resolve_player("Zinedine Zidane").status == "not_found"

    def test_name_variants_resolve_by_player_id(self, engine):
        resolution = engine.resolve_player("Foden")
        assert resolution.status == "ok"
        assert resolution.player_id == 4354
        assert resolution.player_name == "Phil Foden"


class TestQueries:
    def test_engine_reads_committed_tables_without_raw_data(self, config, monkeypatch):
        import tactistat.stats_tool.query as query_module

        monkeypatch.setattr(query_module, "load_matches", lambda _: pytest.fail("raw matches read"))
        monkeypatch.setattr(query_module, "load_lineups", lambda _: pytest.fail("raw lineups read"))
        assert StatsQueryEngine(config).player_metric("Messi", "goals").rows[0].value == 7

    def test_manifest_rejects_a_different_competition(self):
        config = load_config(overrides=["dataset.competition_id=999"], load_env=False)
        with pytest.raises(DataError, match="do not match"):
            StatsQueryEngine(config)

    def test_metric_whitelist_is_enforced(self):
        config = load_config(
            overrides=["stats_tool.metrics=[goals]", "stats_tool.per90_metrics=[goals]"],
            load_env=False,
        )
        answer = StatsQueryEngine(config).player_metric("Messi", "shots")
        assert not answer.ok
        assert "disabled" in answer.note

    def test_player_metric_returns_the_official_number(self, engine):
        answer = engine.player_metric("Messi", "goals")
        assert answer.ok
        assert answer.rows[0].value == 7

    def test_every_answer_carries_traceable_evidence(self, engine):
        """Evidence rows must reproduce the reported total."""
        answer = engine.player_metric("Messi", "goals")
        assert len(answer.evidence) == 7
        assert sum(ref.value for ref in answer.evidence) == answer.rows[0].value
        assert all(ref.match_id > 0 and ref.fixture for ref in answer.evidence)

    def test_zero_minute_stoppage_sub_events_remain_in_evidence(self, engine):
        """A nominal zero-minute appearance can still contain recorded events."""
        answer = engine.player_metric("Filip Djuricic", "key_passes")
        assert answer.rows[0].value == 1
        assert sum(ref.value for ref in answer.evidence) == 1
        assert any(ref.minutes == 0 and ref.value == 1 for ref in answer.evidence)
        assert answer.rows[0].appearances == 2

    def test_leaderboard_ranks_by_the_official_record(self, engine):
        answer = engine.leaderboard("goals", top_n=4)
        assert [row.player_name for row in answer.rows][:2] == [
            "Kylian Mbappé Lottin",
            MESSI,
        ]

    def test_leaderboard_has_evidence_for_every_returned_player(self, engine):
        answer = engine.leaderboard("goals", top_n=4)
        expected_ids = {row.player_id for row in answer.rows}
        assert {ref.player_id for ref in answer.evidence} == expected_ids

    def test_per90_leaderboard_excludes_short_appearances(self, config, engine):
        answer = engine.leaderboard("goals", top_n=10, per90=True)
        threshold = config["dataset.min_minutes_for_per90"]
        assert all(row.minutes >= threshold for row in answer.rows)

    def test_compare_orders_by_value(self, engine):
        answer = engine.compare(["Messi", "Mbappe"], "goals")
        assert [row.value for row in answer.rows] == sorted(
            [row.value for row in answer.rows], reverse=True
        )

    def test_compare_reports_who_it_could_not_resolve(self, engine):
        answer = engine.compare(["Messi", "Zinedine Zidane"], "goals")
        assert answer.ok
        assert len(answer.rows) == 1
        assert "Zidane" in answer.note

    def test_compare_drops_ineligible_per90_rows_without_nan(self, engine):
        answer = engine.compare(["Messi", "Foden"], "goals", per90=True)
        assert answer.ok
        assert [row.player_name for row in answer.rows] == [MESSI]
        assert "Foden" in answer.note
        assert all(row.value == row.value for row in answer.rows)

    def test_an_unknown_metric_is_refused_not_approximated(self, engine):
        answer = engine.player_metric("Messi", "nutmegs")
        assert not answer.ok
        assert "nutmegs" in answer.note

    def test_per90_below_threshold_refuses_with_the_reason(self, engine):
        """A per-90 refusal includes its reason."""
        totals = engine.player_totals
        short = totals[(totals["minutes"] > 0) & ~totals["per90_eligible"]].iloc[0]
        answer = engine.player_metric(short["player_name"], "goals", per90=True)
        assert not answer.ok
        assert "threshold" in answer.note

    def test_valid_metric_without_per90_has_a_specific_refusal(self, engine):
        answer = engine.player_metric("Messi", "pass_accuracy", per90=True)
        assert not answer.ok
        assert "not available" in answer.note

    def test_active_threshold_recomputes_cached_per90(self):
        config = load_config(overrides=["dataset.min_minutes_for_per90=450"], load_env=False)
        engine = StatsQueryEngine(config)
        answer = engine.leaderboard("goals", top_n=50, per90=True)
        assert answer.ok
        assert all(row.minutes >= 450 for row in answer.rows)

    def test_stage_filter_reaggregates_only_matching_games(self, engine):
        answer = engine.player_metric("Messi", "goals", stage="Group Stage")
        assert answer.ok
        assert answer.rows[0].value == 2
        assert answer.rows[0].appearances == 3
        assert {ref.stage for ref in answer.evidence} == {"Group Stage"}

    def test_context_block_states_the_basis(self, engine):
        """Context distinguishes raw totals from per-90 rates."""
        assert "goals per 90" in engine.player_metric("Messi", "goals", per90=True).to_context()
        raw = engine.player_metric("Messi", "goals").to_context()
        assert "goals per 90" not in raw

    def test_a_refusal_renders_as_a_refusal(self, engine):
        context = engine.player_metric("Zinedine Zidane", "goals").to_context()
        assert context.startswith("STATS TOOL: no result")
        assert "Zidane" in context


# LangChain binding


@pytest.fixture(scope="module")
def stats_tool(config, engine):
    from tactistat.stats_tool.langchain_tool import make_stats_tool

    return make_stats_tool(config, engine=engine)


class TestLangChainTool:
    def _call(self, stats_tool, **args):
        return stats_tool.invoke(
            {"type": "tool_call", "id": "t", "name": "football_stats", "args": args}
        )

    def test_the_operation_argument_selects_the_query(self, stats_tool):
        """The router names the operation; it is no longer inferred from arity."""

        def call(**kwargs):
            return self._call(stats_tool, metric="goals", **kwargs).artifact

        ranking = call(operation="ranking", top_n=3)
        lookup = call(operation="player", players=["Messi"])
        comparison = call(operation="compare", players=["Messi", "Mbappe"])
        squad = call(operation="total", team="Argentina")

        assert len(ranking.rows) == 3
        assert len(lookup.rows) == 1 and lookup.rows[0].value == 7
        assert len(comparison.rows) == 2
        # A ranking truncated to top_n does not add up to the squad total.
        assert squad.total.value == 15
        assert squad.total.value > sum(row.value for row in squad.rows[:5])

    def test_the_artifact_carries_the_number_not_a_paraphrase(self, stats_tool):
        """Evaluation receives the structured result and evidence."""
        message = self._call(stats_tool, metric="goals", operation="player", players=["Messi"])
        assert message.artifact.rows[0].value == 7
        assert len(message.artifact.evidence) == 7
        assert isinstance(message.content, str)

    def test_a_refusal_survives_the_binding(self, stats_tool):
        message = self._call(stats_tool, metric="goals", operation="player", players=["Alvarez"])
        assert not message.artifact.ok
        assert "ambiguous" in message.content

    def test_schema_lists_only_implemented_metrics(self, stats_tool):
        from tactistat.stats_tool.aggregate import available_metrics

        metric_schema = stats_tool.args_schema.model_json_schema()["properties"]["metric"]
        described = metric_schema["description"]
        for metric in available_metrics():
            assert metric in described


def test_a_tournament_total_includes_own_goals(engine):
    """Own goals belong to a team and survive in no per-player table."""
    assert engine.total("goals").total.value == 172
    assert sum(engine.player_totals["goals"]) == 169


@pytest.mark.parametrize(
    ("team", "goals", "own"),
    [("Argentina", 15, 0), ("Australia", 4, 1), ("Canada", 2, 1), ("Costa Rica", 3, 1)],
)
def test_a_squad_total_credits_the_own_goals_it_benefited_from(engine, team, goals, own):
    answer = engine.total("goals", team=team)
    assert (answer.total.value, answer.total.own_goals) == (goals, own)
    # The note exists only where an own goal actually contributed.
    assert ("own goal" in (answer.note or "")) is bool(own)


@pytest.mark.parametrize(
    ("operation", "players", "expected"),
    [
        ("compare", ["Lionel Messi"], "compare needs two players"),
        ("compare", [], "compare needs two players"),
        ("player", [], "player lookup needs exactly one player"),
        ("player", ["Lionel Messi", "Kylian Mbappé"], "exactly one player"),
        ("ranking", ["Lionel Messi"], "ranking takes no players"),
        ("total", ["Lionel Messi"], "total takes no players"),
    ],
)
def test_the_dispatcher_refuses_a_wrong_player_count(engine, operation, players, expected):
    """It is also the public LangChain tool: a bad call must not become a leaderboard.

    Falling through returned Mbappe's name at the top of a top-five list to a
    caller who asked about Messi, and reported ok=True while doing it.
    """
    from tactistat.stats_tool.query import run_stats_operation

    answer = run_stats_operation(engine, operation=operation, metric="goals", players=players)
    assert answer.ok is False and expected in answer.note
    assert answer.rows == []


def test_the_dispatcher_still_answers_a_well_formed_call(engine):
    from tactistat.stats_tool.query import run_stats_operation

    compared = run_stats_operation(
        engine, operation="compare", metric="goals", players=["Lionel Messi", "Kylian Mbappé"]
    )
    assert compared.ok is True and len(compared.rows) == 2
    ranked = run_stats_operation(engine, operation="ranking", metric="goals", top_n=3)
    assert ranked.ok is True and len(ranked.rows) == 3
