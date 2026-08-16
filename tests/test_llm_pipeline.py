"""Registry, query translation, routing and synthesis, all without a network call."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from tactistat.config import Config, ConfigError, load_config
from tactistat.llm.registry import ModelSpec, chat_model, parse_handle, resolve, resolve_role
from tactistat.query_translation import QueryRewrite, QueryTranslator, TranslatedQuery
from tactistat.router import Route, RouteDecision, Router
from tactistat.router.route import _capitalised_runs, _split_names_and_team
from tactistat.synthesis import ABSTAIN_TOKEN, Synthesizer

STRUCTURED_ROLES = ("translator", "router", "judge")


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config()


def fake(value):
    """A model stand-in that ignores its prompt and returns `value`."""
    return RunnableLambda(lambda _: value)


def exploding(exc: Exception):
    def boom(_):
        raise exc

    return RunnableLambda(boom)


# Registry


@pytest.mark.parametrize("handle", ["groq", "groq:", ":gpt-oss-20b", "a:b:c", "", None])
def test_parse_handle_rejects_malformed_handles(handle):
    with pytest.raises(ConfigError):
        parse_handle(handle)


def test_resolve_flattens_provider_settings_into_the_spec():
    spec = resolve("dashscope:qwen-turbo")
    assert (spec.provider, spec.model_id, spec.kind) == ("dashscope", "qwen-turbo", "openai_compat")
    assert spec.api_key_env == "DASHSCOPE_API_KEY"
    # A provider-level quirk must reach the model, not just the probe script.
    assert spec.extra_body == {"enable_thinking": False}


@pytest.mark.parametrize("handle", ["nosuch:model", "groq:nosuch"])
def test_resolve_reports_unknown_providers_and_aliases(handle):
    with pytest.raises(ConfigError, match="Unknown"):
        resolve(handle)


@pytest.mark.parametrize(
    ("json_mode", "method"), [("schema", "json_schema"), ("object", "json_mode")]
)
def test_measured_json_mode_selects_the_structured_output_method(json_mode, method):
    spec = ModelSpec("p:a", "p", "openai_compat", "a", json_mode)
    assert spec.structured_method == method


def test_a_model_without_json_support_cannot_take_a_structured_role():
    spec = ModelSpec("ollama:qwen2.5-7b", "ollama", "openai_compat", "qwen2.5:7b", "none")
    with pytest.raises(ConfigError, match="structured-output role"):
        _ = spec.structured_method


def test_unknown_role_names_the_roles_that_exist(config):
    with pytest.raises(ConfigError, match="No model configured"):
        resolve_role(config, "nosuchrole")


@pytest.mark.parametrize("role", STRUCTURED_ROLES)
def test_every_structured_role_is_configured_to_a_model_that_can_do_it(config, role):
    assert resolve_role(config, role).structured_method


def test_a_missing_api_key_is_reported_before_the_call(config, monkeypatch):
    spec = resolve_role(config, "router")
    monkeypatch.setenv(spec.api_key_env, "")
    with pytest.raises(ConfigError, match=spec.api_key_env):
        chat_model(config, "router")


# Query translation


def test_translation_can_be_disabled_without_touching_a_model(config):
    config = Config(config.to_dict())
    config.set("query_translation.enabled", False)
    result = QueryTranslator(
        config, model=exploding(AssertionError("must not be called"))
    ).translate("Messi bàn thắng")
    assert (result.translated, result.query) == (False, "Messi bàn thắng")
    assert result.retrieval_queries == ["Messi bàn thắng"]


def test_an_unknown_strategy_fails_before_a_model_is_built(config):
    config = Config(config.to_dict())
    config.set("query_translation.strategy", "translate-ish")
    with pytest.raises(ConfigError, match="query_translation.strategy"):
        QueryTranslator(config)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("off", ["Messi goals"]),
        ("rewrite", ["Messi goals"]),
        ("multi_query", ["Messi goals", "Messi scoring record", "Argentina captain goals"]),
        # HyDE is defined by exactly one hypothetical document.
        ("hyde", ["Messi scoring record"]),
    ],
)
def test_each_strategy_decides_what_is_actually_retrieved(config, strategy, expected):
    config = Config(config.to_dict())
    config.set("query_translation.strategy", strategy)
    reply = QueryRewrite(
        source_language="vi",
        query="Messi goals",
        variants=["Messi scoring record", "Argentina captain goals"],
    )
    result = QueryTranslator(config, model=fake(reply)).translate("Messi bàn thắng")
    assert result.retrieval_queries == expected
    assert result.source_language == "vi"


def test_variants_are_deduplicated_and_capped_at_n_variants(config):
    config = Config(config.to_dict())
    config.set("query_translation.strategy", "multi_query")
    config.set("query_translation.n_variants", 2)
    reply = QueryRewrite(source_language="vi", query="q", variants=["a", "a", " ", "b", "c"])
    result = QueryTranslator(config, model=fake(reply)).translate("Messi bàn thắng")
    assert result.retrieval_queries == ["q", "a", "b"]


@pytest.mark.parametrize(
    ("reported", "expected"),
    [("vi", "vi"), ("VI ", "vi"), ("Vietnamese", "unknown"), ("", "unknown")],
)
def test_the_reported_language_is_normalised_to_a_code(config, reported, expected):
    reply = QueryRewrite(source_language=reported, query="q", variants=[])
    result = QueryTranslator(config, model=fake(reply)).translate("Messi bàn thắng")
    assert result.source_language == expected


def test_a_provider_failure_degrades_to_the_original_question(config):
    result = QueryTranslator(config, model=exploding(RuntimeError("429 rate limit"))).translate(
        "Messi bàn thắng"
    )
    assert isinstance(result, TranslatedQuery)
    assert (result.translated, result.query) == (False, "Messi bàn thắng")
    assert "429" in result.note


def test_an_empty_question_never_reaches_a_model(config):
    result = QueryTranslator(config, model=exploding(AssertionError())).translate("   ")
    assert result.note == "empty question"


# Router


def test_an_unknown_router_strategy_fails_before_a_model_is_built(config):
    config = Config(config.to_dict())
    config.set("router.strategy", "vibes")
    with pytest.raises(ConfigError, match="router.strategy"):
        Router(config)


def test_a_metric_outside_the_configured_whitelist_is_routed_to_prose(config):
    decision = RouteDecision(label="STAT", metric="yellow_cards", players=["Messi"])
    route = Router(config, model=fake(decision)).route("How many yellow cards did Messi get?")
    assert route.label == "TACTICAL"
    assert route.stats_args is None and route.needs_rag
    assert "yellow_cards" in route.repairs[0]


def test_per90_is_dropped_for_a_metric_that_has_no_per90_rate(config):
    decision = RouteDecision(label="STAT", metric="tackles", per90=True)
    route = Router(config, model=fake(decision)).route("Most tackles per 90?")
    assert route.stats_args["per90"] is False
    assert "per-90" in route.repairs[0]


@pytest.mark.parametrize("top_n", [0, -3, 999])
def test_an_out_of_range_top_n_falls_back_to_the_default(config, top_n):
    decision = RouteDecision(label="STAT", metric="goals", top_n=top_n)
    route = Router(config, model=fake(decision)).route("Top scorers?")
    assert route.stats_args["top_n"] == 5


def test_a_label_disabled_in_config_cannot_reach_a_tool(config):
    config = Config(config.to_dict())
    config.set("router.labels", ["STAT", "TACTICAL"])
    decision = RouteDecision(label="HYBRID", metric="goals", rag_query="messi")
    route = Router(config, model=fake(decision)).route("Was Messi the best?")
    assert route.label == "TACTICAL" and route.stats_args is None


def test_a_tactical_route_without_a_query_falls_back_to_the_question(config):
    decision = RouteDecision(label="TACTICAL", rag_query="  ")
    route = Router(config, model=fake(decision)).route("Why did Morocco defend so well?")
    assert route.rag_query == "Why did Morocco defend so well?"


def test_a_router_failure_degrades_to_the_keyword_baseline(config):
    router = Router(config, model=exploding(RuntimeError("503")))
    route = router.route("How many goals did Lionel Messi score?")
    assert isinstance(route, Route) and route.label == "STAT"
    assert route.stats_args["metric"] == "goals"
    assert "used keywords" in route.repairs[0]


@pytest.mark.parametrize(
    ("question", "label"),
    [
        ("How many goals did Lionel Messi score?", "STAT"),
        ("Why was Morocco hard to break down?", "TACTICAL"),
        ("Who scored the most goals, and why was he so effective?", "HYBRID"),
        ("What was the atmosphere like in Qatar?", "TACTICAL"),
    ],
)
def test_the_keyword_baseline_labels_without_a_model(config, question, label):
    config = Config(config.to_dict())
    config.set("router.strategy", "keyword")
    assert Router(config).route(question).label == label


def test_capitalised_question_words_are_not_sent_to_the_player_resolver():
    names = _capitalised_runs("How many goals did Lionel Messi score at the FIFA World Cup?")
    assert names == ["Lionel Messi"]


@pytest.mark.parametrize(
    "question",
    [
        "In the final, who scored?",
        "Do Messi and Mbappe compare well?",
        "At the Final, how many goals?",
    ],
)
def test_short_sentence_initial_words_are_not_mistaken_for_names(question):
    """ "In", "Do" and "At" capitalise exactly like a surname."""
    assert all(len(name) >= 4 for name in _capitalised_runs(question))


def test_a_national_team_is_read_as_a_team_not_a_player():
    names, team = _split_names_and_team(
        "How many goals did Argentina score?", {"argentina", "france"}
    )
    assert (names, team) == ([], "Argentina")


# Synthesis


def test_synthesis_abstains_when_no_tool_returned_evidence(config):
    answer = Synthesizer(config, model=exploding(AssertionError())).answer("Who won?")
    assert (answer.abstained, answer.ok) == (True, False)
    assert answer.note == "no tool evidence"


def test_the_abstain_token_is_stripped_from_the_answer_shown_to_the_user(config):
    reply = AIMessage(content=f"{ABSTAIN_TOKEN}: the passages never mention 1998.")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Top scorer in 1998?", rag_context="RAG TOOL — 1 passage", n_passages=1
    )
    assert answer.abstained and ABSTAIN_TOKEN not in answer.text
    assert answer.text == "the passages never mention 1998"


def test_a_citation_pointing_at_no_retrieved_passage_is_deleted(config):
    reply = AIMessage(content="Morocco defended in a 4-1-4-1 [1] and conceded once [7].")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Morocco defence?", rag_context="RAG TOOL — 2 passages", n_passages=2
    )
    assert answer.cited == [1] and "[7]" not in answer.text
    assert "removed 1 fabricated citation" in answer.note and answer.ok is False


def test_fullwidth_brackets_still_count_as_citations(config):
    """gpt-oss writes 【2】; the citation is real, only the glyph differs."""
    reply = AIMessage(content="Morocco defended in a compact 4-1-4-1 shape【2】.")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Morocco defence?",
        rag_context="RAG TOOL — 2 passages\n[2] Morocco defended in a 4-1-4-1 shape.",
        n_passages=2,
    )
    assert answer.cited == [2] and answer.ok is True


def test_content_returned_as_blocks_is_read_as_text(config):
    """Gemini returns a list of content blocks where Groq returns a string."""
    reply = AIMessage(content=[{"type": "text", "text": "Messi scored 7 goals."}])
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Messi goals?", stats_context="STATS TOOL — goals\n  Lionel Messi (Argentina): 7.00"
    )
    assert answer.text == "Messi scored 7 goals." and answer.ok is True


def test_a_stats_only_answer_is_never_asked_to_cite_a_passage(config):
    captured = {}

    def record(prompt):
        captured["system"] = prompt[0][1]
        return AIMessage(content="Lionel Messi scored 7 goals.")

    Synthesizer(config, model=RunnableLambda(record)).answer(
        "Messi goals?", stats_context="STATS TOOL — goals", n_passages=0
    )
    assert "Never write square brackets" in captured["system"]


def test_prose_with_no_citation_at_all_is_flagged_as_unsupported(config):
    reply = AIMessage(content="Morocco were simply very well organised.")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Morocco defence?", rag_context="RAG TOOL — 2 passages", n_passages=2
    )
    assert answer.ok is False and "cites no retrieved passage" in answer.note


def test_a_stats_answer_needs_no_bracket_when_nothing_was_retrieved(config):
    reply = AIMessage(content="Lionel Messi scored 7 goals at the tournament.")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Messi goals?",
        stats_context="STATS TOOL — goals\n  Lionel Messi (Argentina): 7.00",
        n_passages=0,
    )
    assert answer.ok is True and answer.cited == []


def test_a_hybrid_answer_citing_nothing_is_not_rescued_by_its_stats_block(config):
    """Passages were retrieved, so the prose beside the number must cite one."""
    reply = AIMessage(content="Messi scored 7 goals and was the tournament's best player.")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Was Messi the best?",
        stats_context="STATS TOOL — goals\n  Lionel Messi (Argentina): 7.00",
        rag_context="RAG TOOL — 2 passages",
        n_passages=2,
    )
    assert answer.ok is False and "cites no retrieved passage" in answer.note


def test_the_abstain_token_does_not_smuggle_a_fabricated_claim_past_the_guards(config):
    reply = AIMessage(content=f"Messi scored 999 goals. {ABSTAIN_TOKEN} nothing else is known.")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Messi goals?", stats_context="STATS TOOL — goals\n  Lionel Messi (Argentina): 7.00"
    )
    assert answer.abstained is True and answer.ok is False
    assert "999" in answer.note


def test_a_digit_inside_a_date_does_not_support_a_different_number(config):
    """Substring matching made "8" supported by the 8 in 2022-12-18."""
    evidence = (
        "STATS TOOL — goals\n  Lionel Messi (Argentina): 7.00 [690 min, 7 apps]\n"
        "    2022-12-18  Argentina 3-3 France (Final): 2 in 120 min"
    )
    answer = Synthesizer(
        config, model=fake(AIMessage(content="Messi scored 8 goals here."))
    ).answer("Messi goals?", stats_context=evidence)
    assert answer.ok is False and "8" in answer.note


def test_a_provider_failure_does_not_end_an_evaluation_run(config):
    answer = Synthesizer(config, model=exploding(RuntimeError("500"))).answer(
        "Messi goals?", stats_context="STATS TOOL — goals"
    )
    assert (answer.ok, answer.abstained) == (False, True)
    assert "500" in answer.note


# Operation contract: a truncated ranking is not a total


def test_a_squad_total_is_a_different_operation_from_a_ranking(config):
    decision = RouteDecision(
        label="STAT", operation="total", metric="goals", team="Argentina", players=[]
    )
    route = Router(config, model=fake(decision)).route("How many goals did Argentina score?")
    assert route.stats_args["operation"] == "total"
    assert route.stats_args["team"] == "Argentina"


@pytest.mark.parametrize(
    ("operation", "players", "expected"),
    [
        ("compare", ["Messi"], "player"),
        ("compare", [], "ranking"),
        ("player", [], "ranking"),
    ],
)
def test_an_operation_that_contradicts_the_player_list_is_repaired(
    config, operation, players, expected
):
    decision = RouteDecision(label="STAT", operation=operation, metric="goals", players=players)
    route = Router(config, model=fake(decision)).route("Goals?")
    assert route.stats_args["operation"] == expected
    assert route.repairs


def test_a_total_ignores_names_the_model_attached_to_it(config):
    decision = RouteDecision(
        label="STAT", operation="total", metric="goals", team="Argentina", players=["Messi"]
    )
    route = Router(config, model=fake(decision)).route("How many goals did Argentina score?")
    assert route.stats_args["players"] == []
    assert "ignores the named players" in route.repairs[0]


def test_per90_is_meaningless_for_a_squad_total(config):
    decision = RouteDecision(label="STAT", operation="total", metric="goals", per90=True)
    route = Router(config, model=fake(decision)).route("Goals per 90 for Argentina?")
    assert route.stats_args["per90"] is False
    assert "squad total" in route.repairs[0]


def test_the_repair_label_respects_a_config_that_disables_tactical(config):
    config = Config(config.to_dict())
    config.set("router.labels", ["STAT", "HYBRID"])
    decision = RouteDecision(label="STAT", metric="yellow_cards")
    route = Router(config, model=fake(decision)).route("How many yellow cards?")
    assert route.label == "HYBRID" and route.label in config["router.labels"]


def test_structured_output_that_does_not_match_the_schema_falls_back(config):
    """A json_mode model returns a mapping that need not satisfy RouteDecision."""
    router = Router(config, model=fake({"label": "BANANA", "metric": 7}))
    route = router.route("How many goals did Lionel Messi score?")
    assert route.label in config["router.labels"]
    assert "router model failed" in route.repairs[0]


def test_the_keyword_fallback_keeps_the_team_and_stage_it_can_see(config):
    config = Config(config.to_dict())
    config.set("router.strategy", "keyword")
    router = Router(config, known_teams={"argentina"})
    route = router.route("How many goals did Argentina score in the Group Stage?")
    assert route.stats_args["team"] == "Argentina"
    assert route.stats_args["stage"] == "Group Stage"
    assert route.stats_args["operation"] == "total"


# Synthesis guards


def test_an_empty_or_near_empty_reply_is_not_a_valid_answer(config):
    answer = Synthesizer(config, model=fake(AIMessage(content="  "))).answer(
        "Messi goals?", stats_context="STATS TOOL — goals"
    )
    assert (answer.ok, answer.abstained) == (False, True)
    assert "too short" in answer.note


def test_a_fabricated_citation_invalidates_the_answer_not_just_the_bracket(config):
    reply = AIMessage(content="Morocco pressed high [7] and defended deep [1].")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Morocco defence?", rag_context="RAG TOOL — 2 passages", n_passages=2
    )
    # The claim beside a deleted citation stays in the text but must not pass.
    assert answer.ok is False and "fabricated citation" in answer.note


def test_a_number_the_tool_never_produced_is_caught(config):
    reply = AIMessage(content="Lionel Messi scored 8.00 goals at the tournament.")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Messi goals?", stats_context="STATS TOOL — goals\n  Lionel Messi (Argentina): 7.00"
    )
    assert answer.ok is False and "8.00" in answer.note


def test_a_number_written_more_briefly_than_the_tool_wrote_it_is_supported(config):
    reply = AIMessage(content="Lionel Messi scored 7 goals in 690 minutes at the 2022 World Cup.")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Messi goals?",
        stats_context="STATS TOOL — goals\n  Lionel Messi (Argentina): 7.00 [690 min, 7 apps]",
    )
    assert answer.ok is True and answer.note is None


def test_a_localised_decimal_is_reported_without_failing_the_answer(config):
    """Same value, rewritten shape: wrong for a fidelity check, right in prose."""
    reply = AIMessage(content="Mbappe ghi 0,94 bàn mỗi 90 phút, cao hơn Messi một chút.")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "So sánh?", stats_context="STATS TOOL — goals per 90\n  Kylian Mbappe (France): 0.94"
    )
    assert answer.ok is True and "reformatted" in answer.note and "0,94" in answer.note


def test_an_operation_outside_the_contract_is_repaired(config):
    """A json_mode model can return a value the Literal would have rejected."""
    decision = RouteDecision.model_construct(
        label="STAT",
        operation="summarise",
        metric="goals",
        players=["Messi"],
        per90=False,
        top_n=5,
        team=None,
        stage=None,
        rag_query=None,
    )
    route = Router(config, model=fake(decision)).route("Goals?")
    assert route.stats_args["operation"] == "ranking"
    assert "unknown operation" in route.repairs[0]


def test_a_three_digit_bracket_is_stripped_not_mistaken_for_a_citation(config):
    """Models bracket stray numbers; no passage rank is ever three digits."""
    reply = AIMessage(content="Mbappe scored 8.00 goals in 597 minutes[597], the most [2].")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Top scorer?",
        stats_context="STATS TOOL — goals\n  Kylian Mbappe (France): 8.00 [597 min, 7 apps]",
        rag_context="RAG TOOL — 2 passages",
        n_passages=2,
    )
    assert "[597]" not in answer.text and answer.cited == [2]


# Orchestration contract


@pytest.mark.parametrize(
    ("strategy", "variants"),
    [
        # The ablation axis owns retrieval, or `off` and `rewrite` would be
        # identical at retrieval time and the axis would measure nothing.
        ("off", ["a literal translation"]),
        ("rewrite", ["a self-contained question"]),
        ("multi_query", ["q", "v1", "v2"]),
        ("hyde", ["a hypothetical passage"]),
    ],
)
def test_the_strategy_not_the_router_decides_what_is_retrieved(strategy, variants):
    from tactistat.pipeline import retrieval_queries_for

    translation = TranslatedQuery(
        original="x",
        query="q",
        source_language="vi",
        strategy=strategy,
        retrieval_queries=variants,
    )
    route = Route("TACTICAL", "q", "few_shot", rag_query="router query")
    assert retrieval_queries_for(route, translation) == variants


def test_the_router_query_is_used_only_when_translation_did_not_run():
    from tactistat.pipeline import retrieval_queries_for

    translation = TranslatedQuery(
        original="x",
        query="x",
        source_language="unknown",
        strategy="rewrite",
        retrieval_queries=["x"],
        translated=False,
    )
    route = Route("TACTICAL", "x", "few_shot", rag_query="router query")
    assert retrieval_queries_for(route, translation) == ["router query"]


def test_a_stat_route_retrieves_nothing():
    from tactistat.pipeline import retrieval_queries_for

    translation = TranslatedQuery(
        original="x",
        query="q",
        source_language="vi",
        strategy="multi_query",
        retrieval_queries=["q", "v1"],
    )
    route = Route("STAT", "q", "few_shot", stats_args={"metric": "goals"}, rag_query=None)
    assert retrieval_queries_for(route, translation) == []


def test_a_failed_tool_is_not_handed_to_synthesis_as_evidence(config):
    """ "STATS TOOL: no result" would otherwise be summarised into an answer."""
    from tactistat.pipeline import TactiStatPipeline
    from tactistat.stats_tool.query import StatsAnswer

    class RefusingEngine:
        player_totals = None

    refusal = StatsAnswer("goals", False, [], ok=False, note="no player matching 'Pele'")
    pipeline = TactiStatPipeline(
        config,
        translator=QueryTranslator(
            config,
            model=fake(
                QueryRewrite(
                    source_language="en", query="How many goals did Pele score?", variants=[]
                )
            ),
        ),
        router=Router(
            config,
            model=fake(
                RouteDecision(label="STAT", operation="player", metric="goals", players=["Pele"])
            ),
        ),
        engine=RefusingEngine(),
        synthesizer=Synthesizer(config, model=exploding(AssertionError("no evidence to send"))),
    )
    pipeline._run_stats = lambda route: refusal
    result = pipeline.run("How many goals did Pele score?")
    assert result.tool_failures and result.ok is False
    assert result.answer.abstained is True


@pytest.mark.parametrize(
    ("question", "metric"),
    [
        ("who was the best goalkeeper?", None),
        ("how many goals were scored?", "goals"),
        ("who made the most key passes?", "key_passes"),
        ("most tackles?", "tackles"),
    ],
)
def test_the_keyword_baseline_matches_whole_words_only(question, metric):
    """ "goalkeeper" contains "goal" but asks about no configured metric."""
    from tactistat.router.route import _keyword_metric

    assert _keyword_metric(question) == metric


# Numbers must be bound to the metric they are claimed of


@pytest.mark.parametrize(
    "claim",
    [
        "Messi scored 9 goals.",
        "Messi scored 18 goals.",  # 18 also appears in the date 2022-12-18
        "Messi scored 2023 goals.",  # a year, but claimed as a count
        "Messi ghi 9 bàn thắng ở giải.",  # the same claim in Vietnamese
    ],
)
def test_a_wrong_count_is_caught_even_when_the_digits_appear_elsewhere(config, claim):
    evidence = (
        "STATS TOOL — goals\n  Lionel Messi (Argentina): 7.00 [690 min, 7 apps]\n"
        "    2022-12-18  Argentina 3-3 France (Final): 2 in 120 min"
    )
    answer = Synthesizer(config, model=fake(AIMessage(content=claim))).answer(
        "Messi goals?", stats_context=evidence, stats_claims={"goals": {7.0, 2.0}}
    )
    assert answer.ok is False and "never computed" in answer.note


def test_the_right_count_still_passes(config):
    evidence = "STATS TOOL — goals\n  Lionel Messi (Argentina): 7.00 [690 min, 7 apps]"
    answer = Synthesizer(config, model=fake(AIMessage(content="Messi scored 7 goals."))).answer(
        "Messi goals?", stats_context=evidence, stats_claims={"goals": {7.0}}
    )
    assert answer.ok is True and answer.note is None


def test_an_invented_scoreline_does_not_hide_inside_punctuation(config):
    """ "8-0" escaped the standalone-number regex entirely."""
    evidence = "RAG TOOL — 1 passage\n[1] Argentina beat Poland 2-0 in the group stage."
    answer = Synthesizer(config, model=fake(AIMessage(content="Argentina won 8-0 [1]."))).answer(
        "Result?", rag_context=evidence, n_passages=1
    )
    assert answer.ok is False and "8" in answer.note


def test_an_abstention_cannot_carry_an_uncited_claim(config):
    reply = AIMessage(content=f"Morocco pressed high up the pitch. {ABSTAIN_TOKEN} little else.")
    answer = Synthesizer(config, model=fake(reply)).answer(
        "Morocco press?", rag_context="RAG TOOL — 2 passages", n_passages=2
    )
    assert answer.ok is False and "cites no retrieved passage" in answer.note


# Pipeline status


def test_a_hybrid_that_lost_one_branch_is_partial_not_successful(config):
    from tactistat.pipeline import TactiStatPipeline
    from tactistat.rag_tool.retrieve import Passage, RagAnswer
    from tactistat.stats_tool.query import StatsAnswer

    passage = Passage(1, "Morocco defended deep.", "Morocco", None, "u", 1, "team", None, 1)
    pipeline = TactiStatPipeline(
        config,
        translator=QueryTranslator(
            config,
            model=fake(QueryRewrite(source_language="en", query="Was Messi best?", variants=[])),
        ),
        router=Router(
            config,
            model=fake(
                RouteDecision(
                    label="HYBRID",
                    operation="player",
                    metric="goals",
                    players=["Pele"],
                    rag_query="Messi",
                )
            ),
        ),
        engine=object(),
        retriever=object(),
        synthesizer=Synthesizer(
            config, model=fake(AIMessage(content="Morocco defended deep [1]."))
        ),
    )
    pipeline._run_stats = lambda route: StatsAnswer("goals", False, [], ok=False, note="no Pele")
    pipeline._run_rag = lambda queries: RagAnswer("Messi", "dense", False, passages=[passage])
    result = pipeline.run("Was Messi the best?")
    assert result.answer.ok is True and result.tool_failures
    assert result.status == "partial" and result.ok is False


def test_a_tool_that_raises_becomes_a_refusal_not_a_crash(config):
    from tactistat.pipeline import TactiStatPipeline

    class Exploding:
        player_totals = None

    pipeline = TactiStatPipeline(
        config,
        translator=QueryTranslator(
            config, model=fake(QueryRewrite(source_language="en", query="Goals?", variants=[]))
        ),
        router=Router(
            config, model=fake(RouteDecision(label="STAT", operation="ranking", metric="goals"))
        ),
        engine=Exploding(),
        synthesizer=Synthesizer(config, model=exploding(AssertionError("no evidence"))),
    )
    result = pipeline.run("Top scorers?")
    assert result.stats is not None and result.stats.ok is False
    assert "raised" in result.stats.note and result.status == "failed"


def test_disabling_translation_retrieves_the_untouched_question():
    """The router's rewrite is itself an LLM translation; the baseline must not use it."""
    from tactistat.pipeline import retrieval_queries_for

    translation = TranslatedQuery(
        original="Vì sao Ma-rốc phòng ngự tốt?",
        query="Vì sao Ma-rốc phòng ngự tốt?",
        source_language="unknown",
        strategy="rewrite",
        retrieval_queries=["Vì sao Ma-rốc phòng ngự tốt?"],
        translated=False,
        status="disabled",
    )
    route = Route("TACTICAL", "q", "few_shot", rag_query="Morocco defensive tactics")
    assert retrieval_queries_for(route, translation) == ["Vì sao Ma-rốc phòng ngự tốt?"]


def test_a_translation_failure_may_fall_back_and_says_so():
    from tactistat.pipeline import retrieval_queries_for

    translation = TranslatedQuery(
        original="x",
        query="x",
        source_language="unknown",
        strategy="rewrite",
        retrieval_queries=["x"],
        translated=False,
        status="failed",
        note="translation failed: 429",
    )
    route = Route("TACTICAL", "x", "few_shot", rag_query="router query")
    assert retrieval_queries_for(route, translation) == ["router query"]
    assert translation.to_dict()["status"] == "failed"


# Keyword router: a total is not a ranking


@pytest.mark.parametrize(
    ("question", "operation"),
    [
        ("How many goals were scored at the tournament?", "total"),
        ("At the Final, how many goals were scored?", "total"),
        ("Who scored the most goals?", "ranking"),
        ("How many goals did Argentina score?", "total"),
    ],
)
def test_the_keyword_baseline_separates_a_total_from_a_ranking(config, question, operation):
    config = Config(config.to_dict())
    config.set("router.strategy", "keyword")
    router = Router(config, known_teams={"argentina"})
    assert router.route(question).stats_args["operation"] == operation


def test_the_stats_context_states_what_it_was_filtered_to(config):
    """Without the scope line, synthesis cannot tell a fixture from a tournament."""
    from tactistat.stats_tool.query import scope_label

    assert scope_label(team="Argentina", opponent="Poland") == (
        "Argentina in the match against Poland"
    )
    assert scope_label() == "the whole 2022 FIFA World Cup"
    assert "Group Stage" in scope_label(team="Argentina", stage="Group Stage")


def test_a_full_hybrid_question_succeeds_end_to_end(config):
    """Both tools answer, the answer cites a passage, and nothing is flagged."""
    from tactistat.pipeline import TactiStatPipeline
    from tactistat.rag_tool.retrieve import Passage, RagAnswer
    from tactistat.stats_tool.query import PlayerRow, StatsAnswer

    passage = Passage(
        1,
        "Messi was named player of the tournament.",
        "Lionel Messi",
        "2022 World Cup",
        "https://example.org",
        1,
        "player",
        "messi",
        1,
    )
    stats = StatsAnswer(
        "goals",
        False,
        [PlayerRow(1, "Lionel Messi", "Argentina", 7.0, 690.0, 7)],
        scope="the whole 2022 FIFA World Cup",
    )
    reply = AIMessage(content="Messi scored 7 goals and was player of the tournament [1].")
    pipeline = TactiStatPipeline(
        config,
        translator=QueryTranslator(
            config,
            model=fake(QueryRewrite(source_language="vi", query="Was Messi best?", variants=[])),
        ),
        router=Router(
            config,
            model=fake(
                RouteDecision(
                    label="HYBRID",
                    operation="player",
                    metric="goals",
                    players=["Lionel Messi"],
                    rag_query="Lionel Messi",
                )
            ),
        ),
        engine=object(),
        retriever=object(),
        synthesizer=Synthesizer(config, model=fake(reply)),
    )
    pipeline._run_stats = lambda route: stats
    pipeline._run_rag = lambda queries: RagAnswer("Messi", "dense", False, passages=[passage])

    result = pipeline.run("Messi có phải hay nhất giải?")
    assert result.status == "ok" and result.ok is True
    assert result.answer.cited == [1] and result.answer.note is None
    assert result.tool_failures == []
    assert set(result.timings_ms) >= {"translate", "route", "stats", "retrieve", "synthesize"}


def test_search_multi_fuses_variants_and_renumbers_the_ranks():
    """multi_query and hyde depend on this; a single query must behave as before."""
    from tactistat.rag_tool.retrieve import Passage, RagAnswer, RagRetriever

    def passage(rank, chunk_id):
        return Passage(rank, f"text {chunk_id}", "T", None, "u", 1, "e", None, chunk_id)

    results = {
        "a": [passage(1, 10), passage(2, 20)],
        "b": [passage(1, 20), passage(2, 30)],
    }
    retriever = RagRetriever.__new__(RagRetriever)
    retriever.mode, retriever.top_k, retriever.candidate_k, retriever.reranked = (
        "dense",
        5,
        20,
        False,
    )
    retriever.search = lambda query, top_k=None: RagAnswer(
        query, "dense", False, passages=results[query]
    )

    fused = RagRetriever.search_multi(retriever, ["a", "b", "a", "  "])
    assert [p.chunk_id for p in fused.passages][0] == 20  # ranked first by both
    assert [p.rank for p in fused.passages] == [1, 2, 3]
    assert "fused 2 query variants" in fused.note


def test_the_ask_command_reports_an_unverified_answer_and_its_sources(config, capsys):
    """The CLI is where an unsupported claim would otherwise look authoritative."""
    import tactistat.cli as cli_module
    from tactistat.pipeline import PipelineResult
    from tactistat.rag_tool.retrieve import Passage, RagAnswer
    from tactistat.synthesis.synthesize import Answer

    passage = Passage(1, "text", "Morocco", "Defence", "https://example.org/m", 1, "team", None, 1)
    result = PipelineResult(
        question="q",
        translation=TranslatedQuery("q", "q", "en", "rewrite", ["q"]),
        route=Route("TACTICAL", "q", "few_shot", rag_query="q"),
        answer=Answer(
            "q",
            "Morocco conceded 8 goals [1].",
            cited=[1],
            ok=False,
            note="number(s) not found in the evidence: 8",
        ),
        rag=RagAnswer("q", "dense", False, passages=[passage]),
    )

    class Args:
        question, trace = "q", False

    monkey = cli_module.TactiStatPipeline
    cli_module.TactiStatPipeline = lambda config: type("P", (), {"run": lambda self, q: result})()
    try:
        code = cli_module._ask(config, Args())
    finally:
        cli_module.TactiStatPipeline = monkey

    out = capsys.readouterr().out
    assert code == 1
    assert "UNVERIFIED" in out and "not found in the evidence" in out
    assert "https://example.org/m" in out
