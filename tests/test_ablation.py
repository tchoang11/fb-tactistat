"""Axis definitions, arm isolation, and the comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tactistat.config import Config, load_config
from tactistat.eval.ablate import (
    AXES,
    MODES,
    Arm,
    Axis,
    arm_config,
    comparison_table,
    retrieval_items,
    run_axis,
    score_retrieval_arm,
)
from tactistat.eval.schema import load_test_set


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config(
        load_env=False, overrides=["tracing.enabled=false", "llm.cache_enabled=false"]
    )


def test_every_axis_declares_a_known_mode_and_at_least_two_arms():
    for name, axis in AXES.items():
        assert axis.name == name
        assert axis.mode in MODES
        assert axis.question.endswith("?")
        assert len({arm.name for arm in axis.arms}) == len(axis.arms)
        if name != "baseline":
            assert len(axis.arms) >= 2, f"{name} cannot be an ablation with one arm"


def test_an_arm_overrides_only_its_own_keys(config):
    arm = Arm("hybrid", {"rag.retrieval.mode": "hybrid"})
    built = arm_config(config, arm)
    assert built.get("rag.retrieval.mode") == "hybrid"
    # The base config is untouched, or the next arm inherits this one.
    assert config.get("rag.retrieval.mode") == "dense"
    assert built.get("rag.chunking.strategy") == config.get("rag.chunking.strategy")


def test_an_index_rebuilding_arm_gets_its_own_directory(config, tmp_path):
    """Sharing one index would compare an arm against another arm's vectors."""
    section = arm_config(config, Arm("section", {}), index_root=tmp_path)
    fixed = arm_config(config, Arm("fixed", {}), index_root=tmp_path)
    assert section.get("rag.index_dir") != fixed.get("rag.index_dir")
    assert section.get("rag.index_dir").endswith("section")


def test_only_labelled_items_can_score_retrieval(config):
    suite = load_test_set(config.path("eval.test_set"))
    items = retrieval_items(suite)
    assert items and all(item.retrieval is not None for item in items)
    assert all(item.route.label in ("TACTICAL", "HYBRID") for item in items)


def _stub_pipeline(passages_by_query=None):
    from tactistat.query_translation import TranslatedQuery
    from tactistat.rag_tool.retrieve import Passage, RagAnswer
    from tactistat.router import Route

    class Stub:
        def __init__(self):
            self.seen = []

        def translate(self, question, history=None):
            return TranslatedQuery(question, question, "en", "rewrite", [question])

        def route(self, query, intent_hint=None):
            return Route("TACTICAL", query, "few_shot", rag_query=query)

        def retrieval_queries(self, route, translation):
            return [route.rag_query]

        def run_rag(self, queries):
            self.seen.append(queries)
            found = (passages_by_query or {}).get(queries[0], [])
            return RagAnswer(queries[0], "dense", False, list(found))

    return Stub(), Passage


def test_a_retrieval_arm_scores_every_labelled_item(config):
    suite = load_test_set(config.path("eval.test_set"))
    items = retrieval_items(suite)[:3]
    stub, _ = _stub_pipeline()

    result = score_retrieval_arm(config, items, [1, 3, 5], pipeline=stub)
    assert result["items"] == 3 and result["errors"] == 0
    # Three timed retrievals, plus one warm-up so the first timing does not also
    # measure loading the embedding model and the cross-encoder.
    assert len(stub.seen) == 4 and stub.seen[0] == stub.seen[1]
    # Nothing retrieved, so recall is a measured zero rather than a missing value.
    assert result["metrics"]["retrieval_recall_at_5"] == {"mean": 0.0, "n": 3}
    # The rerank axis asks whether a cross-encoder earns its latency, so an arm
    # has to carry latency. Retrieval is local compute the LLM cache never
    # serves, which is what makes the number comparable between arms.
    assert result["retrieval_ms"]["n"] == 3
    assert result["retrieval_ms"]["p50"] is not None
    assert result["retrieval_ms"]["p95"] >= result["retrieval_ms"]["p50"]


def test_a_route_that_asks_for_no_search_scores_zero_not_nothing(config):
    """A translation arm that routes away from RAG has lost recall, not skipped it."""
    from tactistat.router import Route

    suite = load_test_set(config.path("eval.test_set"))
    items = retrieval_items(suite)[:2]
    stub, _ = _stub_pipeline()
    stub.route = lambda query, intent_hint=None: Route("STAT", query, "few_shot", {}, None)
    stub.retrieval_queries = lambda route, translation: []

    result = score_retrieval_arm(config, items, [1], pipeline=stub)
    assert stub.seen == []  # the retriever was never called
    assert result["metrics"]["retrieval_recall_at_1"] == {"mean": 0.0, "n": 2}


def test_run_axis_writes_one_comparable_report_per_arm(config, tmp_path, monkeypatch):
    import tactistat.eval.ablate as ablate

    suite = load_test_set(config.path("eval.test_set"))
    axis = Axis(
        name="toy",
        mode="retrieval",
        question="Does the stub retrieve?",
        arms=(Arm("a", {"rag.retrieval.mode": "dense"}), Arm("b", {"rag.retrieval.mode": "bm25"})),
    )
    monkeypatch.setattr(
        ablate,
        "score_retrieval_arm",
        lambda cfg, items, ks: {
            "items": len(items),
            "metrics": {
                "retrieval_mrr": {
                    "mean": 0.5 if cfg["rag.retrieval.mode"] == "bm25" else 0.2,
                    "n": 1,
                }
            },
        },
    )
    output = tmp_path / "toy.json"
    report = run_axis(config, axis, suite=suite, output_path=output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert [arm["name"] for arm in saved["arms"]] == ["a", "b"]
    # Each arm records the config it actually ran, so a result is attributable.
    assert saved["arms"][0]["config_fingerprint"] != saved["arms"][1]["config_fingerprint"]
    assert saved["duration_ms"] >= 0
    assert Path(report["output_path"]).resolve() == output.resolve()
    # A reader who clones the repository can rebuild any arm's effective config
    # from the artifact alone: the base it ran against plus that arm's diff.
    assert saved["base_config"]["rag"]["retrieval"]["mode"] == config["rag.retrieval.mode"]
    assert saved["status"] == "complete" and saved["arm_errors"] == 0
    assert saved["package_version"]


def test_the_table_orders_arms_by_the_leading_metric():
    report = {
        "question": "Does hybrid retrieval beat dense?",
        "arms": [
            {"name": "dense", "metrics": {"retrieval_mrr": {"mean": 0.216}}},
            {"name": "hybrid", "metrics": {"retrieval_mrr": {"mean": 0.293}}},
        ],
    }
    table = comparison_table(report, metrics=["retrieval_mrr"])
    assert table.index("hybrid") < table.index("dense")
    assert "0.293" in table and "Does hybrid retrieval beat dense?" in table


def test_a_metric_missing_from_an_arm_renders_as_not_available():
    report = {
        "question": "Anything?",
        "arms": [{"name": "only", "metrics": {"retrieval_mrr": {"mean": None}}}],
    }
    assert "n/a" in comparison_table(report, metrics=["retrieval_mrr"])


def test_a_pipeline_arm_reports_ungraded_rows(config, tmp_path, monkeypatch):
    """A judge that hit quota leaves means on a denominator of almost nothing."""
    import tactistat.eval.ablate as ablate
    from tactistat.eval.schema import load_test_set

    suite = load_test_set(config.path("eval.test_set"))
    axis = Axis(name="toy", mode="pipeline", question="Does it grade?", arms=(Arm("only", {}),))

    class StubRunner:
        def __init__(self, cfg):
            pass

        def run(self, _suite, output_path=None):
            return {
                "output_path": "somewhere.json",
                "summary": {
                    "items": 45,
                    "metrics": {"judge_correctness": {"mean": 0.333, "n": 6}},
                    "errors": 0,
                    "judge_errors": 39,
                    "abstentions": 4,
                    "timings_trustworthy": False,
                    "timings_ms": {},
                },
            }

    monkeypatch.setattr(ablate, "EvaluationRunner", StubRunner)
    report = ablate.run_axis(config, axis, suite=suite, output_path=tmp_path / "toy.json")
    arm = report["arms"][0]
    assert arm["judge_errors"] == 39 and arm["abstentions"] == 4
    # The mean survives, but so does the denominator that disqualifies it.
    assert arm["metrics"]["judge_correctness"] == {"mean": 0.333, "n": 6}


def test_one_failed_item_does_not_end_the_arm(config):
    """A single retrieval blowing up must not cost every other item's score."""
    suite = load_test_set(config.path("eval.test_set"))
    items = retrieval_items(suite)[:3]
    stub, _ = _stub_pipeline()
    calls = {"n": 0}
    warm_up_and_first = 2

    def explode(queries):
        calls["n"] += 1
        if calls["n"] == warm_up_and_first + 1:
            raise RuntimeError("index unreadable")
        stub.seen.append(queries)
        from tactistat.rag_tool.retrieve import RagAnswer

        return RagAnswer(queries[0], "dense", False, [])

    stub.run_rag = explode
    result = score_retrieval_arm(config, items, [1], pipeline=stub)

    assert result["errors"] == 1
    failed = [row for row in result["rows"] if row["error"]]
    assert len(failed) == 1 and "index unreadable" in failed[0]["error"]
    # The failure scores nothing rather than zero: an absence of a measurement
    # must not average in as if it were a measured miss.
    assert failed[0]["scores"] == {} and failed[0]["retrieval_ms"] is None
    assert result["metrics"]["retrieval_recall_at_1"] == {"mean": 0.0, "n": 2}


def test_a_failed_arm_leaves_the_rest_of_the_sweep_readable(config, tmp_path, monkeypatch):
    """An axis that loses one arm is still a comparison; one that hides it is not."""
    import tactistat.eval.ablate as ablate

    suite = load_test_set(config.path("eval.test_set"))
    axis = Axis(
        name="toy",
        mode="retrieval",
        question="Does it survive?",
        arms=(
            Arm("good", {"rag.retrieval.mode": "dense"}),
            Arm("bad", {"rag.retrieval.mode": "bm25"}),
        ),
    )

    def scorer(cfg, items, ks):
        if cfg["rag.retrieval.mode"] == "bm25":
            raise RuntimeError("bm25 index missing")
        return {"items": len(items), "metrics": {"retrieval_mrr": {"mean": 0.2, "n": 1}}}

    monkeypatch.setattr(ablate, "score_retrieval_arm", scorer)
    output = tmp_path / "toy.json"
    report = run_axis(config, axis, suite=suite, output_path=output)

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert [arm["name"] for arm in saved["arms"]] == ["good", "bad"]
    assert saved["arms"][0]["error"] is None
    assert "bm25 index missing" in saved["arms"][1]["error"]
    assert saved["status"] == "complete" and saved["arm_errors"] == 1
    assert report["arm_errors"] == 1


def test_a_retrieval_arm_records_which_passages_came_back(config):
    """Recall is recheckable from the artifact, not only from a rerun."""
    suite = load_test_set(config.path("eval.test_set"))
    items = retrieval_items(suite)[:1]
    stub, Passage = _stub_pipeline()
    query = items[0].question
    stub.translate = lambda question, history=None: __import__(
        "tactistat.query_translation", fromlist=["TranslatedQuery"]
    ).TranslatedQuery(question, question, "en", "rewrite", [question])
    found = [
        Passage(
            1,
            "body text",
            "Morocco national football team",
            "2022 FIFA World Cup",
            "u",
            1,
            "team",
            "MAR",
            7,
        )
    ]
    stub.run_rag = lambda queries: __import__(
        "tactistat.rag_tool.retrieve", fromlist=["RagAnswer"]
    ).RagAnswer(queries[0], "dense", True, found)

    row = score_retrieval_arm(config, items, [1], pipeline=stub)["rows"][0]
    assert row["passages"] == [
        {
            "rank": 1,
            "title": "Morocco national football team",
            "heading": "2022 FIFA World Cup",
            "chunk_id": 7,
        }
    ]
    assert query  # the item really was the one scored


def test_the_cli_exits_non_zero_when_a_sweep_loses_an_arm(config, tmp_path, monkeypatch, capsys):
    """A sweep that printed a table and dropped an arm must not exit clean."""
    from argparse import Namespace

    import tactistat.eval.ablate as ablate
    from tactistat.cli import _ablate

    axis = Axis(
        name="rerank",
        mode="retrieval",
        question="Does it survive?",
        arms=(Arm("off", {"rag.rerank.enabled": False}), Arm("on", {"rag.rerank.enabled": True})),
    )
    monkeypatch.setitem(ablate.AXES, "rerank", axis)

    def scorer(cfg, items, ks):
        if cfg["rag.rerank.enabled"]:
            raise RuntimeError("cross-encoder unavailable")
        return {
            "items": len(items),
            "metrics": {"retrieval_mrr": {"mean": 0.216, "n": 30}},
            "retrieval_ms": {"p50": 41.0, "p95": 88.0, "n": 30},
        }

    monkeypatch.setattr(ablate, "score_retrieval_arm", scorer)
    args = Namespace(axis="rerank", output=tmp_path / "rerank.json", metrics=["retrieval_mrr"])

    assert _ablate(config, args) == 1
    printed = capsys.readouterr().out
    assert "cross-encoder unavailable" in printed
    # The surviving arm still reports, latency included.
    assert "p50 41 ms" in printed


def test_a_refused_call_is_not_counted_as_a_weak_strategy(config):
    """A provider that rate-limited the sweep must not read as a strategy finding."""
    from tactistat.query_translation import TranslatedQuery

    suite = load_test_set(config.path("eval.test_set"))
    items = retrieval_items(suite)[:3]
    stub, _ = _stub_pipeline()
    statuses = iter(
        [
            ("translated", None),
            ("degraded", "hyde produced no hypothetical passage"),
            ("failed", "translation failed: ResourceExhausted"),
        ]
    )

    def translate(question, history=None):
        status, note = next(statuses)
        return TranslatedQuery(
            question, question, "en", "hyde", [question], status=status, note=note
        )

    stub.translate = translate
    result = score_retrieval_arm(config, items, [1], pipeline=stub)

    assert result["translation_status_counts"] == {"translated": 1, "degraded": 1, "failed": 1}
    # The lumped count stays for continuity, but it can no longer be the only
    # thing a reader has: two of these are the strategy, one is the weather.
    assert result["translation_degradations"] == 2
    notes = [row["translation_note"] for row in result["rows"]]
    assert "ResourceExhausted" in notes[2]


def test_an_arm_records_what_it_actually_differs_by(config, tmp_path, monkeypatch):
    """Declared overrides miss the index directory an index-rebuilding arm gets."""
    import tactistat.eval.ablate as ablate

    suite = load_test_set(config.path("eval.test_set"))
    axis = Axis(
        name="chunking",
        mode="retrieval",
        question="Section or fixed?",
        arms=(Arm("fixed", {"rag.chunking.strategy": "fixed"}),),
        rebuilds_index=True,
    )
    monkeypatch.setattr(ablate, "build_index", lambda cfg: None, raising=False)
    monkeypatch.setattr("tactistat.rag_tool.index.build_index", lambda cfg: None, raising=False)
    monkeypatch.setattr(
        ablate,
        "score_retrieval_arm",
        lambda cfg, items, ks: {"items": 0, "metrics": {}},
    )

    report = run_axis(config, axis, suite=suite, output_path=tmp_path / "toy.json")
    arm = report["arms"][0]
    assert arm["overrides"] == {"rag.chunking.strategy": "fixed"}
    # The override the axis never declared, and without which a reader would
    # rebuild the arm against the shipped index instead of its own.
    assert arm["effective_overrides"]["rag.chunking.strategy"] == "fixed"
    assert arm["effective_overrides"]["rag.index_dir"].endswith("ablation/fixed")


def test_an_identical_arm_differs_by_nothing():
    """The baseline axis has one arm and it must diff clean against its own base."""
    from tactistat.eval.ablate import config_diff

    base = {"rag": {"retrieval": {"mode": "dense", "top_k": 5}}, "eval": {"seed": 42}}
    assert config_diff(base, base) == {}
    changed = dict(base, rag={"retrieval": {"mode": "hybrid", "top_k": 5}})
    assert config_diff(base, changed) == {"rag.retrieval.mode": "hybrid"}


def test_an_index_rebuilding_arm_records_a_portable_path(config):
    """The index directory lands in the artifact, so it must not be one machine's."""
    from pathlib import Path

    from tactistat.config import PROJECT_ROOT

    arm = Arm("fixed", {"rag.chunking.strategy": "fixed"})
    built = arm_config(config, arm, index_root=config.path("rag.index_dir") / "ablation")

    recorded = built["rag.index_dir"]
    assert not Path(recorded).is_absolute()
    assert recorded.endswith("ablation/fixed")
    # Still resolves to the same directory the sweep would build into.
    assert built.path("rag.index_dir") == PROJECT_ROOT / recorded


def test_an_index_outage_is_not_scored_as_weak_retrieval(config):
    """An outage otherwise reads as a bad retriever that is impressively fast."""
    from tactistat.rag_tool.retrieve import RagAnswer

    suite = load_test_set(config.path("eval.test_set"))
    items = retrieval_items(suite)[:3]
    stub, _ = _stub_pipeline()
    calls = {"n": 0}

    def flaky(queries):
        calls["n"] += 1
        # The pipeline catches tool exceptions and returns this, so the arm sees
        # a value rather than a raise. Only `failed` separates it from a miss.
        if calls["n"] == 3:  # one warm-up, then the first timed item
            return RagAnswer(queries[0], "unknown", False, ok=False, failed=True, note="index gone")
        return RagAnswer(queries[0], "dense", False, [])

    stub.run_rag = flaky
    result = score_retrieval_arm(config, items, [1], pipeline=stub)

    assert result["errors"] == 1
    failed = [row for row in result["rows"] if row["error"]][0]
    assert "index gone" in failed["error"]
    assert failed["scores"] == {} and failed["retrieval_ms"] is None
    # Two items measured, not three scored zero.
    assert result["metrics"]["retrieval_recall_at_1"] == {"mean": 0.0, "n": 2}
    assert result["retrieval_ms"]["n"] == 2


def test_a_query_that_matches_nothing_is_still_a_measured_zero(config):
    """The other half of the same distinction: an empty result set is data."""
    from tactistat.rag_tool.retrieve import RagAnswer

    suite = load_test_set(config.path("eval.test_set"))
    items = retrieval_items(suite)[:2]
    stub, _ = _stub_pipeline()
    stub.run_rag = lambda queries: RagAnswer(
        queries[0], "dense", False, ok=False, note="no passage matched the query"
    )

    result = score_retrieval_arm(config, items, [1], pipeline=stub)
    assert result["errors"] == 0
    assert result["metrics"]["retrieval_recall_at_1"] == {"mean": 0.0, "n": 2}
    assert result["retrieval_ms"]["n"] == 2


def test_a_refused_translation_leaves_the_metric_rather_than_lowering_it(config):
    """A rate-limited translator searches with the raw question and scores ~0."""
    from tactistat.query_translation import TranslatedQuery

    suite = load_test_set(config.path("eval.test_set"))
    items = retrieval_items(suite)[:3]
    stub, _ = _stub_pipeline()
    statuses = iter(["translated", "failed", "translated"])

    def translate(question, history=None):
        status = next(statuses)
        note = "translation failed: ResourceExhausted" if status == "failed" else None
        return TranslatedQuery(
            question, question, "en", "hyde", [question], status=status, note=note
        )

    stub.translate = translate
    result = score_retrieval_arm(config, items, [1], pipeline=stub)

    assert result["errors"] == 1
    refused = [row for row in result["rows"] if row["error"]][0]
    assert "ResourceExhausted" in refused["error"] and refused["scores"] == {}
    # It never reached the retriever, so it cannot have a retrieval time either.
    assert refused["retrieval_ms"] is None and refused["queries"] == []
    assert result["metrics"]["retrieval_recall_at_1"]["n"] == 2


def test_a_degraded_strategy_stays_in_the_mean(config):
    """Producing fewer variants than promised is the strategy's own behaviour."""
    from tactistat.query_translation import TranslatedQuery

    suite = load_test_set(config.path("eval.test_set"))
    items = retrieval_items(suite)[:2]
    stub, _ = _stub_pipeline()
    stub.translate = lambda question, history=None: TranslatedQuery(
        question,
        question,
        "en",
        "multi_query",
        [question],
        status="degraded",
        note="multi_query produced 1 of 3 alternative queries",
    )

    result = score_retrieval_arm(config, items, [1], pipeline=stub)
    assert result["errors"] == 0
    assert result["metrics"]["retrieval_recall_at_1"]["n"] == 2
    assert result["translation_status_counts"] == {"degraded": 2}


def test_a_sweep_that_lost_every_arm_renders_the_reasons(config):
    """names[0] on an empty metric set would crash on the way to reporting it."""
    report = {
        "question": "Does anything work?",
        "arms": [
            {"name": "off", "metrics": {}, "error": "RuntimeError: index gone"},
            {"name": "on", "metrics": {}, "error": None},
        ],
    }
    table = comparison_table(report)
    assert "no arm produced a metric" in table
    assert "index gone" in table and "`on`" in table


def test_the_cli_survives_a_sweep_where_every_item_failed(config, tmp_path, monkeypatch, capsys):
    """A totally failed arm has a latency key with nothing under it."""
    from argparse import Namespace

    import tactistat.eval.ablate as ablate
    from tactistat.cli import _ablate

    axis = Axis(
        name="rerank",
        mode="retrieval",
        question="Anything at all?",
        arms=(Arm("off", {"rag.rerank.enabled": False}),),
    )
    monkeypatch.setitem(ablate.AXES, "rerank", axis)
    monkeypatch.setattr(
        ablate,
        "score_retrieval_arm",
        lambda cfg, items, ks: {
            "items": 30,
            "metrics": {},
            "errors": 30,
            "retrieval_ms": {"p50": None, "p95": None, "n": 0},
            "rows": [],
        },
    )
    args = Namespace(axis="rerank", output=tmp_path / "rerank.json", metrics=None)

    assert _ablate(config, args) == 1
    printed = capsys.readouterr().out
    assert "no arm produced a metric" in printed
    assert "30 item(s) errored" in printed


def test_the_cli_axis_list_matches_the_registry(config):
    """Two axes shipped documented and unrunnable because this list drifted."""
    from tactistat.cli import _AXIS_NAMES
    from tactistat.eval.ablate import AXES

    assert set(_AXIS_NAMES) == set(AXES)


def test_a_bare_output_filename_joins_the_other_reports(config, tmp_path, monkeypatch):
    """`--output foo.json` promised eval/results/ablations and wrote the repo root."""
    import tactistat.eval.ablate as ablate
    from tactistat.eval.ablate import _destination

    axis = AXES["rerank"]
    assert _destination(config, axis, "foo.json") == (
        config.path("eval.results_dir") / "ablations" / "foo.json"
    )
    # An explicit directory is still honoured, and absolute paths pass through.
    assert _destination(config, axis, "somewhere/foo.json").name == "foo.json"
    assert _destination(config, axis, tmp_path / "x.json") == tmp_path / "x.json"
    assert ablate  # imported for the registry the axis came from
