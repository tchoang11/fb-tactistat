"""Tests for chunking, index identity, and retrieval."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, ValidationError

from tactistat.config import load_config
from tactistat.data.statsbomb import DataError
from tactistat.data.wikipedia import content_hash, load_corpus
from tactistat.rag_tool import chunking as chunking_module
from tactistat.rag_tool.chunking import chunk_pages, index_manifest
from tactistat.rag_tool.index import build_index
from tactistat.rag_tool.retrieve import (
    MAX_TOP_K,
    MODES,
    RagAnswer,
    RagRetriever,
    bm25_tokens,
)

EMBEDDING_WINDOW = 512


@pytest.fixture(scope="module")
def config():
    return load_config(load_env=False)


@pytest.fixture(scope="module")
def pages(config):
    try:
        return load_corpus(config)
    except (FileNotFoundError, DataError) as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def local_tokenizer():
    """Build a small tokenizer without downloading a model."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        model_max_length=EMBEDDING_WINDOW,
    )


@pytest.fixture
def tokenizer(monkeypatch, local_tokenizer):
    """Keep unit tests independent of Hugging Face and the network."""
    monkeypatch.setattr(chunking_module, "_tokenizer", lambda _model: local_tokenizer)
    chunking_module._splitter.cache_clear()
    yield local_tokenizer
    chunking_module._splitter.cache_clear()


_CHUNK_CACHE: dict[tuple, list] = {}


def _chunks(strategy: str, pages, max_tokens: int = 512, overlap: int = 0):
    """Chunk once per distinct setting; several tests share each result."""
    key = (strategy, max_tokens, overlap, len(pages))
    if key not in _CHUNK_CACHE:
        config = load_config(
            load_env=False,
            overrides=[
                f"rag.chunking.strategy={strategy}",
                f"rag.chunking.max_tokens={max_tokens}",
                f"rag.chunking.overlap_tokens={overlap}",
            ],
        )
        _CHUNK_CACHE[key] = chunk_pages(config, pages)
    return _CHUNK_CACHE[key]


@pytest.mark.parametrize(
    "strategy,max_tokens",
    [("section", 512), ("section", 800), ("fixed", 300), ("fixed", 500)],
)
def test_no_chunk_exceeds_the_model_window(pages, tokenizer, strategy, max_tokens):
    documents = _chunks(strategy, pages, max_tokens)
    budget = min(max_tokens, EMBEDDING_WINDOW)
    encoded = tokenizer([document.page_content for document in documents])["input_ids"]
    over = [
        document.metadata["chunk_id"]
        for document, ids in zip(documents, encoded, strict=True)
        if len(ids) > budget
    ]
    assert not over, f"{len(over)} chunks exceed the budget: {over[:5]}"


def test_section_strategy_keeps_section_identity(pages, tokenizer):
    documents = _chunks("section", pages[:40])
    assert all(document.metadata["heading"] for document in documents)
    headings = {(d.metadata["title"], d.metadata["heading"]) for d in documents}
    expected = {(p.title, s.heading) for p in pages[:40] for s in p.sections}
    assert headings == expected, "section chunking must not invent or drop a section"


def test_fixed_strategy_ignores_section_boundaries(pages, tokenizer):
    documents = _chunks("fixed", pages[:40], max_tokens=300)
    assert all(document.metadata["heading"] is None for document in documents)
    section_chunks = _chunks("section", pages[:40])
    assert len(documents) != len(section_chunks)


def test_long_sections_are_split_not_dropped(pages, tokenizer):
    multi = [d for d in _chunks("section", pages) if d.metadata["n_parts"] > 1]
    assert multi, "expected at least one section too long for the window"
    assert {d.metadata["part"] for d in multi} >= {0, 1}


def test_every_chunk_carries_citation_metadata(pages, tokenizer):
    for document in _chunks("section", pages[:40]):
        meta = document.metadata
        assert meta["title"]
        assert meta["url"].startswith("https://")
        assert meta["revision_id"] > 0
        assert meta["entity_type"] in {"team", "player", "tournament", "concept"}


def test_chunk_ids_are_unique(pages, tokenizer):
    documents = _chunks("section", pages)
    ids = [document.metadata["chunk_id"] for document in documents]
    assert len(set(ids)) == len(ids)


def test_chunk_text_carries_its_page_title(pages, tokenizer):
    for document in _chunks("section", pages[:20]):
        assert document.page_content.startswith(document.metadata["title"])


def test_an_unknown_strategy_is_refused(pages, tokenizer):
    with pytest.raises(ValueError, match="chunking strategy"):
        _chunks("semantic", pages[:2])


def test_a_prefix_larger_than_the_budget_is_refused(pages, tokenizer):
    with pytest.raises(ValueError, match="prefix needs"):
        _chunks("section", pages[:1], max_tokens=1)


def test_a_negative_overlap_is_refused(pages, tokenizer):
    with pytest.raises(ValueError, match="overlap_tokens"):
        _chunks("section", pages[:1], overlap=-1)


def test_corpus_content_hash_tracks_revisions(pages):
    assert content_hash(pages) == content_hash(list(reversed(pages))), "order must not matter"

    edited = list(pages)
    edited[0] = type(edited[0])(**{**edited[0].__dict__, "revision_id": edited[0].revision_id + 1})
    assert content_hash(edited) != content_hash(pages)


def test_index_manifest_depends_on_corpus_content(config, monkeypatch):
    before = index_manifest(config)
    monkeypatch.setattr(
        "tactistat.rag_tool.chunking.corpus_content_hash", lambda _config: "different"
    )
    assert index_manifest(config) != before


def test_an_interrupted_rebuild_leaves_no_valid_manifest(tmp_path, monkeypatch, pages, tokenizer):
    import os

    from langchain_core.embeddings import Embeddings

    from tactistat.rag_tool import index as index_module

    class Stub(Embeddings):
        def embed_documents(self, texts):
            return [[float(len(text) % 7), 1.0] for text in texts]

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    config = load_config(load_env=False, overrides=[f"rag.index_dir={tmp_path}"])
    monkeypatch.setattr(index_module, "chunk_corpus", lambda c: chunk_pages(c, pages[:2]))
    monkeypatch.setattr(index_module, "embeddings", lambda c: Stub())

    build_index(config)
    manifest_path = tmp_path / index_module.INDEX_MANIFEST_FILE
    assert manifest_path.exists()

    real_replace = os.replace

    def fail_on_faiss_swap(src, dst):
        if str(dst).endswith(index_module.FAISS_DIR):
            raise RuntimeError("interrupted")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_on_faiss_swap)
    with pytest.raises(RuntimeError, match="interrupted"):
        build_index(config, force=True)

    assert not manifest_path.exists(), "a stale manifest would validate a half-written index"
    assert not (tmp_path / index_module.STAGING_DIR).exists(), "staging must be cleaned up"


class TestIndexManifest:
    def _manifest(self, **overrides):
        return index_manifest(
            load_config(load_env=False, overrides=[f"{k}={v}" for k, v in overrides.items()])
        )

    def test_identical_settings_agree(self):
        assert self._manifest() == self._manifest()

    @pytest.mark.parametrize(
        "key,value",
        [
            ("rag.chunking.strategy", "fixed"),
            ("rag.chunking.max_tokens", 300),
            ("rag.chunking.overlap_tokens", 50),
            ("rag.embedding.model", "BAAI/bge-base-en-v1.5"),
            ("rag.embedding.normalize", "false"),
            ("wikipedia.min_appearances", 6),
        ],
    )
    def test_any_input_change_invalidates_the_index(self, key, value):
        assert self._manifest(**{key: value}) != self._manifest()


class StubRetriever(BaseRetriever):
    """Return deterministic documents at the requested depth."""

    search_kwargs: dict[str, int] = Field(default_factory=lambda: {"k": 5})

    def _get_relevant_documents(self, query, *, run_manager):
        return [
            Document(
                page_content=f"{query} — passage {index}",
                metadata={
                    "title": f"Result {index}",
                    "heading": "Test",
                    "url": f"https://example.test/{index}",
                    "revision_id": index + 1,
                    "entity_type": "concept",
                    "entity_key": None,
                    "chunk_id": index,
                },
            )
            for index in range(self.search_kwargs["k"])
        ]


@pytest.fixture
def retriever(config):
    return RagRetriever(config, retriever=StubRetriever())


class TestBm25Tokenisation:
    def test_case_folds(self):
        assert bm25_tokens("Lionel Messi") == bm25_tokens("lionel messi")

    def test_accents_fold(self):
        assert bm25_tokens("Mbappé") == bm25_tokens("Mbappe") == ["mbappe"]

    def test_punctuation_is_not_a_term(self):
        assert bm25_tokens("Morocco's defence, 2022!") == ["morocco", "s", "defence", "2022"]

    @pytest.mark.parametrize("pair", [("Lionel Messi", "lionel messi"), ("Mbappé", "mbappe")])
    def test_retrieval_is_invariant_to_how_a_name_is_typed(self, pair):
        from langchain_community.retrievers import BM25Retriever

        documents = [
            Document(page_content="Lionel Messi won the Golden Ball", metadata={"title": "Messi"}),
            Document(
                page_content="Kylian Mbappé won the Golden Boot", metadata={"title": "Mbappe"}
            ),
            Document(
                page_content="Harry Maguire played for England", metadata={"title": "Maguire"}
            ),
        ]
        retriever = BM25Retriever.from_documents(documents, preprocess_func=bm25_tokens)
        titles = [[d.metadata["title"] for d in retriever.invoke(query)] for query in pair]
        assert titles[0] == titles[1]


class TestRetrieval:
    def test_returns_top_k_passages_with_citations(self, config, retriever):
        answer = retriever.search("Who won the 2022 World Cup final?")
        assert answer.ok
        assert len(answer.passages) == config["rag.retrieval.top_k"]
        for passage in answer.passages:
            assert passage.revision_id > 0
            assert passage.url.startswith("https://")

    def test_ranks_are_dense_and_one_based(self, retriever):
        answer = retriever.search("Morocco defence")
        assert [p.rank for p in answer.passages] == list(range(1, len(answer.passages) + 1))

    @pytest.mark.parametrize("k", [2, 5, 10, 20])
    def test_top_k_is_honoured_in_both_directions(self, retriever, k):
        assert len(retriever.search("Lionel Messi", top_k=k).passages) == k

    @pytest.mark.parametrize("k", [0, -3, 2.5, True, MAX_TOP_K + 1])
    def test_a_nonsensical_top_k_is_refused(self, retriever, k):
        answer = retriever.search("Messi", top_k=k)
        assert not answer.ok
        assert "top_k" in answer.note

    def test_reranking_still_sees_more_candidates_than_it_returns(self, config):
        class Compressor:
            top_n = 5

        class CompositeRetriever:
            base_compressor = Compressor()
            base_retriever = StubRetriever()

            def invoke(self, query):
                documents = self.base_retriever.invoke(query)
                return documents[: self.base_compressor.top_n]

        rerank_config = load_config(
            load_env=False,
            overrides=["rag.retrieval.mode=hybrid", "rag.rerank.enabled=true"],
        )
        retriever = RagRetriever(rerank_config, retriever=CompositeRetriever())
        assert retriever._fetch_depth(3) == config["rag.rerank.candidate_k"]
        assert retriever._fetch_depth(50) == 50
        assert len(retriever.search("Morocco defence", top_k=3).passages) == 3

    def test_parallel_calls_do_not_share_depth(self, config):
        class TrackingDict(dict):
            four_was_set = Event()

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if value == 4:
                    self.four_was_set.set()

        class CoordinatedRetriever:
            search_kwargs = TrackingDict(k=5)
            first_started = Event()
            release_first = Event()

            def invoke(self, query):
                if query == "first":
                    self.first_started.set()
                    self.release_first.wait(timeout=2)
                depth = self.search_kwargs["k"]
                return StubRetriever(search_kwargs={"k": depth}).invoke(f"depth={depth}")

        base = CoordinatedRetriever()
        retriever = RagRetriever(config, retriever=base)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(retriever.search, "first", 2)
            assert base.first_started.wait(timeout=1)
            second = pool.submit(retriever.search, "second", 4)
            assert not base.search_kwargs.four_was_set.wait(timeout=0.05)
            base.release_first.set()
            first_answer = first.result(timeout=2)
            second_answer = second.result(timeout=2)

        assert first_answer.passages[0].text.startswith("depth=2")
        assert second_answer.passages[0].text.startswith("depth=4")

    def test_an_empty_query_is_refused_not_searched(self, retriever):
        answer = retriever.search("   ")
        assert not answer.ok
        assert "empty" in answer.note

    def test_context_numbers_passages_for_citation(self, retriever):
        context = retriever.search("Golden Boot winner").to_context(max_chars=120)
        assert "[1]" in context
        assert "rev " in context, "a citation without a revision is not reproducible"

    def test_citations_round_trip(self, retriever):
        answer = retriever.search("Argentina Croatia semi-final")
        citations = answer.citations()
        assert len(citations) == len(answer.passages)
        assert all(citation["url"].startswith("https://") for citation in citations)

    def test_a_refusal_renders_as_a_refusal(self):
        answer = RagAnswer("q", "dense", False, ok=False, note="no passage matched the query")
        assert answer.to_context().startswith("RAG TOOL: no passages")

    def test_an_unknown_mode_is_refused(self, config):
        broken = load_config(load_env=False, overrides=["rag.retrieval.mode=magic"])
        with pytest.raises(ValueError, match="retrieval mode"):
            RagRetriever(broken, retriever=StubRetriever())
        assert "magic" not in MODES

    @pytest.mark.parametrize(
        "override,message",
        [
            ("rag.retrieval.top_k=0", "top_k"),
            ("rag.rerank.candidate_k=0", "candidate_k"),
            ("rag.retrieval.hybrid_alpha=1.5", "hybrid_alpha"),
        ],
    )
    def test_invalid_retrieval_config_is_refused(self, override, message):
        config = load_config(
            load_env=False,
            overrides=["rag.retrieval.mode=hybrid", override],
        )
        with pytest.raises(ValueError, match=message):
            RagRetriever(config, retriever=StubRetriever())


@pytest.fixture
def rag_tool(config, retriever):
    from tactistat.rag_tool.langchain_tool import make_rag_tool

    return make_rag_tool(config, retriever=retriever)


class TestLangChainTool:
    def test_artifact_carries_the_passages(self, rag_tool):
        """Evaluation scores retrieval itself, not the model's retelling."""
        message = rag_tool.invoke(
            {
                "type": "tool_call",
                "id": "t",
                "name": "football_context",
                "args": {"query": "Morocco semi-final run"},
            }
        )
        assert message.artifact.ok
        assert message.artifact.passages
        assert isinstance(message.content, str)
        assert "[1]" in message.content

    def test_top_k_schema_has_safe_bounds(self):
        from tactistat.rag_tool.langchain_tool import FootballContextInput

        with pytest.raises(ValidationError):
            FootballContextInput(query="Morocco", top_k=MAX_TOP_K + 1)


@pytest.mark.integration
class TestRagIntegration:
    """Run against the real index only when explicitly requested."""

    def test_demo_query_retrieves_the_match(self):
        if os.environ.get("TACTISTAT_RUN_RAG_INTEGRATION") != "1":
            pytest.skip("set TACTISTAT_RUN_RAG_INTEGRATION=1 to run model tests")

        config = load_config(
            load_env=False,
            overrides=["rag.retrieval.mode=hybrid", "rag.rerank.enabled=true"],
        )
        answer = RagRetriever(config).search("Morocco Spain round of 16 2022")
        top_text = " ".join(
            f"{passage.title} {passage.text}" for passage in answer.passages[:3]
        ).lower()
        assert answer.ok
        assert "morocco" in top_text and "spain" in top_text
