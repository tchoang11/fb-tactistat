"""Dense, BM25, and hybrid retrieval with optional reranking."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from tactistat.config import Config
from tactistat.rag_tool.index import load_chunks, load_vectorstore

MODES = ("dense", "bm25", "hybrid")
MAX_TOP_K = 50
MAX_CANDIDATE_K = 200

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def bm25_tokens(text: str) -> list[str]:
    """Fold case and accents before BM25 term matching."""
    folded = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return _WORD_RE.findall(stripped)


@dataclass
class Passage:
    """One retrieved chunk, with everything a citation needs."""

    rank: int
    text: str
    title: str
    heading: str | None
    url: str
    revision_id: int
    entity_type: str
    entity_key: str | None
    chunk_id: int

    @classmethod
    def from_document(cls, rank: int, document: Document) -> Passage:
        meta = document.metadata
        return cls(
            rank=rank,
            text=document.page_content,
            title=meta.get("title", ""),
            heading=meta.get("heading"),
            url=meta.get("url", ""),
            revision_id=int(meta.get("revision_id") or 0),
            entity_type=meta.get("entity_type", ""),
            entity_key=meta.get("entity_key"),
            chunk_id=int(meta.get("chunk_id") or 0),
        )

    def cite(self) -> str:
        where = f"{self.title} — {self.heading}" if self.heading else self.title
        return f"{where} (rev {self.revision_id})"


@dataclass
class RagAnswer:
    query: str
    mode: str
    reranked: bool
    passages: list[Passage] = field(default_factory=list)
    note: str | None = None
    ok: bool = True

    def to_context(self, max_chars: int = 1200) -> str:
        """Render numbered passages for synthesis and citation."""
        if not self.ok or not self.passages:
            return f"RAG TOOL: no passages retrieved. {self.note or ''}".strip()

        header = f"RAG TOOL — {len(self.passages)} passages ({self.mode}"
        header += ", reranked)" if self.reranked else ")"
        lines = [header]
        for passage in self.passages:
            body = passage.text[:max_chars]
            if len(passage.text) > max_chars:
                body += " ..."
            lines.append(f"[{passage.rank}] {passage.cite()}")
            lines.append(f"    {body}")
        return "\n".join(lines)

    def citations(self) -> list[dict[str, Any]]:
        return [
            {
                "rank": passage.rank,
                "title": passage.title,
                "heading": passage.heading,
                "url": passage.url,
                "revision_id": passage.revision_id,
                "chunk_id": passage.chunk_id,
            }
            for passage in self.passages
        ]


def _dense_retriever(config: Config, k: int) -> BaseRetriever:
    return load_vectorstore(config).as_retriever(search_kwargs={"k": k})


def _bm25_retriever(config: Config, k: int) -> BaseRetriever:
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(load_chunks(config), preprocess_func=bm25_tokens)
    retriever.k = k
    return retriever


def _set_depth(retriever: BaseRetriever, fetch_k: int, top_k: int) -> None:
    """Apply per-query depths to the leaf retrievers and reranker."""
    compressor = getattr(retriever, "base_compressor", None)
    if compressor is not None:
        if hasattr(compressor, "top_n"):
            compressor.top_n = top_k
        _set_depth(retriever.base_retriever, fetch_k, top_k)
        return

    members = getattr(retriever, "retrievers", None)
    if members:
        for member in members:
            _set_depth(member, fetch_k, top_k)
        return

    if hasattr(retriever, "search_kwargs"):
        retriever.search_kwargs["k"] = fetch_k
    elif hasattr(retriever, "k"):
        retriever.k = fetch_k


def _reranker(config: Config):
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    return CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=config["rag.rerank.model"]),
        top_n=config["rag.retrieval.top_k"],
    )


def _validate_depth(name: str, value: Any, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}, got {value!r}")
    return value


def _retrieval_settings(config: Config) -> tuple[str, int, int, bool]:
    """Validate retrieval configuration before loading any models."""
    mode = config["rag.retrieval.mode"]
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {MODES}")

    top_k = _validate_depth("rag.retrieval.top_k", config["rag.retrieval.top_k"], MAX_TOP_K)
    candidate_k = _validate_depth(
        "rag.rerank.candidate_k", config["rag.rerank.candidate_k"], MAX_CANDIDATE_K
    )
    rerank = bool(config["rag.rerank.enabled"])
    if rerank and candidate_k < top_k:
        raise ValueError("rag.rerank.candidate_k must be greater than or equal to top_k")

    alpha = config["rag.retrieval.hybrid_alpha"]
    if mode == "hybrid" and (
        not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 <= alpha <= 1
    ):
        raise ValueError(f"rag.retrieval.hybrid_alpha must be between 0 and 1, got {alpha!r}")
    return mode, top_k, candidate_k, rerank


def build_retriever(config: Config) -> BaseRetriever:
    """Assemble the retriever the config describes."""
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever

    mode, top_k, candidate_k, rerank = _retrieval_settings(config)
    fetch_k = candidate_k if rerank else top_k

    if mode == "dense":
        base = _dense_retriever(config, fetch_k)
    elif mode == "bm25":
        base = _bm25_retriever(config, fetch_k)
    else:
        alpha = config["rag.retrieval.hybrid_alpha"]
        base = EnsembleRetriever(
            retrievers=[_dense_retriever(config, fetch_k), _bm25_retriever(config, fetch_k)],
            weights=[alpha, 1.0 - alpha],
        )

    if not rerank:
        return base
    return ContextualCompressionRetriever(base_compressor=_reranker(config), base_retriever=base)


class RagRetriever:
    """Reusable retriever that returns citation-ready answers."""

    def __init__(self, config: Config, retriever: BaseRetriever | None = None):
        self.config = config
        self.mode, self.top_k, self.candidate_k, self.reranked = _retrieval_settings(config)
        self.retriever = build_retriever(config) if retriever is None else retriever
        self._search_lock = Lock()

    def _fetch_depth(self, top_k: int) -> int:
        """Return how many candidates the reranker should inspect."""
        if not self.reranked:
            return top_k
        return max(self.candidate_k, top_k)

    def search(self, query: str, top_k: int | None = None) -> RagAnswer:
        query = (query or "").strip()
        if not query:
            return RagAnswer(query, self.mode, self.reranked, ok=False, note="empty query")

        limit = self.top_k if top_k is None else top_k
        try:
            limit = _validate_depth("top_k", limit, MAX_TOP_K)
        except ValueError as exc:
            return RagAnswer(query, self.mode, self.reranked, ok=False, note=str(exc))

        # Depth is mutable in LangChain retrievers, so keep each call isolated.
        with self._search_lock:
            _set_depth(self.retriever, self._fetch_depth(limit), limit)
            documents = self.retriever.invoke(query)
        passages = [
            Passage.from_document(rank, document)
            for rank, document in enumerate(documents[:limit], start=1)
        ]
        if not passages:
            return RagAnswer(
                query, self.mode, self.reranked, ok=False, note="no passage matched the query"
            )
        return RagAnswer(query, self.mode, self.reranked, passages=passages)
