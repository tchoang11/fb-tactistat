"""Split corpus pages by section or fixed token windows."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tactistat.artifacts import ARTIFACT_SCHEMA_VERSION, stable_hash
from tactistat.config import Config
from tactistat.data.statsbomb import dataset_identity
from tactistat.data.wikipedia import WikiPage, corpus_content_hash, corpus_manifest, load_corpus

STRATEGIES = ("section", "fixed")


@lru_cache(maxsize=4)
def _tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=256)
def _splitter(model_name: str, chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """Cache token-aware splitters by body budget."""
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _tokenizer(model_name),
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _effective_budget(tokenizer: Any, max_tokens: int) -> int:
    """Cap the requested budget at the embedding model's window."""
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError(f"rag.chunking.max_tokens must be a positive integer, got {max_tokens!r}")

    model_window = getattr(tokenizer, "model_max_length", max_tokens)
    if not isinstance(model_window, int) or model_window < 1 or model_window > 1_000_000:
        model_window = max_tokens
    return min(max_tokens, model_window)


def _split_with_prefix(
    config: Config, prefix: str, text: str, max_tokens: int, overlap: int
) -> list[str]:
    """Split text while counting the title/heading prefix in the budget."""
    model_name = config["rag.embedding.model"]
    tokenizer = _tokenizer(model_name)
    budget = _effective_budget(tokenizer, max_tokens)
    special_tokens = len(tokenizer.encode("", add_special_tokens=True))
    prefix_tokens = len(tokenizer.tokenize(prefix))
    body_tokens = budget - special_tokens - prefix_tokens
    if body_tokens < 1:
        raise ValueError(
            f"Chunk prefix needs {prefix_tokens + special_tokens} tokens, "
            f"exceeding the {budget}-token budget"
        )
    splitter = _splitter(model_name, body_tokens, min(overlap, body_tokens // 2))
    return [f"{prefix}{part}" for part in splitter.split_text(text)]


def _base_metadata(page: WikiPage) -> dict[str, Any]:
    """Return metadata needed for a reproducible citation."""
    return {
        "page_id": page.page_id,
        "title": page.title,
        "url": page.url,
        "revision_id": page.revision_id,
        "entity_type": page.entity_type,
        "entity_key": page.entity_key,
    }


def chunk_pages(config: Config, pages: list[WikiPage]) -> list[Document]:
    strategy = config["rag.chunking.strategy"]
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy {strategy!r}; expected one of {STRATEGIES}")

    max_tokens = config["rag.chunking.max_tokens"]
    overlap = config["rag.chunking.overlap_tokens"]
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError(
            f"rag.chunking.overlap_tokens must be a non-negative integer, got {overlap!r}"
        )
    documents: list[Document] = []

    for page in pages:
        base = _base_metadata(page)

        if strategy == "section":
            # Keep overlap inside each Wikipedia section.
            for section in page.sections:
                prefix = f"{page.title} — {section.heading}\n\n"
                parts = _split_with_prefix(config, prefix, section.text, max_tokens, overlap)
                for index, part in enumerate(parts):
                    documents.append(
                        Document(
                            page_content=part,
                            metadata={
                                **base,
                                "heading": section.heading,
                                "level": section.level,
                                "part": index,
                                "n_parts": len(parts),
                            },
                        )
                    )
        else:
            # Keep headings so the fixed-window baseline sees the same text.
            joined = "\n\n".join(f"== {s.heading} ==\n{s.text}" for s in page.sections)
            parts = _split_with_prefix(config, f"{page.title}\n\n", joined, max_tokens, overlap)
            for index, part in enumerate(parts):
                documents.append(
                    Document(
                        page_content=part,
                        metadata={
                            **base,
                            "heading": None,
                            "level": None,
                            "part": index,
                            "n_parts": len(parts),
                        },
                    )
                )

    for position, document in enumerate(documents):
        document.metadata["chunk_id"] = position
    return documents


def chunk_corpus(config: Config) -> list[Document]:
    return chunk_pages(config, load_corpus(config))


def index_manifest(config: Config) -> dict[str, Any]:
    """Fingerprint every input that changes the index vectors."""
    settings = {
        "corpus": corpus_manifest(config),
        "corpus_content": corpus_content_hash(config),
        "chunking": config.section("rag.chunking"),
        "embedding_model": config["rag.embedding.model"],
        "normalize": config["rag.embedding.normalize"],
    }
    return {
        "artifact": "rag_index",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "dataset": dataset_identity(config),
        "settings_hash": stable_hash(settings),
    }
