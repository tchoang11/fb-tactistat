"""Build and load the FAISS index and its shared chunk table."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document

from tactistat.artifacts import (
    manifest_matches,
    read_manifest,
    write_json_atomic,
    write_parquet_atomic,
)
from tactistat.config import Config
from tactistat.data.statsbomb import DataError
from tactistat.rag_tool.chunking import chunk_corpus, index_manifest

CHUNKS_FILE = "chunks.parquet"
INDEX_MANIFEST_FILE = "manifest.json"
FAISS_DIR = "faiss"
STAGING_DIR = ".staging"

# Chunk metadata that must survive the Parquet round trip.
_METADATA_COLUMNS = (
    "chunk_id",
    "page_id",
    "title",
    "url",
    "revision_id",
    "entity_type",
    "entity_key",
    "heading",
    "level",
    "part",
    "n_parts",
)


def index_dir(config: Config) -> Path:
    path = config.path("rag.index_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def embeddings(config: Config):
    """Load the configured sentence-transformer embeddings."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config["rag.embedding.model"],
        encode_kwargs={
            "batch_size": config["rag.embedding.batch_size"],
            "normalize_embeddings": config["rag.embedding.normalize"],
        },
    )


def _documents_to_frame(documents: list[Document]) -> pd.DataFrame:
    rows = [
        {"text": d.page_content, **{key: d.metadata.get(key) for key in _METADATA_COLUMNS}}
        for d in documents
    ]
    return pd.DataFrame(rows)


def _frame_to_documents(frame: pd.DataFrame) -> list[Document]:
    return [
        Document(
            page_content=row["text"],
            metadata={key: row[key] for key in _METADATA_COLUMNS if key in row},
        )
        for _, row in frame.iterrows()
    ]


def build_index(config: Config, force: bool = False) -> dict[str, Any]:
    """Chunk the corpus, embed it, and write the index and chunk table."""
    from langchain_community.vectorstores import FAISS

    out = index_dir(config)
    manifest_path = out / INDEX_MANIFEST_FILE
    expected = index_manifest(config)

    complete = (out / CHUNKS_FILE).exists() and (out / FAISS_DIR).exists()
    if not force and complete and manifest_matches(read_manifest(manifest_path), expected):
        frame = pd.read_parquet(out / CHUNKS_FILE)
        return {"chunks": len(frame), "rebuilt": False, "path": out}

    documents = chunk_corpus(config)
    if not documents:
        raise DataError("Chunking produced no documents; build the Wikipedia corpus first.")

    # Build separately so an interrupted write is never accepted as current.
    staging = out / STAGING_DIR
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        FAISS.from_documents(documents, embeddings(config)).save_local(str(staging / FAISS_DIR))
        write_parquet_atomic(_documents_to_frame(documents), staging / CHUNKS_FILE)

        # The manifest is written last, after both live artifacts are replaced.
        manifest_path.unlink(missing_ok=True)
        shutil.rmtree(out / FAISS_DIR, ignore_errors=True)
        os.replace(staging / FAISS_DIR, out / FAISS_DIR)
        os.replace(staging / CHUNKS_FILE, out / CHUNKS_FILE)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    write_json_atomic(manifest_path, expected)
    return {"chunks": len(documents), "rebuilt": True, "path": out}


def _require_current_index(config: Config) -> Path:
    out = index_dir(config)
    if not (out / CHUNKS_FILE).exists() or not (out / FAISS_DIR).exists():
        raise DataError(f"No index in {out}. Build it first:\n    tactistat build-index")
    if not manifest_matches(read_manifest(out / INDEX_MANIFEST_FILE), index_manifest(config)):
        raise DataError(
            "The index does not match the configured corpus, chunking, or embedding model. "
            "Rebuild it:\n    tactistat build-index --force"
        )
    return out


def load_chunks(config: Config) -> list[Document]:
    return _frame_to_documents(pd.read_parquet(_require_current_index(config) / CHUNKS_FILE))


def load_vectorstore(config: Config):
    from langchain_community.vectorstores import FAISS

    out = _require_current_index(config)
    # Only load the local, gitignored index produced by build_index().
    return FAISS.load_local(
        str(out / FAISS_DIR), embeddings(config), allow_dangerous_deserialization=True
    )
