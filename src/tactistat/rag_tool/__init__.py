"""Retrieval over the Wikipedia corpus: the TACTICAL half of the system.

corpus.jsonl  ->  chunking.py  section or fixed windows, token-budgeted
              ->  index.py     embeddings + FAISS, with the chunk table
              ->  retrieve.py  dense / bm25 / hybrid, optionally reranked
"""

from tactistat.rag_tool.chunking import chunk_corpus, chunk_pages, index_manifest
from tactistat.rag_tool.index import build_index, load_chunks, load_vectorstore
from tactistat.rag_tool.retrieve import MODES, Passage, RagAnswer, RagRetriever, build_retriever

__all__ = [
    "MODES",
    "Passage",
    "RagAnswer",
    "RagRetriever",
    "build_index",
    "build_retriever",
    "chunk_corpus",
    "chunk_pages",
    "index_manifest",
    "load_chunks",
    "load_vectorstore",
]
