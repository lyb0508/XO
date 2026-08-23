"""Phase-four retrieval: manual RAG with citation metadata."""

from app.retrieval.embeddings import DeterministicCharacterEmbeddings, create_embeddings
from app.retrieval.store import ManualVectorStore, RetrievedChunk, build_default_store

__all__ = [
    "DeterministicCharacterEmbeddings",
    "ManualVectorStore",
    "RetrievedChunk",
    "build_default_store",
    "create_embeddings",
]
