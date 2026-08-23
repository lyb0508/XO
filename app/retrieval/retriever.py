"""Retrieval entry point wiring settings, store, and citation metadata."""

from __future__ import annotations

from app.config.settings import Settings
from app.retrieval.store import ManualVectorStore, RetrievedChunk, build_default_store


def create_manual_store(settings: Settings) -> ManualVectorStore:
    """Build the configured manual store; construction performs no network I/O."""

    embeddings = _create_embeddings(settings)
    return build_default_store(embeddings)


def retrieve_manual_citations(
    store: ManualVectorStore,
    query: str,
    *,
    device_id: str,
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    """Retrieve manual chunks for one query under the given thresholds.

    Callers pass the calibrated values explicitly (usually from Settings) so
    retrieval behaviour stays visible at the call site.
    """

    return store.retrieve(query, device_id=device_id, top_k=top_k, min_score=min_score)


def _create_embeddings(settings: Settings):
    from app.retrieval.embeddings import create_embeddings

    return create_embeddings(
        settings.embeddings_provider,
        model=settings.embeddings_model,
        base_url=str(settings.ollama_base_url),
    )
