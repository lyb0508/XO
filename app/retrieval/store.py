"""In-memory vector store and manual ingestion with source metadata.

The store keeps everything in process memory: no external vector database is
involved at this milestone. Chunks always carry their document id, section,
version, and device binding so any retrieved text can be cited in a report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from langchain_core.embeddings import Embeddings


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieval hit with its complete citation metadata."""

    doc_id: str
    section: str
    title: str
    content: str
    version: str
    device_id: str
    score: float


class ManualVectorStore:
    """Minimal cosine-similarity store over ingested manual chunks."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._chunks: list[dict[str, str]] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def ingest(self, sections: list[Mapping[str, Any]]) -> int:
        """Ingest manual sections; each must bind to one device and version."""

        new_chunks = []
        for section in sections:
            doc_id = _required(section.get("source_id"), "source_id")
            title = _required(section.get("title"), "title")
            content = _required(section.get("content"), "content")
            version = _required(section.get("version"), "version")
            device_id = _required(section.get("device_id"), "device_id")
            new_chunks.append(
                {
                    "doc_id": doc_id,
                    "section": doc_id.rsplit(":", 1)[-1],
                    "title": title,
                    "content": content,
                    "version": version,
                    "device_id": device_id,
                }
            )
        self._chunks.extend(new_chunks)
        self._vectors.extend(
            self._embeddings.embed_documents(
                [f"{chunk['title']} {chunk['content']}" for chunk in new_chunks]
            )
        )
        return len(new_chunks)

    def retrieve(
        self,
        query: str,
        *,
        device_id: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return at most ``top_k`` chunks for the device scoring >= min_score.

        Results are sorted by descending score with the document id as a
        deterministic tiebreaker so identical inputs yield identical order.
        """

        if not query.strip() or top_k < 1:
            return []
        query_vector = self._embeddings.embed_query(query)
        scored = [
            (_cosine(query_vector, vector), chunk)
            for vector, chunk in zip(self._vectors, self._chunks)
            if chunk["device_id"] == device_id
        ]
        hits = [
            RetrievedChunk(
                doc_id=chunk["doc_id"],
                section=chunk["section"],
                title=chunk["title"],
                content=chunk["content"],
                version=chunk["version"],
                device_id=chunk["device_id"],
                score=round(score, 6),
            )
            for score, chunk in scored
            if score >= min_score
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return hits[:top_k]

    def snapshot(self) -> Mapping[str, Any]:
        """Read-only view for diagnostics and tests."""

        return MappingProxyType({"chunks": tuple(self._chunks), "count": len(self._chunks)})


def _required(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manual section field {field} must be a non-empty string")
    return value.strip()


def build_default_store(embeddings: Embeddings) -> ManualVectorStore:
    """Ingest the fixed mock manual into a fresh store."""

    from app.tools.mock_data import MANUAL_SECTIONS

    store = ManualVectorStore(embeddings)
    store.ingest(list(MANUAL_SECTIONS))
    return store
