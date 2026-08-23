"""Embedding providers for manual retrieval.

Two implementations share the standard LangChain ``Embeddings`` interface:

* ``DeterministicCharacterEmbeddings`` hashes overlapping character bigrams
  into a fixed-width vector. It has no semantic quality at all and exists so
  offline tests can verify ranking, threshold, and metadata contracts without
  a model; it must never be presented as semantic search.
* ``OllamaEmbeddings`` is built by the factory for live use. Building the
  object performs no network I/O; the first embedding call (ingestion or
  query) contacts the configured Ollama endpoint.
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings


class DeterministicCharacterEmbeddings(Embeddings):
    """Stable hash embedding over character bigrams; offline testing only."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32:
            raise ValueError("embedding dimensions must be at least 32")
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        normalized = "".join(text.lower().split())
        vector = [0.0] * self.dimensions
        if not normalized:
            return vector
        for index in range(len(normalized) - 1):
            bigram = normalized[index : index + 2]
            digest = hashlib.sha256(bigram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def create_embeddings(provider: str, *, model: str, base_url: str) -> Embeddings:
    """Build the configured embeddings provider without any network request."""

    from langchain_ollama import OllamaEmbeddings

    if provider == "ollama":
        return OllamaEmbeddings(model=model, base_url=base_url)
    if provider == "deterministic":
        return DeterministicCharacterEmbeddings()
    raise ValueError(f"Unsupported embeddings provider: {provider}")
