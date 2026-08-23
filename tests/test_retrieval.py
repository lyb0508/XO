"""Manual retrieval contracts: ingestion, ranking, thresholds, calibration."""

from __future__ import annotations

import pytest

from app.retrieval.embeddings import DeterministicCharacterEmbeddings
from app.retrieval.store import ManualVectorStore, build_default_store
from app.tools.mock_data import MANUAL_SECTIONS

SECTIONS = [
    {
        "source_id": "manual:v2:section-4.2",
        "device_id": "PUMP-003",
        "title": "振动超限处置",
        "content": "振动连续超限时核查轴承联轴器找正和基础紧固状态，停机须负责人批准。",
        "version": "2.0",
    },
    {
        "source_id": "manual:v2:section-3.1",
        "device_id": "PUMP-003",
        "title": "日常巡检",
        "content": "记录振动趋势并与报警阈值比较。",
        "version": "2.0",
    },
    {
        "source_id": "manual:v1:section-9.9",
        "device_id": "PUMP-999",
        "title": "其他设备手册",
        "content": "完全不同的设备内容。",
        "version": "1.0",
    },
]


@pytest.mark.unit
def test_ingestion_requires_complete_metadata() -> None:
    store = ManualVectorStore(DeterministicCharacterEmbeddings())
    with pytest.raises(ValueError, match="source_id"):
        store.ingest([{"title": "t", "content": "c", "version": "v", "device_id": "d"}])
    with pytest.raises(ValueError, match="version"):
        store.ingest([{"source_id": "s", "title": "t", "content": "c", "version": "", "device_id": "d"}])


@pytest.mark.unit
def test_retrieval_respects_device_binding_top_k_and_threshold() -> None:
    store = ManualVectorStore(DeterministicCharacterEmbeddings())
    store.ingest(SECTIONS)
    hits = store.retrieve("振动超限 轴承", device_id="PUMP-003", top_k=2)
    assert len(hits) <= 2
    assert all(hit.device_id == "PUMP-003" for hit in hits)
    assert all(hit.score >= 0.0 for hit in hits)
    # The other device's chunk is never retrievable for this device.
    assert all(hit.doc_id != "manual:v1:section-9.9" for hit in hits)


@pytest.mark.unit
def test_calibration_related_queries_outrank_unrelated_ones() -> None:
    """Threshold discipline: relative ordering on fixed positive/negative samples.

    These samples are calibrated only against the current mock corpus and the
    deterministic hash embedding; changing either requires recalibration.
    """

    store = ManualVectorStore(DeterministicCharacterEmbeddings())
    store.ingest(list(MANUAL_SECTIONS))
    positives = ["振动超限如何处置", "轴承联轴器检查", "日常巡检记录振动"]
    negatives = ["今天午饭吃什么", "天气预报说明天下雨", "随机无关节选内容"]
    positive_scores = [
        store.retrieve(query, device_id="PUMP-003", top_k=1)[0].score
        for query in positives
    ]
    negative_scores = [
        store.retrieve(query, device_id="PUMP-003", top_k=3)[-1].score
        for query in negatives
    ]
    # Complete separability: the weakest positive outranks the strongest negative.
    assert min(positive_scores) > max(negative_scores), (positive_scores, negative_scores)
    threshold = max(negative_scores)
    for query in positives:
        top = store.retrieve(query, device_id="PUMP-003", top_k=1, min_score=threshold)
        assert top, f"positive sample dropped at calibrated threshold: {query}"
    for query in negatives:
        hits = store.retrieve(query, device_id="PUMP-003", top_k=3, min_score=threshold)
        assert not [hit for hit in hits if hit.score >= threshold]


@pytest.mark.unit
def test_identical_queries_return_deterministic_order() -> None:
    store = ManualVectorStore(DeterministicCharacterEmbeddings())
    store.ingest(SECTIONS)
    first = store.retrieve("振动 巡检", device_id="PUMP-003", top_k=3)
    second = store.retrieve("振动 巡检", device_id="PUMP-003", top_k=3)
    assert [(hit.doc_id, hit.score) for hit in first] == [(hit.doc_id, hit.score) for hit in second]
    scores = [hit.score for hit in first]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_retrieval_boundary_contract() -> None:
    store = ManualVectorStore(DeterministicCharacterEmbeddings())
    store.ingest(SECTIONS)
    assert store.retrieve("   ", device_id="PUMP-003") == []
    assert store.retrieve("振动", device_id="PUMP-003", top_k=0) == []
    assert store.retrieve("振动", device_id="PUMP-003", top_k=-1) == []
    assert store.retrieve("振动", device_id="PUMP-003", top_k=99) is not None
    assert len(store.retrieve("振动", device_id="PUMP-003", top_k=99)) <= len(SECTIONS)
    assert store.retrieve("振动", device_id="PUMP-003", min_score=1.1) == []


@pytest.mark.unit
def test_zero_vector_query_returns_nothing() -> None:
    class EmptyEmbeddings(DeterministicCharacterEmbeddings):
        def embed_query(self, text: str) -> list[float]:
            return [0.0] * self.dimensions

    store = ManualVectorStore(EmptyEmbeddings())
    store.ingest(SECTIONS)
    assert store.retrieve("任意查询", device_id="PUMP-003", top_k=3) == []


@pytest.mark.unit
def test_score_ties_break_deterministically_by_doc_id() -> None:
    from langchain_core.embeddings import Embeddings as _Embeddings

    class ConstantEmbeddings(_Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    store = ManualVectorStore(ConstantEmbeddings())
    store.ingest(list(reversed(SECTIONS)))
    hits = store.retrieve("任何内容", device_id="PUMP-003", top_k=5)
    doc_ids = [hit.doc_id for hit in hits]
    assert doc_ids == sorted(doc_ids)


@pytest.mark.unit
def test_default_store_ingests_the_mock_manual() -> None:
    store = build_default_store(DeterministicCharacterEmbeddings())
    assert len(store) == len(MANUAL_SECTIONS)
    hits = store.retrieve("振动超限处置", device_id="PUMP-003", top_k=1)
    assert hits and hits[0].section and hits[0].version


@pytest.mark.unit
def test_embeddings_factory_validates_provider() -> None:
    from app.retrieval.embeddings import create_embeddings

    with pytest.raises(ValueError, match="Unsupported embeddings provider"):
        create_embeddings("nope", model="m", base_url="http://127.0.0.1:11434")
    assert create_embeddings("deterministic", model="", base_url="")
