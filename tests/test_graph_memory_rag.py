"""Graph-level phase-four integration: RAG citations, ledger, memory context."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from conftest import GraphFakeChatModel

from app.graphs.builder import GRAPH_RECURSION_LIMIT, build_diagnosis_graph
from app.memory.ledger import LongTermLedger
from app.memory.session import SessionMemory
from app.retrieval.embeddings import DeterministicCharacterEmbeddings
from app.retrieval.store import build_default_store
from app.schemas.query_plan import QueryPlan
from test_graph_flow import _sufficient_draft

MANUAL_DOC_ID = "manual:circulation-pump-v2:section-4.2"
SENSOR_AND_MANUAL_PLAN = {
    "scope_status": "in_scope",
    "reason": "振动研判并检索手册处置",
    "device_id": "PUMP-003",
    "start_at": "2026-08-22T00:00:00+00:00",
    "end_at": "2026-08-22T02:00:00+00:00",
    "metrics": ["vibration_mm_s"],
    "manual_query": "振动超限 轴承 处置",
    "requested_evidence_types": ["sensor", "manual"],
}


def _draft_with_manual(request_id: str = "request-approval-001") -> dict[str, Any]:
    draft = _sufficient_draft(request_id, review=True)
    ids = [*draft["evidence_ids"], MANUAL_DOC_ID]
    draft["evidence_ids"] = ids
    draft["likely_causes"][0]["evidence_ids"] = [*draft["likely_causes"][0]["evidence_ids"], MANUAL_DOC_ID]
    return draft


def _build(model, **extra: Any):
    return build_diagnosis_graph(
        model,
        structured_output_method="json_schema",
        checkpointer=InMemorySaver(),
        manual_store=build_default_store(DeterministicCharacterEmbeddings()),
        **extra,
    )


CONFIG = {"configurable": {"thread_id": "rag-thread-1"}, "recursion_limit": GRAPH_RECURSION_LIMIT}


@pytest.mark.unit
def test_manual_retrieval_produces_cited_evidence_in_report() -> None:
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(SENSOR_AND_MANUAL_PLAN)],
        draft_responses=[_draft_with_manual()],
    )
    graph = _build(model)
    result = graph.invoke(
        {
            "request_id": "request-approval-001",
            "device_id": "PUMP-003",
            "question": "研判振动并检索手册。",
        },
        config=CONFIG,
    )
    payloads = result["tool_payloads"]
    rag_payload = [item for item in payloads if item.get("source_type") == "rag_manual_store"]
    assert rag_payload and rag_payload[0]["results"]
    manual_evidence = [
        item for item in result["report"]["evidence"] if item["evidence_type"] == "manual"
    ]
    assert any(item["evidence_id"] == MANUAL_DOC_ID for item in manual_evidence)
    cited = next(item for item in manual_evidence if item["evidence_id"] == MANUAL_DOC_ID)
    assert cited["version"] == "2.0"


@pytest.mark.unit
def test_strict_threshold_drops_all_manual_hits_without_failing() -> None:
    # With every manual hit filtered out, the draft must not cite manual
    # evidence at all; sensor+device evidence still produces a valid report.
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(SENSOR_AND_MANUAL_PLAN)],
        draft_responses=[_sufficient_draft("request-approval-001")],
    )
    graph = build_diagnosis_graph(
        model,
        structured_output_method="json_schema",
        checkpointer=InMemorySaver(),
        manual_store=build_default_store(DeterministicCharacterEmbeddings()),
        manual_min_score=0.99,
    )
    result = graph.invoke(
        {
            "request_id": "request-approval-001",
            "device_id": "PUMP-003",
            "question": "研判振动并检索手册。",
        },
        config=CONFIG,
    )
    assert result["report"] is not None
    manual_payloads = [
        item
        for item in result["tool_payloads"]
        if item.get("source_type") == "rag_manual_store"
    ]
    assert all(not item["results"] for item in manual_payloads)
    manual_evidence = [
        item for item in result["report"]["evidence"] if item["evidence_type"] == "manual"
    ]
    assert manual_evidence == []


@pytest.mark.unit
def test_approved_action_is_recorded_into_ledger_and_rejection_is_not(tmp_path) -> None:
    ledger = LongTermLedger(tmp_path / "ledger.jsonl")

    def run(decision: str, thread: str) -> dict[str, Any]:
        model = GraphFakeChatModel(
            plan_responses=[QueryPlan.model_validate(SENSOR_AND_MANUAL_PLAN)],
            draft_responses=[_draft_with_manual(f"req-{thread}")],
        )
        graph = _build(model, ledger=ledger)
        config = {"configurable": {"thread_id": thread}, "recursion_limit": GRAPH_RECURSION_LIMIT}
        paused = graph.invoke(
            {"request_id": f"req-{thread}", "device_id": "PUMP-003", "question": "q"},
            config=config,
        )
        assert "__interrupt__" in paused
        return graph.invoke(Command(resume=_decision(decision)), config=config)

    def _decision(value: str) -> dict[str, Any]:
        return {"decision": value, "decided_by": "officer", "reason": "复核通过" if value == "approved" else "不批准"}

    approved = run("approved", "thread-ledger-a")
    assert approved["action_audit"]["status"] == "executed"
    rejected = run("rejected", "thread-ledger-b")
    assert rejected["action_audit"]["status"] == "rejected"

    history = ledger.history_for_device("PUMP-003")
    assert len(history) == 1
    assert history[0]["request_id"] == "req-thread-ledger-a"
    assert history[0]["ticket_id"] == "MNT-req-thread-ledger-a"
    assert all(record["risk_level"] in {"medium", "high"} for record in history)


@pytest.mark.unit
def test_planner_receives_untrusted_memory_context_when_available(tmp_path) -> None:
    captured_messages: list[Any] = []

    class RecordingPlanner:
        def invoke(self, messages: list[Any], config: Any = None) -> QueryPlan:
            captured_messages.append(list(messages))
            return QueryPlan.model_validate(SENSOR_AND_MANUAL_PLAN)

    session_memory = SessionMemory(max_turns=3)
    session_memory.append_turn(
        "sess-1",
        question="上一轮问了什么",
        device_id="PUMP-003",
        risk_level="high",
        summary="上一轮结论摘要",
    )
    ledger = LongTermLedger(tmp_path / "history.jsonl")
    state = {
        "request_id": "r1",
        "device_id": "PUMP-003",
        "question": "当前问题",
        "session_id": "sess-1",
    }
    from app.graphs.nodes import make_plan_queries

    node = make_plan_queries(RecordingPlanner(), session_memory=session_memory, ledger=ledger)
    node(state)
    rendered = str(captured_messages[0])
    assert "上一轮结论摘要" in rendered and "untrusted reference context" in rendered

    node_without_memory = make_plan_queries(RecordingPlanner())
    node_without_memory({"request_id": "r1", "device_id": "PUMP-003", "question": "当前问题"})
    assert "untrusted reference context" not in str(captured_messages[1])


def _decision(value: str) -> dict[str, Any]:
    return {"decision": value, "decided_by": "officer", "reason": "复核"}
