"""Phase-three approval flow: interrupt, three decisions, idempotency, isolation."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from conftest import GraphFakeChatModel
from test_graph_flow import OFF_TOPIC_PLAN, VIBRATION_PLAN, _sufficient_draft

from app.graphs.builder import GRAPH_RECURSION_LIMIT, build_diagnosis_graph
from app.graphs.nodes import execute_approved_action
from app.schemas.approval import ApprovalDecision
from app.schemas.query_plan import QueryPlan
from app.tools.mock_actions import (
    execute_maintenance_action,
    reset_execution_ledger,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    reset_execution_ledger()
    yield
    reset_execution_ledger()


THREAD_CONFIG = {"configurable": {"thread_id": "approval-thread-1"}, "recursion_limit": GRAPH_RECURSION_LIMIT}


def _build(model: GraphFakeChatModel):
    return build_diagnosis_graph(
        model,
        structured_output_method="json_schema",
        checkpointer=InMemorySaver(),
    )


def _invoke(graph, question: str = "研判 PUMP-003 振动并评估风险。", config: dict | None = None):
    return graph.invoke(
        {
            "request_id": "request-approval-001",
            "device_id": "PUMP-003",
            "question": question,
        },
        config=config or THREAD_CONFIG,
    )


def _decision(decision: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "decision": decision,
        "decided_by": "duty-officer",
        "reason": "现场复核后确认。",
    }
    payload.update(overrides)
    return payload


# --- schema contract -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("decision", ["approved", "modified", "rejected"])
def test_approval_decision_shapes(decision: str) -> None:
    modified = ["修订动作一。"] if decision == "modified" else None
    parsed = ApprovalDecision.model_validate(_decision(decision, modified_actions=modified))
    assert parsed.decision == decision


@pytest.mark.unit
def test_modified_decision_requires_actions_and_others_forbid_them() -> None:
    with pytest.raises(Exception, match="must provide modified_actions"):
        ApprovalDecision.model_validate(_decision("modified"))
    with pytest.raises(Exception, match="only allowed for a modified"):
        ApprovalDecision.model_validate(_decision("approved", modified_actions=["x"]))


@pytest.mark.unit
def test_decided_by_is_restricted_to_safe_characters() -> None:
    with pytest.raises(Exception, match="decided_by"):
        ApprovalDecision.model_validate(_decision("approved", decided_by="bad name!"))


# --- idempotent mock action -------------------------------------------------


@pytest.mark.unit
def test_maintenance_action_is_idempotent_per_request() -> None:
    first = execute_maintenance_action("request-idem-001", "PUMP-003")
    second = execute_maintenance_action("request-idem-001", "PUMP-003")
    assert first["status"] == "executed"
    assert second["status"] == "already_executed"
    assert first["ticket_id"] == second["ticket_id"]
    other = execute_maintenance_action("request-idem-002", "PUMP-003")
    assert other["status"] == "executed" and other["ticket_id"] != first["ticket_id"]


@pytest.mark.unit
def test_execute_node_skips_when_audit_already_exists() -> None:
    state = {
        "request_id": "request-replay-001",
        "device_id": "PUMP-003",
        "approval": {"decision": "approved", "decided_by": "officer"},
        "action_audit": {"status": "executed", "ticket_id": "MNT-existing"},
    }
    assert execute_approved_action(state) == {}
    from app.tools.mock_actions import _EXECUTION_LEDGER

    assert ("schedule_maintenance", "request-replay-001") not in _EXECUTION_LEDGER


@pytest.mark.unit
def test_execute_node_reports_already_executed_on_ledger_hit() -> None:
    execute_maintenance_action("request-replay-002", "PUMP-003")
    state = {
        "request_id": "request-replay-002",
        "device_id": "PUMP-003",
        "approval": {"decision": "approved", "decided_by": "officer"},
    }
    audit = execute_approved_action(state)["action_audit"]
    assert audit["status"] == "already_executed"
    assert audit["ticket_id"] == "MNT-request-replay-002"


# --- graph-level approval flows --------------------------------------------


@pytest.mark.unit
def test_high_risk_report_interrupts_then_approves_and_executes() -> None:
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(VIBRATION_PLAN)],
        draft_responses=[_sufficient_draft("request-approval-001", review=True)],
    )
    graph = _build(model)
    result = _invoke(graph)
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["proposed_action"]["action_type"] == "schedule_maintenance"
    assert payload["proposed_action"]["request_id"] == "request-approval-001"

    resumed = graph.invoke(Command(resume=_decision("approved")), config=THREAD_CONFIG)
    audit = resumed["action_audit"]
    assert audit["status"] == "executed" and audit["ticket_id"] == "MNT-request-approval-001"
    assert resumed["approval"]["decided_by"] == "duty-officer"


@pytest.mark.unit
def test_modified_decision_updates_report_actions_before_execution() -> None:
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(VIBRATION_PLAN)],
        draft_responses=[_sufficient_draft("request-approval-001", review=True)],
    )
    graph = _build(model)
    _invoke(graph)
    revised = _decision("modified", modified_actions=["先复测振动，再由值班负责人决定停机。"])
    resumed = graph.invoke(Command(resume=revised), config=THREAD_CONFIG)
    assert resumed["report"]["recommended_actions"] == ["先复测振动，再由值班负责人决定停机。"]
    assert resumed["action_audit"]["status"] == "executed"


@pytest.mark.unit
def test_rejected_decision_records_audit_without_side_effect() -> None:
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(VIBRATION_PLAN)],
        draft_responses=[_sufficient_draft("request-approval-001", review=True)],
    )
    graph = _build(model)
    _invoke(graph)
    resumed = graph.invoke(Command(resume=_decision("rejected")), config=THREAD_CONFIG)
    audit = resumed["action_audit"]
    assert audit["status"] == "rejected" and audit["ticket_id"] is None
    from app.tools.mock_actions import _EXECUTION_LEDGER

    assert _EXECUTION_LEDGER == {}, "a rejected decision must leave no execution record"


@pytest.mark.unit
def test_no_review_path_completes_without_interrupt() -> None:
    insufficient = {
        "request_id": "request-approval-001",
        "device_id": "PUMP-003",
        "scope_status": "out_of_scope",
        "risk_level": "unknown",
        "summary": "问题超出范围。",
        "evidence_sufficient": False,
        "likely_causes": [],
        "evidence_ids": [],
        "recommended_actions": [],
        "requires_human_review": False,
        "limitations": ["未采集证据。"],
    }
    model = GraphFakeChatModel(plan_responses=[OFF_TOPIC_PLAN], draft_responses=[insufficient])
    graph = _build(model)
    result = _invoke(graph, question="今天天气如何？")
    assert "__interrupt__" not in result
    assert result.get("approval") is None and result.get("action_audit") is None


@pytest.mark.unit
def test_threads_are_isolated_under_a_shared_checkpointer() -> None:
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(VIBRATION_PLAN)],
        draft_responses=[_sufficient_draft("req-shared", review=True)],
    )
    graph = _build(model)
    config_a = {"configurable": {"thread_id": "thread-a"}, "recursion_limit": GRAPH_RECURSION_LIMIT}
    config_b = {"configurable": {"thread_id": "thread-b"}, "recursion_limit": GRAPH_RECURSION_LIMIT}
    paused_a = graph.invoke({"request_id": "req-shared", "device_id": "PUMP-003", "question": "q"}, config=config_a)
    paused_b = graph.invoke({"request_id": "req-shared", "device_id": "PUMP-003", "question": "q"}, config=config_b)
    assert "__interrupt__" in paused_a and "__interrupt__" in paused_b

    approved = graph.invoke(Command(resume=_decision("approved")), config=config_a)
    rejected = graph.invoke(Command(resume=_decision("rejected")), config=config_b)
    assert approved["action_audit"]["status"] == "executed"
    assert approved["report"]["request_id"] == "req-shared"
    assert rejected["action_audit"]["status"] == "rejected"
    assert rejected["report"]["request_id"] == "req-shared"
