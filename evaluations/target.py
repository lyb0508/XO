"""Evaluation target: run one diagnosis and normalize observable outcomes.

The target is the only place that knows how to drive the graph; evaluators
stay pure functions over normalized outputs so they can be tested offline.
"""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings


def make_live_target():
    """Build a target that runs the real graph against the configured model.

    A crashed case still returns a comparable error payload so evaluators and
    the summary can count it as an explicit failure instead of a silent gap.
    """

    settings = get_settings()

    def live_target(inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            return _run_once(inputs, settings)
        except Exception as error:
            return {
                "scope_status": None,
                "plan_evidence_types": [],
                "tool_source_types": [],
                "report_valid": False,
                "risk_level": None,
                "evidence_sufficient": None,
                "evidence_count": 0,
                "requires_human_review": None,
                "limitations_count": 0,
                "recommended_actions": [],
                "approval_decision": None,
                "error": f"{error.__class__.__name__}: {error}",
            }

    return live_target


def _run_once(inputs: dict[str, Any], settings: Any) -> dict[str, Any]:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from app.graphs.builder import GRAPH_RECURSION_LIMIT, build_diagnosis_graph
    from app.memory.ledger import LongTermLedger
    from app.models.factory import create_chat_model
    from app.retrieval.retriever import create_manual_store

    model = create_chat_model(settings)
    graph = build_diagnosis_graph(
        model,
        structured_output_method=settings.structured_output_method,
        checkpointer=InMemorySaver(),
        manual_store=create_manual_store(settings),
        manual_top_k=settings.manual_retrieval_top_k,
        manual_min_score=settings.manual_retrieval_min_score,
        # Evaluation must never touch the production long-term memory: the
        # ledger goes to a throwaway file that dies with this process.
        ledger=LongTermLedger(_throwaway_ledger_path()),
    )
    config = {
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "configurable": {"thread_id": f"eval-{inputs.get('case_id', 'unknown')}"},
    }
    graph_input = {
        "request_id": f"eval-{inputs.get('case_id', 'unknown')}",
        "device_id": inputs["device_id"],
        "question": inputs["question"],
    }
    result = graph.invoke(graph_input, config=config)
    for _ in range(3):
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if not interrupts:
            break
        plan_scope = (result.get("query_plan") or {}).get("scope_status")
        if plan_scope == "in_scope":
            resume_value = {
                "decision": "approved",
                "decided_by": "eval-runner",
                "reason": "automatic approval during evaluation",
            }
        else:
            # A non-in-scope report must never receive an approved action,
            # even from the evaluation harness.
            resume_value = {
                "decision": "rejected",
                "decided_by": "eval-runner",
                "reason": f"evaluation auto-reject for scope={plan_scope}",
            }
        result = graph.invoke(Command(resume=resume_value), config=config)
    return _normalize(result)


def _throwaway_ledger_path() -> str:
    """One-shot ledger location inside the ignored tmp directory."""

    import tempfile
    from pathlib import Path

    handle = Path(tempfile.gettempdir()) / "industrial-agent-eval-ledgers"
    handle.mkdir(parents=True, exist_ok=True)
    return str(handle / f"eval-{datetime_now_stamp()}.jsonl")


def datetime_now_stamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _normalize(result: dict[str, Any]) -> dict[str, Any]:
    """Project graph state onto the small surface evaluators may inspect."""

    plan = result.get("query_plan") or {}
    payloads = result.get("tool_payloads") or []
    source_types = sorted({str(item.get("source_type")) for item in payloads if item.get("source_type")})
    report = result.get("report")
    return {
        "scope_status": plan.get("scope_status"),
        "plan_evidence_types": sorted(plan.get("requested_evidence_types", [])),
        "tool_source_types": source_types,
        "report_valid": isinstance(report, dict) and bool(report),
        "risk_level": (report or {}).get("risk_level"),
        "evidence_sufficient": (report or {}).get("evidence_sufficient"),
        "evidence_count": len((report or {}).get("evidence", []) or []),
        "requires_human_review": (report or {}).get("requires_human_review"),
        "limitations_count": len((report or {}).get("limitations", []) or []),
        "recommended_actions": (report or {}).get("recommended_actions", []),
        "approval_decision": (result.get("approval") or {}).get("decision"),
        "error": result.get("error"),
    }


def make_offline_target(plan_by_scenario: dict[str, dict[str, Any]]):
    """Deterministic pipeline-smoke target; currently unused by tests but kept
    as the documented seam for offline evaluation runs."""

    def offline_target(inputs: dict[str, Any]) -> dict[str, Any]:
        plan = plan_by_scenario.get(inputs.get("scenario", ""), {"requested_evidence_types": []})
        return {
            "scope_status": plan.get("scope_status", "in_scope"),
            "plan_evidence_types": sorted(plan.get("requested_evidence_types", [])),
            "tool_source_types": [],
            "report_valid": True,
            "risk_level": "medium",
            "evidence_sufficient": True,
            "evidence_count": 1,
            "requires_human_review": False,
            "limitations_count": 0,
            "recommended_actions": [],
            "approval_decision": None,
            "error": None,
        }

    return offline_target
