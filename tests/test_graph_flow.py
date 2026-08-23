"""End-to-end graph flow tests over a scripted fake model; fully offline."""

from __future__ import annotations

from typing import Any, Literal, Protocol

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from app.graphs.builder import GRAPH_RECURSION_LIMIT, build_diagnosis_graph
from app.schemas.diagnostics import DiagnosisDraft
from app.schemas.query_plan import QueryPlan

WINDOW = {"start_at": "2026-08-22T00:00:00+00:00", "end_at": "2026-08-22T02:00:00+00:00"}
SENSOR_IDS = [f"sensor:PUMP-003:2026-08-22T01:{minute:02d}:00Z" for minute in (0, 10, 20)]
DEVICE_ID = "asset:PUMP-003"

VIBRATION_PLAN = {
    "scope_status": "in_scope",
    "reason": "研判 PUMP-003 振动报警",
    "device_id": "PUMP-003",
    **WINDOW,
    "metrics": ["vibration_mm_s"],
    "manual_query": None,
    "requested_evidence_types": ["sensor"],
}

OFF_TOPIC_PLAN = {
    "scope_status": "out_of_scope",
    "reason": "问题与设备诊断无关",
    "device_id": None,
    "start_at": None,
    "end_at": None,
    "metrics": [],
    "manual_query": None,
    "requested_evidence_types": [],
}

INSUFFICIENT_DRAFT = {
    "request_id": "request-graph-001",
    "device_id": "PUMP-003",
    "scope_status": "out_of_scope",
    "risk_level": "unknown",
    "summary": "问题超出设备研判范围。",
    "evidence_sufficient": False,
    "likely_causes": [],
    "evidence_ids": [],
    "recommended_actions": [],
    "requires_human_review": False,
    "limitations": ["未采集任何设备证据。"],
}


class _Bound(Protocol):
    calls: int


class _FakeBound:
    """Schema-bound invokable returning scripted responses in order."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls = 0
        self.inputs: list[list[BaseMessage]] = []

    def invoke(self, messages: list[Any], config: dict[str, Any] | None = None) -> Any:
        assert config is None or "tags" not in config or isinstance(config["tags"], list)
        if not self._responses:
            raise AssertionError("no scripted response configured")
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        self.inputs.append(list(messages))
        return self._responses[index]


class GraphFakeChatModel(BaseChatModel):
    """Deterministic offline stand-in exposing two schema-bound invokables."""

    plan_responses: list[Any]
    draft_responses: list[Any]
    _planner: _FakeBound | None = PrivateAttr(default=None)
    _formatter: _FakeBound | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "phase-two-graph-fake"

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        raise AssertionError("the diagnosis graph must never call the raw model directly")

    def with_structured_output(self, schema: object, **kwargs: Any) -> _FakeBound:
        target_schema = DiagnosisDraft.__name__ if isinstance(schema, type) else ""
        if schema is QueryPlan:
            self._planner = _FakeBound(list(self.plan_responses))
            return self._planner  # type: ignore[return-value]
        if target_schema == "DiagnosisDraft":
            self._formatter = _FakeBound(list(self.draft_responses))
            return self._formatter  # type: ignore[return-value]
        raise AssertionError(f"unexpected structured schema {schema!r}")


def _sufficient_draft(request_id: str = "request-graph-001") -> dict[str, Any]:
    return {
        "request_id": request_id,
        "device_id": "PUMP-003",
        "scope_status": "in_scope",
        "risk_level": "high",
        "summary": "振动持续超限，需要现场复核。",
        "evidence_sufficient": True,
        "likely_causes": [
            {
                "cause": "轴承或联轴器找正异常。",
                "confidence": 0.7,
                "evidence_ids": [*SENSOR_IDS, DEVICE_ID],
            }
        ],
        "evidence_ids": [*SENSOR_IDS, DEVICE_ID],
        "recommended_actions": ["安排现场复核并记录复测结果。"],
        "requires_human_review": True,
        "limitations": [],
    }


def _run_graph(
    *,
    plans: list[Any],
    drafts: list[Any],
    question: str = "研判 PUMP-003 在窗口内的振动历史并给出风险。",
    request_id: str = "request-graph-001",
):
    model = GraphFakeChatModel(plan_responses=plans, draft_responses=drafts)
    graph = build_diagnosis_graph(model, structured_output_method="json_schema")
    return graph.invoke(
        {"request_id": request_id, "device_id": "PUMP-003", "question": question},
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )


@pytest.mark.unit
def test_vibration_flow_fans_out_formats_and_gates() -> None:
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(VIBRATION_PLAN)],
        draft_responses=[_sufficient_draft()],
    )
    graph = build_diagnosis_graph(model, structured_output_method="json_schema")
    result = graph.invoke(
        {
            "request_id": "request-graph-001",
            "device_id": "PUMP-003",
            "question": "研判 PUMP-003 振动。",
        },
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )
    assert model._planner is not None and model._planner.calls == 1
    assert model._formatter is not None and model._formatter.calls == 1
    payloads = result["tool_payloads"]
    # Sensor request fans out exactly the sensor node plus the forced device threshold lookup.
    source_types = sorted(payload["source_type"] for payload in payloads)
    assert source_types == ["mock_asset_registry", "mock_sensor_store"]
    assert result.get("error", "") == ""
    report = result["report"]
    assert report is not None
    evidence_types = sorted(item["evidence_type"] for item in report["evidence"])
    assert evidence_types == ["device"] + ["sensor"] * len(SENSOR_IDS)
    selected_ids = {item["evidence_id"] for item in report["evidence"]}
    assert set(SENSOR_IDS).issubset(selected_ids) and DEVICE_ID in selected_ids
    assert report["request_id"] == "request-graph-001"
    assert report["risk_level"] == "high" and report["requires_human_review"] is True


@pytest.mark.unit
def test_out_of_scope_plan_never_invokes_tools() -> None:
    result = _run_graph(
        plans=[OFF_TOPIC_PLAN],
        drafts=[INSUFFICIENT_DRAFT],
        question="今天天气怎么样？",
    )
    assert result.get("tool_payloads", []) == []
    assert result.get("registry_entries", []) == []
    report = result["report"]
    assert report["scope_status"] == "out_of_scope"
    assert report["risk_level"] == "unknown" and report["limitations"]


@pytest.mark.unit
def test_tool_failure_fails_closed_without_report(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.graphs.nodes as nodes_module

    def broken_tool(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("simulated store outage")

    monkeypatch.setitem(nodes_module._TOOL_FUNCTIONS, "query_sensor_history", broken_tool)
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(VIBRATION_PLAN)],
        draft_responses=[],
    )
    graph = build_diagnosis_graph(model, structured_output_method="json_schema")
    result = graph.invoke(
        {
            "request_id": "request-graph-001",
            "device_id": "PUMP-003",
            "question": "研判振动。",
        },
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )
    assert result["report"] is None
    assert "query_sensor_history" in result["error"]
    # The formatter must never run when evidence collection failed.
    assert model._formatter is not None and model._formatter.calls == 0


@pytest.mark.unit
def test_identity_mismatch_blocks_formatting() -> None:
    wrong_identity = _sufficient_draft(request_id="someone-else")
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(VIBRATION_PLAN)],
        draft_responses=[wrong_identity],
    )
    graph = build_diagnosis_graph(model, structured_output_method="json_schema")
    with pytest.raises(RuntimeError, match="did not preserve request identity"):
        graph.invoke(
            {
                "request_id": "request-graph-001",
                "device_id": "PUMP-003",
                "question": "研判振动。",
            },
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
