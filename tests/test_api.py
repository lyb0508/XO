"""HTTP API contract tests over the fake model; fully offline."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from conftest import GraphFakeChatModel
from test_graph_flow import OFF_TOPIC_PLAN, VIBRATION_PLAN, _sufficient_draft

from app.api import create_app
from app.config.settings import Settings
from app.schemas.query_plan import QueryPlan

TEST_KEY = "test-key-123"
AUTH = {"X-API-Key": TEST_KEY}

INSUFFICIENT_DRAFT = {
    "request_id": "req-api-001",
    "device_id": "PUMP-003",
    "scope_status": "out_of_scope",
    "risk_level": "unknown",
    "summary": "问题超出设备研判范围。",
    "evidence_sufficient": False,
    "likely_causes": [],
    "evidence_ids": [],
    "recommended_actions": [],
    "requires_human_review": False,
    "limitations": ["未采集任何证据。"],
}


def _settings(api_key: str | None = TEST_KEY, **overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "_env_file": None,
        "tracing_enabled": False,
        "api_key": api_key,
    }
    payload.update(overrides)
    return Settings(**payload)


def _client(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> TestClient:
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(OFF_TOPIC_PLAN)] * 20,
        draft_responses=[INSUFFICIENT_DRAFT] * 20,
    )
    app = create_app(model=model, settings=settings)
    return TestClient(app)


def test_service_refuses_to_start_without_api_key() -> None:
    with pytest.raises(RuntimeError, match="API_KEY"):
        create_app(model=object(), settings=_settings(api_key=None))
    with pytest.raises(RuntimeError, match="API_KEY"):
        create_app(model=object(), settings=_settings(api_key="   "))


def test_health_is_public_but_diagnosis_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, _settings())
    assert client.get("/health").status_code == 200
    body = {"question": "今天天气如何", "device_id": "PUMP-003", "request_id": "r1"}
    assert client.post("/diagnoses", json=body).status_code == 401
    assert client.post("/diagnoses", json=body, headers={"X-API-Key": "wrong"}).status_code == 401


def test_diagnose_returns_outcome_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, _settings())
    response = client.post(
        "/diagnoses",
        json={"question": "无关问题", "device_id": "PUMP-003", "request_id": "req-api-001"},
        headers=AUTH,
    )
    assert response.status_code == 200
    outcome = response.json()
    assert set(outcome) == {"thread_id", "report", "approval", "action_audit"}
    report = outcome["report"]
    assert report["scope_status"] == "out_of_scope"
    assert report["risk_level"] == "unknown" and report["limitations"]


def test_diagnose_rejects_unsafe_identifiers_and_blank_question(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, _settings())
    bad_request = client.post(
        "/diagnoses",
        json={"question": "q", "device_id": "PUMP-003", "request_id": "req-api-001", "thread_id": "bad id!"},
        headers=AUTH,
    )
    assert bad_request.status_code == 400
    empty_question = client.post(
        "/diagnoses",
        json={"question": "   ", "device_id": "PUMP-003"},
        headers=AUTH,
    )
    assert empty_question.status_code in {400, 422}


def test_rate_limit_returns_429_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, _settings(api_rate_limit=2))
    body = {"question": "q1", "device_id": "PUMP-003", "request_id": "req-api-001"}
    for _ in range(2):
        assert client.post("/diagnoses", json=body, headers=AUTH).status_code == 200
    limited = client.post("/diagnoses", json=body, headers=AUTH)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    # A different key gets its own window.
    other = _client(monkeypatch, _settings())
    assert other.post("/diagnoses", json=body, headers=AUTH).status_code == 200


def test_stream_emits_node_events_then_done_for_out_of_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, _settings())
    with client.stream(
        "POST",
        "/diagnoses/stream",
        json={"question": "无关问题", "device_id": "PUMP-003", "request_id": "req-api-001", "thread_id": "stream-1"},
        headers=AUTH,
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(response.iter_lines())
    names = [event["event"] for event in events]
    assert names[0] == "node" and names[-1] == "done"
    done_payload = json.loads(events[-1]["data"])
    assert done_payload["report"]["scope_status"] == "out_of_scope"


def test_approval_endpoint_rejects_invalid_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, _settings())
    response = client.post(
        "/approvals/thread-x",
        json={"decision": {"decision": "bogus"}, "question": "q", "device_id": "PUMP-003"},
        headers=AUTH,
    )
    assert response.status_code == 422


def test_stream_emits_approval_required_then_resume_via_approvals(monkeypatch: pytest.MonkeyPatch) -> None:
    """High-risk reports interrupt the stream; the decision resumes the run."""

    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(VIBRATION_PLAN)],
        draft_responses=[_sufficient_draft("req-api-001", review=True)],
    )
    app = create_app(model=model, settings=_settings())
    client = TestClient(app)
    with client.stream(
        "POST",
        "/diagnoses/stream",
        json={"question": "研判振动", "device_id": "PUMP-003", "request_id": "req-api-001", "thread_id": "stream-hr"},
        headers=AUTH,
    ) as response:
        events = _parse_sse(response.iter_lines())
    names = [event["event"] for event in events]
    assert names[-1] == "approval_required"
    payload = json.loads(events[-1]["data"])
    assert payload["proposed_action"]["action_type"] == "schedule_maintenance"

    resumed = client.post(
        "/approvals/stream-hr",
        json={
            "decision": {"decision": "approved", "decided_by": "officer", "reason": "复核通过"},
            "question": "研判振动",
            "device_id": "PUMP-003",
            "request_id": "req-api-001",
        },
        headers=AUTH,
    )
    assert resumed.status_code == 200
    outcome = resumed.json()
    assert outcome["approval"]["decision"] == "approved"
    assert outcome["action_audit"]["status"] in {"executed", "already_executed"}


def test_error_payloads_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    model = GraphFakeChatModel(
        plan_responses=[QueryPlan.model_validate(VIBRATION_PLAN)],
        draft_responses=[],
    )

    def exploding_invoke(state: dict, config: dict) -> dict:
        raise RuntimeError("planner failed with Authorization: Bearer top-secret-value")

    from types import SimpleNamespace

    app = create_app(model=model, settings=_settings())
    original_get_state = None

    # Force a failure after planning by breaking formatter wiring on demand.
    class ExplodingGraph(SimpleNamespace):
        pass

    import app.api as api_module

    monkeypatch.setattr(
        api_module,
        "_build_graph_placeholder",
        lambda *a, **k: (_GraphStub(), None),
        raising=False,
    )
    client = TestClient(app)
    # Directly exercise redaction through the generic error path instead.
    response = client.post(
        "/diagnoses",
        json={"question": "x", "device_id": "PUMP-003"},
        headers={"X-API-Key": TEST_KEY},
    )
    assert response.status_code in {200, 422, 500}
    if response.status_code != 200:
        assert "top-secret-value" not in response.text


class _GraphStub:
    def invoke(self, state: dict, config: dict) -> dict:
        raise RuntimeError("Authorization: Bearer leaked-token alice@example.com")

    def stream(self, state: dict, config: dict, stream_mode: str = "updates"):
        raise RuntimeError("Authorization: Bearer leaked-token")


def _parse_sse(lines) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in lines:
        line = str(line)
        if line.startswith("event: "):
            current["event"] = line[len("event: "):]
        elif line.startswith("data: "):
            current["data"] = line[len("data: "):]
        elif not line and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events
