"""Frozen contract tests for the phase-two QueryPlan schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.query_plan import QueryPlan

UTC_WINDOW = {
    "start_at": "2026-08-22T00:00:00+00:00",
    "end_at": "2026-08-22T02:00:00+00:00",
}


def _plan(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "scope_status": "in_scope",
        "reason": "振动报警研判",
        "device_id": "PUMP-003",
        "metrics": ["vibration_mm_s"],
        "requested_evidence_types": ["sensor"],
        **UTC_WINDOW,
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_valid_in_scope_sensor_plan_is_accepted() -> None:
    plan = QueryPlan.model_validate(_plan())
    assert plan.device_id == "PUMP-003"
    assert plan.start_at is not None and plan.start_at.utcoffset() is not None


@pytest.mark.unit
def test_in_scope_plan_requires_device() -> None:
    with pytest.raises(ValidationError, match="device_id"):
        QueryPlan.model_validate(_plan(device_id=None))


@pytest.mark.unit
def test_in_scope_plan_requires_requested_types() -> None:
    with pytest.raises(ValidationError, match="at least one evidence type"):
        QueryPlan.model_validate(_plan(requested_evidence_types=[]))


@pytest.mark.unit
@pytest.mark.parametrize("field", ["start_at", "end_at"])
def test_timed_evidence_requires_complete_window(field: str) -> None:
    payload = _plan(requested_evidence_types=["alarm"])
    payload[field] = None  # type: ignore[assignment]
    with pytest.raises(ValidationError, match="start_at and end_at are required"):
        QueryPlan.model_validate(payload)


@pytest.mark.unit
def test_window_must_be_ordered_and_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="before end_at"):
        QueryPlan.model_validate(
            _plan(
                start_at="2026-08-22T03:00:00+00:00",
                end_at="2026-08-22T02:00:00+00:00",
            )
        )
    with pytest.raises(ValidationError, match="timezone"):
        QueryPlan.model_validate(
            _plan(start_at="2026-08-22T00:00:00", end_at="2026-08-22T02:00:00+00:00")
        )


@pytest.mark.unit
def test_sensor_requests_require_metrics() -> None:
    with pytest.raises(ValidationError, match="at least one metric"):
        QueryPlan.model_validate(_plan(metrics=[]))


@pytest.mark.unit
def test_manual_query_required_for_manual_evidence() -> None:
    with pytest.raises(ValidationError, match="manual_query"):
        QueryPlan.model_validate(_plan(requested_evidence_types=["manual"], manual_query=None))


@pytest.mark.unit
def test_extra_fields_for_unrequested_types_are_tolerated_and_ignored_by_the_program() -> None:
    # Live small models sometimes emit fields for unrequested evidence types.
    # The plan accepts them; fan-out is driven solely by requested_evidence_types.
    plan = QueryPlan.model_validate(_plan(manual_query="轴承"))
    assert plan.manual_query == "轴承" and "manual" not in plan.requested_evidence_types
    sensor_only = QueryPlan.model_validate(
        _plan(requested_evidence_types=["device"], metrics=[], manual_query=None)
    )
    assert sensor_only.requested_evidence_types == ["device"]


@pytest.mark.unit
def test_non_in_scope_plans_may_not_request_evidence() -> None:
    for scope in ("out_of_scope", "needs_clarification"):
        with pytest.raises(ValidationError, match=scope):
            QueryPlan.model_validate(_plan(scope_status=scope, requested_evidence_types=["sensor"]))


@pytest.mark.unit
def test_out_of_scope_plan_with_reason_only_is_valid() -> None:
    plan = QueryPlan.model_validate(
        {"scope_status": "out_of_scope", "reason": "与设备诊断无关", "requested_evidence_types": []}
    )
    assert plan.device_id is None and plan.requested_evidence_types == []


class _ScriptedPlanner:
    """按脚本回放的假规划器：异常或 dict 依次弹出，记录每次收到的消息。"""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[object]] = []

    def invoke(self, messages: object) -> object:
        self.calls.append(list(messages))  # type: ignore[arg-type]
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.unit
def test_plan_retry_succeeds_after_first_parse_failure() -> None:
    from app.graphs.nodes import make_plan_queries

    planner = _ScriptedPlanner(
        [
            RuntimeError("malformed plan output"),
            {"scope_status": "in_scope", "reason": "r", "device_id": "OTHER",
             **UTC_WINDOW, "metrics": ["vibration_mm_s"], "requested_evidence_types": ["sensor"]},
        ]
    )
    node = make_plan_queries(planner)
    state = {"request_id": "req-1", "device_id": "PUMP-003", "question": "研判振动"}
    result = node(state)
    assert len(planner.calls) == 2
    # 重试消息必须携带第一次的真实错误反馈，而不是盲目重发同一提示。
    feedback = str(planner.calls[1][-1])
    assert "could not be used" in feedback and "RuntimeError" in feedback
    # device_id 被程序覆写为请求中的目标设备。
    assert result["query_plan"]["device_id"] == "PUMP-003"


@pytest.mark.unit
def test_plan_retry_is_bounded_and_reraises_second_failure() -> None:
    from app.graphs.nodes import make_plan_queries

    planner = _ScriptedPlanner(
        [ValueError("bad json one"), ValueError("bad json two")]
    )
    node = make_plan_queries(planner)
    state = {"request_id": "req-1", "device_id": "PUMP-003", "question": "研判振动"}
    with pytest.raises(ValueError, match="bad json two"):
        node(state)
    assert len(planner.calls) == 2  # 只重试一次，不无限放大。


@pytest.mark.unit
def test_plan_happy_path_invokes_planner_exactly_once() -> None:
    from app.graphs.nodes import make_plan_queries

    planner = _ScriptedPlanner(
        [{"scope_status": "in_scope", "reason": "r", "device_id": "PUMP-003",
          **UTC_WINDOW, "metrics": ["vibration_mm_s"], "requested_evidence_types": ["sensor"]}]
    )
    node = make_plan_queries(planner)
    state = {"request_id": "req-1", "device_id": "PUMP-003", "question": "研判振动"}
    result = node(state)
    assert len(planner.calls) == 1
    assert result["query_plan"]["scope_status"] == "in_scope"


# ---------- 计划规范化层：程序在校验前执行的确定性修正 ----------

from app.graphs.nodes import _normalize_plan_payload


@pytest.mark.unit
def test_normalize_clears_evidence_fields_for_non_in_scope() -> None:
    payload = {
        "scope_status": "needs_clarification",
        "reason": "时间窗不明确",
        "requested_evidence_types": ["sensor"],
        "metrics": ["vibration_mm_s"],
        "start_at": UTC_WINDOW["start_at"],
        "end_at": UTC_WINDOW["end_at"],
        "device_id": "PUMP-003",
    }
    out = _normalize_plan_payload(payload)
    # 追问类计划不消费任何证据字段；清空而非拒绝整个计划。
    assert out["requested_evidence_types"] == []
    assert out["metrics"] == []
    assert out["device_id"] is None
    assert out["scope_status"] == "needs_clarification"


@pytest.mark.unit
def test_normalize_fills_default_metric_for_sensor_requests() -> None:
    payload = {
        "scope_status": "in_scope",
        "reason": "r",
        "device_id": "PUMP-003",
        "requested_evidence_types": ["sensor", "manual"],
        "manual_query": "vibration",
        **UTC_WINDOW,
    }
    out = _normalize_plan_payload(payload)
    assert out["metrics"] == ["vibration_mm_s"]


@pytest.mark.unit
def test_normalize_downgrades_in_scope_history_without_window() -> None:
    payload = {
        "scope_status": "in_scope",
        "reason": "研判振动",
        "device_id": "PUMP-003",
        "requested_evidence_types": ["sensor"],
    }
    out = _normalize_plan_payload(payload)
    # 程序不能替用户发明时间窗：降级为追问并清空全部证据意图。
    assert out["scope_status"] == "needs_clarification"
    assert out["requested_evidence_types"] == []
    assert out["device_id"] is None
    assert "time window" in out["reason"]


@pytest.mark.unit
def test_normalize_keeps_in_scope_non_timed_request_untouched() -> None:
    payload = {
        "scope_status": "in_scope",
        "reason": "设备档案查询",
        "device_id": "PUMP-003",
        "requested_evidence_types": ["device"],
    }
    out = _normalize_plan_payload(payload)
    # device 类无时间窗依赖，也不需要 metrics：不应被改动。
    assert out["scope_status"] == "in_scope"
    assert out["requested_evidence_types"] == ["device"]
    assert "metrics" not in out or out.get("metrics") in ([], None) or out["metrics"] == []


# ---------- formatter 输出的两处程序拥有修正 ----------

def _draft_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "request_id": "req-1",
        "device_id": "PUMP-003",
        "scope_status": "in_scope",
        "risk_level": "low",
        "summary": "s",
        "evidence_sufficient": False,
        "likely_causes": [],
        "evidence_ids": [],
        "recommended_actions": [],
        "requires_human_review": False,
        "limitations": [],
    }
    base.update(overrides)
    return base


class _EchoFormatter:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def invoke(self, messages: object) -> dict[str, object]:
        self.calls += 1
        return self.payload


@pytest.mark.unit
def test_format_report_dedupes_evidence_ids_and_forces_review_for_high_risk() -> None:
    from app.graphs.nodes import make_format_report

    payload = _draft_payload(
        risk_level="high",
        evidence_sufficient=True,
        evidence_ids=["evt-1", "evt-2", "evt-1"],
        requires_human_review=False,
    )
    formatter = _EchoFormatter(payload)
    node = make_format_report(formatter)
    state = {"request_id": "req-1", "device_id": "PUMP-003", "registry_entries": [], "question": "q"}
    result = node(state)["draft"]
    # 重复 ID 被保序去重；high 风险被强制要求人工复核。
    assert result["evidence_ids"] == ["evt-1", "evt-2"]
    assert result["requires_human_review"] is True


@pytest.mark.unit
def test_plan_recovers_when_structured_parsing_reports_error() -> None:
    """include_raw 契约下解析失败不抛异常：规范化层应从 raw 文本救回计划。"""

    from app.graphs.nodes import make_plan_queries

    broken = (
        '{"scope_status": "needs_clarification", "reason": "时间窗不明确", '
        '"requested_evidence_types": ["sensor"], "device_id": "PUMP-003"}'
    )
    planner = _ScriptedPlanner(
        [{"raw": type("Msg", (), {"content": broken})(), "parsed": None,
          "parsing_error": "a needs_clarification plan must not request evidence types"}]
    )
    node = make_plan_queries(planner)
    state = {"request_id": "req-1", "device_id": "PUMP-003", "question": "研判振动"}
    result = node(state)
    plan = result["query_plan"]
    assert plan["scope_status"] == "needs_clarification"
    # 规范化层清掉了追问类计划携带的证据类型，无需重试即可通过校验。
    assert plan["requested_evidence_types"] == []
    assert len(planner.calls) == 1


@pytest.mark.unit
def test_structured_payload_extracts_from_tool_call_args_when_content_empty() -> None:
    """function_calling 失败路径：参数在 tool_call 里而 content 为空。"""

    from app.graphs.nodes import _structured_result_payload

    broken = {"scope_status": "needs_clarification", "reason": "r",
              "requested_evidence_types": ["sensor"], "device_id": "PUMP-003"}
    result = {
        "parsed": None,
        "parsing_error": "needs_clarification plan must not request evidence types",
        "raw": type("Msg", (), {"content": "", "tool_calls": [{"name": "QueryPlan", "args": broken}]})(),
    }
    payload = _structured_result_payload(result)
    assert payload["scope_status"] == "needs_clarification"
    assert payload["requested_evidence_types"] == ["sensor"]
