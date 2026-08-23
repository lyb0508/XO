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
def test_sensor_requests_require_metrics_and_metric_needs_sensor() -> None:
    with pytest.raises(ValidationError, match="at least one metric"):
        QueryPlan.model_validate(_plan(metrics=[]))
    with pytest.raises(ValidationError, match="only allowed when sensor"):
        QueryPlan.model_validate(_plan(requested_evidence_types=["device"], metrics=["vibration_mm_s"]))


@pytest.mark.unit
def test_manual_query_binding() -> None:
    with pytest.raises(ValidationError, match="manual_query"):
        QueryPlan.model_validate(_plan(requested_evidence_types=["manual"], manual_query=None))
    with pytest.raises(ValidationError, match="manual_query is only allowed"):
        QueryPlan.model_validate(_plan(manual_query="轴承"))


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
