from __future__ import annotations

from datetime import UTC, datetime
import json

import pytest
from langchain_core.messages import ToolMessage

from app.agents.evidence import build_evidence_registry
from app.tools.industrial import (
    get_device_info, query_alarm_history, query_sensor_history, query_work_order_history, search_manual,
)


def _message(name: str, payload: object, ident: str, *, status: str = "success") -> ToolMessage:
    # LangGraph's tool node serializes structured return values before they
    # cross into ToolMessage; construct the boundary exactly that way here.
    content = payload if status == "error" and isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return ToolMessage(content=content, name=name, tool_call_id=ident, status=status)


def test_registry_builds_all_five_canonical_evidence_types_with_stable_facts() -> None:
    start, end = datetime(2026, 8, 14, tzinfo=UTC), datetime(2026, 8, 23, tzinfo=UTC)
    registry = build_evidence_registry([
        _message("get_device_info", get_device_info.invoke({"device_id": "PUMP-003"}), "d"),
        _message("query_sensor_history", query_sensor_history.invoke({"device_id": "PUMP-003", "start_at": start, "end_at": end, "metric": "vibration_mm_s"}), "s"),
        _message("query_alarm_history", query_alarm_history.invoke({"device_id": "PUMP-003", "start_at": start, "end_at": end}), "a"),
        _message("query_work_order_history", query_work_order_history.invoke({"device_id": "PUMP-003", "start_at": start, "end_at": end}), "w"),
        _message("search_manual", search_manual.invoke({"device_id": "PUMP-003", "query": "振动"}), "m"),
    ], "PUMP-003")
    types = {entry.evidence.evidence_type for entry in registry.entries.values()}
    assert types == {"device", "sensor", "alarm", "work_order", "manual"}
    assert registry.entries["asset:PUMP-003"].evidence.version == "2026.08.mock.1"
    assert registry.entries["sensor:PUMP-003:2026-08-22T01:10:00Z"].evidence.observed_at.isoformat() == "2026-08-22T01:10:00+00:00"
    assert registry.entries["manual:circulation-pump-v2:section-4.2"].evidence.source_id.endswith("section-4.2")


def test_not_found_and_successful_empty_lists_do_not_create_evidence() -> None:
    start, end = datetime(2026, 8, 14, tzinfo=UTC), datetime(2026, 8, 23, tzinfo=UTC)
    registry = build_evidence_registry([
        _message("get_device_info", get_device_info.invoke({"device_id": "UNKNOWN"}), "missing"),
        _message("query_sensor_history", {"status": "ok", "source_id": "mock_sensor_store:PUMP-003", "points": []}, "empty"),
    ], "PUMP-003")
    assert dict(registry.entries) == {}


def test_natural_language_tool_error_is_cleared_only_by_same_tool_success() -> None:
    registry = build_evidence_registry([
        _message("get_device_info", "device_id 缺失，请重新提供设备编号。", "failed", status="error"),
        _message("get_device_info", get_device_info.invoke({"device_id": "PUMP-003"}), "retried"),
    ], "PUMP-003")

    assert registry.unresolved_tool_errors == frozenset()
    assert set(registry.entries) == {"asset:PUMP-003"}


def test_natural_language_tool_error_remains_unresolved_when_another_tool_succeeds() -> None:
    registry = build_evidence_registry([
        _message("query_alarm_history", "时间范围格式错误，请修正后重试。", "failed", status="error"),
        _message("get_device_info", get_device_info.invoke({"device_id": "PUMP-003"}), "other-tool"),
    ], "PUMP-003")

    assert registry.unresolved_tool_errors == frozenset({"query_alarm_history"})
    assert set(registry.entries) == {"asset:PUMP-003"}


def test_natural_language_tool_error_is_cleared_by_same_tool_not_found_without_evidence() -> None:
    registry = build_evidence_registry([
        _message("query_work_order_history", "参数校验失败，请补充起止时间。", "failed", status="error"),
        _message(
            "query_work_order_history",
            {"status": "not_found", "source_id": "work-order:UNKNOWN", "source_type": "mock_work_order_store"},
            "retried-missing",
        ),
    ], "PUMP-003")

    assert registry.unresolved_tool_errors == frozenset()
    assert dict(registry.entries) == {}


def test_success_status_with_non_json_content_fails_closed() -> None:
    message = ToolMessage(
        content="这不是 JSON 工具结果",
        name="get_device_info",
        tool_call_id="bad-success",
        status="success",
    )
    with pytest.raises(RuntimeError, match="get_device_info.content is not JSON"):
        build_evidence_registry([message], "PUMP-003")


@pytest.mark.parametrize("payload", [
    {"status": "ok", "source_id": "x", "points": "not-list"},
    {"status": "ok", "source_id": "x", "points": [{"device_id": "PUMP-003"}]},
])
def test_malformed_successful_payload_fails_closed(payload: dict[str, object]) -> None:
    with pytest.raises(RuntimeError, match="malformed successful tool payload"):
        build_evidence_registry([_message("query_sensor_history", payload, "bad")], "PUMP-003")


def test_cross_device_success_and_conflicting_duplicate_id_are_rejected() -> None:
    cross_device = {"status": "ok", "source_id": "s", "points": [{"event_id": "same", "device_id": "OTHER", "metric": "vibration_mm_s", "value": 1.0, "unit": "mm/s", "observed_at": "2026-08-22T01:00:00+00:00"}]}
    with pytest.raises(RuntimeError, match="does not match requested"):
        build_evidence_registry([_message("query_sensor_history", cross_device, "cross")], "PUMP-003")
    one = {"status": "ok", "source_id": "s", "points": [{"event_id": "same", "device_id": "PUMP-003", "metric": "vibration_mm_s", "value": 1.0, "unit": "mm/s", "observed_at": "2026-08-22T01:00:00+00:00"}]}
    two = {"status": "ok", "source_id": "s", "points": [{"event_id": "same", "device_id": "PUMP-003", "metric": "vibration_mm_s", "value": 2.0, "unit": "mm/s", "observed_at": "2026-08-22T01:00:00+00:00"}]}
    with pytest.raises(RuntimeError, match="conflicting stable evidence_id"):
        build_evidence_registry([_message("query_sensor_history", one, "one"), _message("query_sensor_history", two, "two")], "PUMP-003")
