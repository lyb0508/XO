from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.tools.industrial import (
    INDUSTRIAL_TOOLS,
    get_device_info,
    query_alarm_history,
    query_sensor_history,
    query_work_order_history,
    search_manual,
)
import app.tools.industrial as industrial


def test_exactly_five_frozen_read_only_tool_names_are_registered() -> None:
    assert {tool.name for tool in INDUSTRIAL_TOOLS} == {
        "get_device_info",
        "query_sensor_history",
        "query_alarm_history",
        "query_work_order_history",
        "search_manual",
    }


def test_device_lookup_has_stable_success_and_not_found_contract() -> None:
    success = get_device_info.invoke({"device_id": "PUMP-003"})
    missing = get_device_info.invoke({"device_id": "UNKNOWN"})

    assert success["status"] == "ok"
    assert success["device"]["name"] == "3号循环泵"
    assert success["source_id"] == "asset:PUMP-003"
    assert missing == {
        "status": "not_found",
        "source_id": "asset:UNKNOWN",
        "source_type": "mock_asset_registry",
    }


@pytest.mark.parametrize(
    ("tool", "key", "expected_schema_fields", "unexpected_limit"),
    [
        (query_sensor_history, "points", {"device_id", "start_at", "end_at", "metric"}, "max_points"),
        (query_alarm_history, "alarms", {"device_id", "start_at", "end_at"}, "max_records"),
        (query_work_order_history, "work_orders", {"device_id", "start_at", "end_at"}, "max_records"),
    ],
)
def test_timed_tool_schemas_exclude_model_selectable_limits_and_return_evidence(
    tool: object,
    key: str,
    expected_schema_fields: set[str],
    unexpected_limit: str,
    utc_window: tuple[datetime, datetime],
) -> None:
    start_at, end_at = utc_window
    payload = {"device_id": "PUMP-003", "start_at": start_at, "end_at": end_at}
    if key == "points":
        payload["metric"] = "vibration_mm_s"
    assert set(tool.args_schema.model_fields) == expected_schema_fields
    with pytest.raises(ValidationError):
        tool.invoke({**payload, unexpected_limit: 1})

    result = tool.invoke(payload)

    assert result["status"] == "ok"
    assert result["source_type"].startswith("mock_")
    assert len(result[key]) <= 20


def test_unknown_device_returns_not_found_from_each_history_tool(utc_window: tuple[datetime, datetime]) -> None:
    start_at, end_at = utc_window
    for tool in (query_sensor_history, query_alarm_history, query_work_order_history):
        result = tool.invoke({"device_id": "UNKNOWN", "start_at": start_at, "end_at": end_at})
        assert result["status"] == "not_found"


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {
            "device_id": "PUMP-003",
            "start_at": "2026-08-22T00:00:00",
            "end_at": "2026-08-22T01:00:00",
        },
        {
            "device_id": "PUMP-003",
            "start_at": "2026-08-22T02:00:00+00:00",
            "end_at": "2026-08-22T01:00:00+00:00",
        },
        {
            "device_id": "PUMP-003",
            "start_at": "2026-08-22T00:00:00+00:00",
            "end_at": "2026-08-22T01:00:00+00:00",
            "max_points": 1,
        },
    ],
)
def test_sensor_input_rejects_invalid_timestamps_and_bounds_before_tool_body(
    invalid_payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        query_sensor_history.invoke(invalid_payload)


def test_manual_schema_excludes_model_selectable_limit_and_returns_reference_metadata() -> None:
    assert set(search_manual.args_schema.model_fields) == {"device_id", "query"}
    with pytest.raises(ValidationError):
        search_manual.invoke({"device_id": "PUMP-003", "query": "轴承 联轴器", "max_results": 1})

    result = search_manual.invoke({"device_id": "PUMP-003", "query": "轴承 联轴器"})

    assert result["status"] == "ok"
    assert result["source_id"] == "manual:circulation-pump-v2"
    assert result["version"] == "2.0"
    assert result["results"][0]["source_id"].startswith("manual:circulation-pump-v2:section-")
    assert "控制" not in result["results"][0]["content"]


def test_program_owned_result_caps_truncate_more_than_the_frozen_limits(
    monkeypatch: pytest.MonkeyPatch, utc_window: tuple[datetime, datetime]
) -> None:
    start_at, end_at = utc_window
    sensor_start = datetime(2026, 8, 20, tzinfo=UTC)
    monkeypatch.setattr(
        industrial,
        "SENSOR_EVENTS",
        tuple(
            {
                **industrial.SENSOR_EVENTS[0],
                "event_id": f"sensor:synthetic:{index}",
                "observed_at": sensor_start + timedelta(minutes=index),
            }
            for index in range(21)
        ),
    )
    monkeypatch.setattr(
        industrial,
        "ALARM_EVENTS",
        tuple(
            {
                **industrial.ALARM_EVENTS[0],
                "alarm_id": f"alarm:synthetic:{index}",
                "observed_at": sensor_start + timedelta(minutes=index),
            }
            for index in range(21)
        ),
    )
    monkeypatch.setattr(
        industrial,
        "WORK_ORDERS",
        tuple(
            {
                **industrial.WORK_ORDERS[0],
                "work_order_id": f"WO-SYNTHETIC-{index:02d}",
                "observed_at": sensor_start + timedelta(minutes=index),
            }
            for index in range(21)
        ),
    )
    monkeypatch.setattr(
        industrial,
        "MANUAL_SECTIONS",
        tuple(
            {
                **industrial.MANUAL_SECTIONS[0],
                "source_id": f"manual:synthetic:section-{index}",
                "title": f"振动 synthetic {index}",
            }
            for index in range(4)
        ),
    )

    base = {"device_id": "PUMP-003", "start_at": start_at, "end_at": end_at}
    sensor = query_sensor_history.invoke({**base, "metric": "vibration_mm_s"})
    alarms = query_alarm_history.invoke(base)
    work_orders = query_work_order_history.invoke(base)
    manual = search_manual.invoke({"device_id": "PUMP-003", "query": "振动"})

    assert len(sensor["points"]) == 20
    assert len(alarms["alarms"]) == 20
    assert len(work_orders["work_orders"]) == 20
    assert len(manual["results"]) == 3
    assert sensor["observed_at"] == sensor["points"][-1]["observed_at"]
