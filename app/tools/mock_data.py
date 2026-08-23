"""Fixed mock evidence for the first learning milestone; no external I/O."""

from __future__ import annotations

from datetime import UTC, datetime

PUMP_003 = {
    "device_id": "PUMP-003",
    "name": "3号循环泵",
    "device_type": "circulation_pump",
    "location": "循环水站 A 区",
    "status": "running",
    "vibration_alarm_threshold_mm_s": 7.1,
    "source_id": "asset:PUMP-003",
    "source_type": "mock_asset_registry",
    "version": "2026.08.mock.1",
}

SENSOR_EVENTS = (
    {
        "event_id": "sensor:PUMP-003:2026-08-22T01:00:00Z",
        "device_id": "PUMP-003",
        "metric": "vibration_mm_s",
        "value": 7.8,
        "unit": "mm/s",
        "observed_at": datetime(2026, 8, 22, 1, 0, tzinfo=UTC),
    },
    {
        "event_id": "sensor:PUMP-003:2026-08-22T01:10:00Z",
        "device_id": "PUMP-003",
        "metric": "vibration_mm_s",
        "value": 8.2,
        "unit": "mm/s",
        "observed_at": datetime(2026, 8, 22, 1, 10, tzinfo=UTC),
    },
    {
        "event_id": "sensor:PUMP-003:2026-08-22T01:20:00Z",
        "device_id": "PUMP-003",
        "metric": "vibration_mm_s",
        "value": 8.0,
        "unit": "mm/s",
        "observed_at": datetime(2026, 8, 22, 1, 20, tzinfo=UTC),
    },
)

ALARM_EVENTS = (
    {
        "alarm_id": "alarm:PUMP-003:2026-08-22T01:10:00Z",
        "device_id": "PUMP-003",
        "code": "PUMP_VIBRATION_HIGH",
        "severity": "high",
        "message": "振动连续 20 分钟超过 7.1 mm/s 阈值。",
        "observed_at": datetime(2026, 8, 22, 1, 10, tzinfo=UTC),
    },
    {
        "alarm_id": "alarm:PUMP-003:2026-08-21T04:40:00Z",
        "device_id": "PUMP-003",
        "code": "PUMP_VIBRATION_HIGH",
        "severity": "medium",
        "message": "前一日发生过一次短时振动超限。",
        "observed_at": datetime(2026, 8, 21, 4, 40, tzinfo=UTC),
    },
)

WORK_ORDERS = (
    {
        "work_order_id": "WO-20260815-003",
        "device_id": "PUMP-003",
        "status": "closed",
        "summary": "检查联轴器找正，发现轻微偏差后已调整。",
        "observed_at": datetime(2026, 8, 15, 6, 0, tzinfo=UTC),
    },
)

MANUAL_SECTIONS = (
    {
        "source_id": "manual:circulation-pump-v2:section-4.2",
        "device_id": "PUMP-003",
        "title": "振动超限处置",
        "content": "振动连续超限时，应核查轴承、联轴器找正和基础紧固状态。涉及停机须由值班负责人批准。",
        "version": "2.0",
    },
    {
        "source_id": "manual:circulation-pump-v2:section-3.1",
        "device_id": "PUMP-003",
        "title": "日常巡检",
        "content": "记录振动趋势并与报警阈值比较；本模拟手册不包含可执行控制指令。",
        "version": "2.0",
    },
)
