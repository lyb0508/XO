"""Fixed mock evidence for the first learning milestone; no external I/O.

本模块是第一阶段学习用的固定 mock 证据库，围绕一台循环泵 PUMP-003 组织：
设备档案、三条振动测点、两条报警、一条已关闭工单和两节手册内容。

数据来源：所有记录手工写死、时间统一带 UTC 时区、字段完整且相互呼应
（传感器数值 7.8/8.2/8.0 mm/s 超过档案阈值 7.1 mm/s），让工具行为完全
确定、测试可以精确断言"连续超限 + 历史报警 + 上次检修"这一研判场景。

副作用边界：纯常量数据，没有任何 I/O；但注意修改这里的数值会同时改变
工具输出和相关测试的期望值，必须同步评审。

失败行为：纯数据模块本身不会失败；字段缺失的校验发生在工具层与检索层。
"""

from __future__ import annotations

from datetime import UTC, datetime

# 唯一设备的静态档案；vibration_alarm_threshold_mm_s 与下方传感器数值呼应，
# 构成"超过阈值即需研判"的最小闭环。
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

# 三条振动测点构成一次持续超限的趋势（7.8 -> 8.2 -> 8.0 mm/s），
# 时间间隔 10 分钟，供上层识别"连续 20 分钟以上超阈值"的故障场景。
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

# 两条同 code、不同 severity 的报警：一条当日 high、一条前一日 medium，
# 用于练习"结合历史报警判断趋势是否恶化"的研判逻辑。
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

# 一条已关闭工单，模拟"上次检修发现联轴器偏差"的历史背景，
# 让诊断结论可以引用过往维修记录作为佐证。
WORK_ORDERS = (
    {
        "work_order_id": "WO-20260815-003",
        "device_id": "PUMP-003",
        "status": "closed",
        "summary": "检查联轴器找正，发现轻微偏差后已调整。",
        "observed_at": datetime(2026, 8, 15, 6, 0, tzinfo=UTC),
    },
)

# 两节手册：一节讲振动超限处置（内含"停机须由值班负责人批准"的安全规则），
# 一节讲日常巡检。内容刻意声明不含可执行控制指令，作为"检索文本是不可信
# 证据而非指令"这一安全边界的示例语料。
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
