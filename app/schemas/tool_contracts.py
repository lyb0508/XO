"""Validated, side-effect-free tool parameter contracts.

本模块定义各只读工具的参数契约：DeviceLookupInput、TimeRangeInput、
SensorHistoryInput、AlarmHistoryInput、WorkOrderHistoryInput 与
ManualSearchInput。

在整个项目中的位置：位于模型与工具实现之间——LLM 生成的工具调用参数先经过
这些 schema 校验，才会到达真正的工具函数，从机制上阻止分页/限额类注入参数、
缺时区的时间戳或空查询进入执行层。

数据来源：字段值来自模型输出的工具调用参数，属于不可信输入；extra="forbid"
保证模型多塞的任何未知字段都会被拒绝。

副作用边界：纯 schema 定义，无 I/O、无写入；工具本身的只读边界由工具实现保证，
这里负责的是"参数合法"这一道闸门。

失败时的行为：校验失败会抛出 ValidationError，由工具入口转成显式错误返回给
模型，让它在调用限额内自行修正参数后重试。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolInput(BaseModel):
    """所有工具参数的公共基类：禁止未知字段，并自动去除字符串首尾空白。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DeviceLookupInput(ToolInput):
    """按 device_id 查询设备信息的最小参数集。"""

    device_id: str = Field(min_length=1, max_length=128)


class TimeRangeInput(DeviceLookupInput):
    """带时间窗的查询基类：时间戳必须携带时区信息，且起点必须早于终点。

    强制时区是为了消除"本地时间是哪天"的歧义，保证不同环境下
    同一参数解析出同一时刻。
    """

    start_at: datetime
    end_at: datetime

    @field_validator("start_at", "end_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include timezone information")
        return value

    @model_validator(mode="after")
    def valid_range(self) -> "TimeRangeInput":
        if self.start_at >= self.end_at:
            raise ValueError("start_at must be before end_at")
        return self


class SensorHistoryInput(TimeRangeInput):
    """传感器历史查询参数：默认关注振动指标 vibration_mm_s。"""

    metric: str = Field(default="vibration_mm_s", min_length=1, max_length=64)


class AlarmHistoryInput(TimeRangeInput):
    """报警历史查询参数：时间窗 + device_id，无额外字段。"""


class WorkOrderHistoryInput(TimeRangeInput):
    """工单历史查询参数：时间窗 + device_id，无额外字段。"""


class ManualSearchInput(ToolInput):
    """手册检索参数：query 限制在 500 字符内，防止超长查询拖垮检索。"""

    device_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=500)
