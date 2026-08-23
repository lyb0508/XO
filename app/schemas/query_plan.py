"""Structured query plan produced by the planning node before any tool runs.

本模块定义 QueryPlan：规划节点让模型输出的"查询计划"契约。它位于工具执行之前
——模型只负责声明意图（查哪台设备、要哪几类证据、时间窗与指标），Graph 再据此
确定性地展开工具调用，模型不直接决定工具如何执行。

数据来源：由规划节点的 LLM 结构化输出生成，属于不可信输入，因此每个字段都要
经过严格校验与交叉一致性检查。

副作用边界：纯 schema 定义，本身没有任何 I/O 或写入。

失败时的行为：字段缺失、时间窗缺失或倒序、"范围外计划却请求证据"等矛盾组合
都会触发 ValidationError，交由上游节点转入追问或拒答分支。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.diagnostics import StrictModel

EVIDENCE_TYPE = Literal["device", "sensor", "alarm", "work_order", "manual"]

TIMED_EVIDENCE_TYPES = frozenset({"sensor", "alarm", "work_order"})


class QueryPlan(StrictModel):
    """Model-written intent; the program alone decides how tools are executed.

    A plan never carries tool results or facts. Time windows are mandatory for
    evidence types whose stores are time-bounded so the graph can fan out with
    complete, validated arguments.

    中文说明：计划只描述"意图"，不携带任何工具结果或事实；Graph 拿到计划后
    才真正执行工具。这样把模型的自由度限制在声明层，执行层保持确定性、可测试。
    """

    scope_status: Literal["in_scope", "out_of_scope", "needs_clarification"]
    reason: str = Field(min_length=1, max_length=400)
    device_id: str | None = Field(default=None, min_length=1, max_length=128)
    start_at: datetime | None = None
    end_at: datetime | None = None
    # Graph 对每类证据只发起一次工具调用，所以传感器请求最多只能列一个指标；
    # 要放宽这里必须先放宽执行契约，而不是只改 schema。
    metrics: list[str] = Field(default_factory=list, max_length=1)
    manual_query: str | None = Field(default=None, min_length=1, max_length=500)
    requested_evidence_types: list[EVIDENCE_TYPE] = Field(max_length=5)

    @field_validator("metrics")
    @classmethod
    def nonempty_metrics(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 64 for value in values):
            raise ValueError("each metric must be non-empty and at most 64 characters")
        if len(values) != len({value.strip() for value in values}):
            raise ValueError("metrics must not contain duplicates")
        return values

    @field_validator("start_at", "end_at")
    @classmethod
    def timezone_required(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("plan timestamps must include timezone information")
        return value

    @field_validator("requested_evidence_types")
    @classmethod
    def unique_evidence_types(cls, values: list[EVIDENCE_TYPE]) -> list[EVIDENCE_TYPE]:
        if len(values) != len(set(values)):
            raise ValueError("requested_evidence_types must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_plan_consistency(self) -> "QueryPlan":
        if self.scope_status == "in_scope":
            if not self.device_id:
                raise ValueError("an in-scope plan must name a device_id")
            if not self.requested_evidence_types:
                raise ValueError("an in-scope plan must request at least one evidence type")
        elif self.requested_evidence_types:
            raise ValueError(f"a {self.scope_status} plan must not request evidence types")
        timed = [value for value in self.requested_evidence_types if value in TIMED_EVIDENCE_TYPES]
        if timed:
            missing_window = (
                "start_at and end_at are required for "
                + ", ".join(sorted(timed))
                + " evidence"
            )
            if self.start_at is None or self.end_at is None:
                raise ValueError(missing_window)
            if self.start_at >= self.end_at:
                raise ValueError("start_at must be before end_at")
        # 前向要求从严（请求了的证据必须给齐参数），但对未请求证据类型的多余
        # 字段从宽：Graph 只按 requested_evidence_types 展开，其余字段不会被
        # 消费；在这里拒绝它们只会把模型无害的冗余输出变成一次失败。
        if "sensor" in self.requested_evidence_types and not self.metrics:
            raise ValueError("a plan requesting sensor evidence must list at least one metric")
        if "manual" in self.requested_evidence_types and not (self.manual_query and self.manual_query.strip()):
            raise ValueError("a plan requesting manual evidence must provide manual_query")
        return self
