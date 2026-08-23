"""Structured query plan produced by the planning node before any tool runs."""

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
    """

    scope_status: Literal["in_scope", "out_of_scope", "needs_clarification"]
    reason: str = Field(min_length=1, max_length=400)
    device_id: str | None = Field(default=None, min_length=1, max_length=128)
    start_at: datetime | None = None
    end_at: datetime | None = None
    # The graph executes one tool call per evidence type, so the plan may name
    # exactly one metric per sensor request. Widening this requires widening
    # the execution contract first, not just the schema.
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
        else:
            if not self.requested_evidence_types:
                pass
            else:
                raise ValueError(
                    f"a {self.scope_status} plan must not request evidence types"
                )
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
        sensor_requested = "sensor" in self.requested_evidence_types
        if sensor_requested and not self.metrics:
            raise ValueError("a plan requesting sensor evidence must list at least one metric")
        if not sensor_requested and self.metrics:
            raise ValueError("metrics are only allowed when sensor evidence is requested")
        manual_requested = "manual" in self.requested_evidence_types
        if manual_requested and not (self.manual_query and self.manual_query.strip()):
            raise ValueError("a plan requesting manual evidence must provide manual_query")
        if not manual_requested and self.manual_query is not None:
            raise ValueError("manual_query is only allowed when manual evidence is requested")
        return self
