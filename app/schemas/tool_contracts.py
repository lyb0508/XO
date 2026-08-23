"""Validated, side-effect-free tool parameter contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DeviceLookupInput(ToolInput):
    device_id: str = Field(min_length=1, max_length=128)


class TimeRangeInput(DeviceLookupInput):
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
    metric: str = Field(default="vibration_mm_s", min_length=1, max_length=64)


class AlarmHistoryInput(TimeRangeInput):
    pass


class WorkOrderHistoryInput(TimeRangeInput):
    pass


class ManualSearchInput(ToolInput):
    device_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=500)
