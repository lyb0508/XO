"""Strict schema for a diagnostician's final, evidence-backed response."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Reject unknown output fields so model output cannot silently expand the API."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LikelyCause(StrictModel):
    cause: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("evidence_ids must not contain duplicates")
        return values


class EvidenceItem(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=128)
    evidence_type: Literal["device", "sensor", "alarm", "work_order", "manual"]
    source_id: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=1, max_length=800)
    observed_at: datetime | None = None
    version: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("observed_at")
    @classmethod
    def observed_at_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("observed_at must include timezone information")
        return value


class DiagnosisDraft(StrictModel):
    """Model-writable diagnosis that can select evidence IDs but cannot write facts."""

    request_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    scope_status: Literal["in_scope", "out_of_scope", "needs_clarification"]
    risk_level: Literal["low", "medium", "high", "critical", "unknown"]
    summary: str = Field(min_length=1, max_length=1200)
    evidence_sufficient: bool
    likely_causes: list[LikelyCause] = Field(max_length=5)
    evidence_ids: list[str] = Field(max_length=20)
    recommended_actions: list[str] = Field(max_length=8)
    requires_human_review: bool
    limitations: list[str] = Field(max_length=8)

    @field_validator("recommended_actions", "limitations")
    @classmethod
    def nonempty_text_items(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 400 for value in values):
            raise ValueError("text items must be non-empty and at most 400 characters")
        return values

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence_ids must not contain blank values")
        if len(values) != len(set(values)):
            raise ValueError("evidence_ids must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_evidence_consistency(self) -> "DiagnosisDraft":
        selected_ids = set(self.evidence_ids)
        referenced_ids = {
            evidence_id
            for cause in self.likely_causes
            for evidence_id in cause.evidence_ids
        }
        if not referenced_ids.issubset(selected_ids):
            raise ValueError("likely_causes may reference only selected evidence_ids")
        if self.evidence_sufficient and not self.evidence_ids:
            raise ValueError("at least one evidence_id is required when evidence is sufficient")
        if not self.evidence_sufficient:
            if self.risk_level != "unknown":
                raise ValueError("risk_level must be unknown when evidence is insufficient")
            if not self.limitations:
                raise ValueError("limitations are required when evidence is insufficient")
        if self.risk_level in {"high", "critical"} and not self.requires_human_review:
            raise ValueError("high or critical risk requires human review")
        return self


class DiagnosisReport(StrictModel):
    """Program-finalized report whose evidence facts originate only from the registry."""

    request_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    scope_status: Literal["in_scope", "out_of_scope", "needs_clarification"]
    risk_level: Literal["low", "medium", "high", "critical", "unknown"]
    summary: str = Field(min_length=1, max_length=1200)
    evidence_sufficient: bool
    likely_causes: list[LikelyCause] = Field(max_length=5)
    evidence: list[EvidenceItem] = Field(max_length=20)
    recommended_actions: list[str] = Field(max_length=8)
    requires_human_review: bool
    limitations: list[str] = Field(max_length=8)

    @field_validator("recommended_actions", "limitations")
    @classmethod
    def nonempty_text_items(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 400 for value in values):
            raise ValueError("text items must be non-empty and at most 400 characters")
        return values

    @field_validator("evidence")
    @classmethod
    def unique_evidence(cls, values: list[EvidenceItem]) -> list[EvidenceItem]:
        ids = [item.evidence_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique")
        return values

    @model_validator(mode="after")
    def validate_evidence_consistency(self) -> "DiagnosisReport":
        evidence_ids = {item.evidence_id for item in self.evidence}
        referenced_ids = {
            evidence_id
            for cause in self.likely_causes
            for evidence_id in cause.evidence_ids
        }
        if not referenced_ids.issubset(evidence_ids):
            raise ValueError("likely_causes may reference only included evidence_ids")
        if self.evidence_sufficient and not self.evidence:
            raise ValueError("at least one evidence item is required when evidence is sufficient")
        if not self.evidence_sufficient:
            if self.risk_level != "unknown":
                raise ValueError("risk_level must be unknown when evidence is insufficient")
            if not self.limitations:
                raise ValueError("limitations are required when evidence is insufficient")
        if self.risk_level in {"high", "critical"} and not self.requires_human_review:
            raise ValueError("high or critical risk requires human review")
        return self
