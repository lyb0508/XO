"""Human-in-the-loop approval contracts for the phase-three gate.

The proposed action is derived by the program from the finalized report; the
model never defines business actions. A human decision is required before any
side effect, and every decision leaves an auditable record.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.diagnostics import DiagnosisReport, StrictModel
from app.observability.tracing import _SAFE_VALUE

CONTROLLED_ACTION_TYPES = ("schedule_maintenance",)


class ProposedAction(StrictModel):
    """The single controlled action type this project can propose."""

    action_type: Literal["schedule_maintenance"]
    request_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=400)


class ApprovalDecision(StrictModel):
    """Structured human decision consumed after an interrupt resumes."""

    decision: Literal["approved", "modified", "rejected"]
    decided_by: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=400)
    # Aligned with DiagnosisReport.recommended_actions so a modified decision
    # can never produce a report that fails final validation.
    modified_actions: list[str] | None = Field(default=None, max_length=8)

    @field_validator("decided_by")
    @classmethod
    def decided_by_is_safe(cls, value: str) -> str:
        if not _SAFE_VALUE.fullmatch(value):
            raise ValueError("decided_by may contain only letters, digits, . _ : or -")
        return value

    @field_validator("modified_actions")
    @classmethod
    def modified_actions_are_valid(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if any(not action.strip() or len(action) > 400 for action in values):
            raise ValueError("modified actions must be non-empty and at most 400 characters")
        return values

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "ApprovalDecision":
        if self.decision == "modified":
            if not self.modified_actions:
                raise ValueError("a modified decision must provide modified_actions")
        elif self.modified_actions is not None:
            raise ValueError("modified_actions are only allowed for a modified decision")
        return self


def derive_proposed_action(report: dict[str, Any]) -> ProposedAction:
    """Derive the controlled action from a finalized report mapping.

    Only program-owned fields feed this derivation; model text never widens
    what the action can do.
    """

    validated = report if isinstance(report, DiagnosisReport) else DiagnosisReport.model_validate(report)
    return ProposedAction(
        action_type="schedule_maintenance",
        request_id=validated.request_id,
        device_id=validated.device_id,
        reason=f"risk_level={validated.risk_level}; {validated.summary[:300]}",
    )
