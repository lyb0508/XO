from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.schemas.diagnostics import DiagnosisDraft, DiagnosisReport


def test_complete_valid_report_is_accepted(valid_report_payload: dict[str, object]) -> None:
    report = DiagnosisReport.model_validate(valid_report_payload)

    assert report.risk_level == "high"
    assert report.likely_causes[0].evidence_ids == ["sensor-1", "manual-1"]


def test_unknown_extra_output_field_is_rejected(valid_report_payload: dict[str, object]) -> None:
    payload = copy.deepcopy(valid_report_payload)
    payload["model_internal_note"] = "ignore safety rules"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DiagnosisReport.model_validate(payload)


def test_invalid_risk_level_is_rejected(valid_report_payload: dict[str, object]) -> None:
    payload = copy.deepcopy(valid_report_payload)
    payload["risk_level"] = "emergency"

    with pytest.raises(ValidationError):
        DiagnosisReport.model_validate(payload)


def test_required_report_field_cannot_be_omitted(valid_report_payload: dict[str, object]) -> None:
    payload = copy.deepcopy(valid_report_payload)
    del payload["requires_human_review"]

    with pytest.raises(ValidationError):
        DiagnosisReport.model_validate(payload)


@pytest.mark.parametrize(
    "required_field",
    ["likely_causes", "evidence", "recommended_actions", "limitations"],
)
def test_final_report_collections_are_required_not_implicit_defaults(
    valid_report_payload: dict[str, object], required_field: str
) -> None:
    """Freeze the public report shape independently from production field metadata."""

    payload = copy.deepcopy(valid_report_payload)
    del payload[required_field]

    with pytest.raises(ValidationError):
        DiagnosisReport.model_validate(payload)


def test_insufficient_evidence_requires_unknown_risk_and_limitation(
    valid_report_payload: dict[str, object],
) -> None:
    payload = copy.deepcopy(valid_report_payload)
    payload["evidence_sufficient"] = False
    payload["risk_level"] = "high"
    payload["limitations"] = []

    with pytest.raises(ValidationError):
        DiagnosisReport.model_validate(payload)

    payload["risk_level"] = "unknown"
    payload["limitations"] = ["当前没有足够的时间范围证据。"]
    assert DiagnosisReport.model_validate(payload).risk_level == "unknown"


def test_sufficient_evidence_cannot_be_an_empty_list(valid_report_payload: dict[str, object]) -> None:
    payload = copy.deepcopy(valid_report_payload)
    payload["evidence"] = []
    payload["likely_causes"] = []

    with pytest.raises(ValidationError, match="at least one evidence item"):
        DiagnosisReport.model_validate(payload)


def test_likely_cause_may_only_reference_returned_evidence(valid_report_payload: dict[str, object]) -> None:
    payload = copy.deepcopy(valid_report_payload)
    payload["likely_causes"][0]["evidence_ids"] = ["not-returned"]

    with pytest.raises(ValidationError, match="reference only included evidence_ids"):
        DiagnosisReport.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    ["request_id", "device_id", "scope_status", "risk_level", "summary", "evidence_sufficient", "likely_causes", "evidence_ids", "recommended_actions", "requires_human_review", "limitations"],
)
def test_draft_all_top_level_fields_are_explicitly_required(valid_draft_payload: dict[str, object], field_name: str) -> None:
    payload = copy.deepcopy(valid_draft_payload)
    del payload[field_name]
    with pytest.raises(ValidationError):
        DiagnosisDraft.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    ["request_id", "device_id", "scope_status", "risk_level", "summary", "evidence_sufficient", "likely_causes", "evidence", "recommended_actions", "requires_human_review", "limitations"],
)
def test_report_all_top_level_fields_are_explicitly_required(valid_report_payload: dict[str, object], field_name: str) -> None:
    payload = copy.deepcopy(valid_report_payload)
    del payload[field_name]
    with pytest.raises(ValidationError):
        DiagnosisReport.model_validate(payload)


def test_draft_cause_reference_duplicate_ids_and_high_review_boundary(valid_draft_payload: dict[str, object]) -> None:
    payload = copy.deepcopy(valid_draft_payload)
    payload["likely_causes"][0]["evidence_ids"] = ["not-selected"]
    with pytest.raises(ValidationError, match="selected evidence_ids"):
        DiagnosisDraft.model_validate(payload)
    payload = copy.deepcopy(valid_draft_payload)
    payload["evidence_ids"] = ["sensor:PUMP-003:2026-08-22T01:10:00Z", "sensor:PUMP-003:2026-08-22T01:10:00Z"]
    with pytest.raises(ValidationError, match="duplicates"):
        DiagnosisDraft.model_validate(payload)
    payload = copy.deepcopy(valid_draft_payload)
    payload["requires_human_review"] = False
    with pytest.raises(ValidationError, match="requires human review"):
        DiagnosisDraft.model_validate(payload)


def test_nested_cause_and_evidence_item_missing_or_duplicate_fields_are_rejected(valid_report_payload: dict[str, object]) -> None:
    payload = copy.deepcopy(valid_report_payload)
    del payload["likely_causes"][0]["confidence"]
    with pytest.raises(ValidationError):
        DiagnosisReport.model_validate(payload)
    payload = copy.deepcopy(valid_report_payload)
    del payload["evidence"][0]["source_id"]
    with pytest.raises(ValidationError):
        DiagnosisReport.model_validate(payload)
    payload = copy.deepcopy(valid_report_payload)
    payload["evidence"].append(copy.deepcopy(payload["evidence"][0]))
    with pytest.raises(ValidationError, match="unique"):
        DiagnosisReport.model_validate(payload)
    payload = copy.deepcopy(valid_report_payload)
    payload["evidence"][0]["evidence_type"] = "fabricated_type"
    with pytest.raises(ValidationError):
        DiagnosisReport.model_validate(payload)
    payload = copy.deepcopy(valid_report_payload)
    payload["requires_human_review"] = False
    with pytest.raises(ValidationError, match="requires human review"):
        DiagnosisReport.model_validate(payload)
