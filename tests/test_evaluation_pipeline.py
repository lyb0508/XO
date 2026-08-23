"""Offline pipeline smoke tests for the evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluations.evaluators import (
    final_contract_score,
    refusal_behavior_score,
    security_score,
    tool_selection_score,
    trajectory_score,
)
from evaluations.run_evaluation import load_examples, summarize

DATASET_PATH = Path(__file__).resolve().parents[1] / "evaluations" / "dataset.json"


def _outputs(**overrides: object) -> dict:
    base = {
        "scope_status": "in_scope",
        "plan_evidence_types": ["sensor"],
        "tool_source_types": ["mock_sensor_store"],
        "report_valid": True,
        "risk_level": "medium",
        "evidence_sufficient": True,
        "evidence_count": 3,
        "requires_human_review": False,
        "limitations_count": 0,
        "recommended_actions": [],
        "approval_decision": None,
        "error": None,
    }
    base.update(overrides)
    return base


def test_dataset_structure_and_scenario_coverage() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    examples = data["examples"]
    assert len(examples) == 50, "the frozen suite must keep its contracted size"
    scenarios = {row["scenario"] for row in examples}
    assert scenarios == set(data["scenarios"])
    counts: dict[str, int] = {}
    for row in examples:
        counts[row["scenario"]] = counts.get(row["scenario"], 0) + 1
        assert row["case_id"] and row["inputs"]["question"].strip()
        expected = row["expected"]
        assert expected["scope_status"] in {"in_scope", "needs_clarification", "out_of_scope"}
        assert isinstance(expected["evidence_types"], list)
    assert counts["vibration_normal"] >= 10
    assert counts["multi_tool"] >= 6
    assert counts["insufficient"] + counts["out_of_scope"] >= 10
    assert counts["injection"] >= 5 and counts["unauthorized"] >= 5


def test_example_ids_are_stable_across_loads() -> None:
    first = {str(example.id) for example in load_examples(str(DATASET_PATH))}
    second = {str(example.id) for example in load_examples(str(DATASET_PATH))}
    assert first == second and len(first) == 50


def test_tool_selection_scores_match_subset_and_mismatch() -> None:
    reference = {"evidence_types": ["sensor", "manual"]}
    exact = tool_selection_score({}, {"plan_evidence_types": ["manual", "sensor"]}, reference)
    subset = tool_selection_score({}, {"plan_evidence_types": ["sensor"]}, reference)
    wrong = tool_selection_score({}, {"plan_evidence_types": ["device"]}, reference)
    assert (exact["score"], subset["score"], wrong["score"]) == (1.0, 0.5, 0.0)


def test_refusal_evaluator_marks_in_scope_not_applicable() -> dict:
    result = refusal_behavior_score(
        {}, {"scope_status": "in_scope", "plan_evidence_types": ["sensor"]},
        {"scope_status": "in_scope"},
    )
    assert result["comment"] == "not-applicable"
    refused = refusal_behavior_score(
        {}, {"scope_status": "out_of_scope", "plan_evidence_types": []},
        {"scope_status": "out_of_scope"},
    )
    assert refused["score"] == 1.0
    partial = refusal_behavior_score(
        {}, {"scope_status": "needs_clarification", "plan_evidence_types": ["sensor"]},
        {"scope_status": "needs_clarification"},
    )
    assert partial["score"] == 0.5
    return result


def test_trajectory_and_security_and_contract() -> None:
    reference = {
        "must_include": ["mock_sensor_store"],
        "must_exclude": ["mock_asset_registry"],
        "expect_report": True,
        "evidence_types": [],
    }
    ok = trajectory_score({}, {"tool_source_types": ["mock_sensor_store"]}, reference)
    bad = trajectory_score({}, {"tool_source_types": ["mock_asset_registry"]}, reference)
    assert (ok["score"], bad["score"]) == (1.0, 0.0)

    pure_attack = security_score({}, {"plan_evidence_types": []}, reference)
    leaky = security_score({}, {"plan_evidence_types": ["device"]}, reference)
    assert (pure_attack["score"], leaky["score"]) == (1.0, 0.0)

    good_contract = final_contract_score({}, {"report_valid": True, "error": None}, reference)
    failed_contract = final_contract_score({}, {"report_valid": False, "error": "x"}, reference)
    assert (good_contract["score"], failed_contract["score"]) == (1.0, 0.0)


def test_summarize_excludes_not_applicable_from_refusal_metric() -> None:
    import datetime as dt
    import uuid

    from langsmith.evaluation import EvaluationResults
    from langsmith.schemas import Example, Feedback

    def make_feedback_row(case_id: str, scenario: str, scores: dict[str, float], comments: dict[str, str]):
        now = dt.datetime.now(dt.UTC)
        example = Example(
            id=uuid.uuid4(), dataset_id=uuid.uuid4(),
            inputs={"case_id": case_id, "scenario": scenario}, outputs={},
            created_at=now, modified_at=now,
        )
        results = []
        for key, score in scores.items():
            now_feedback = dt.datetime.now(dt.UTC)
            results.append(
                Feedback(
                    id=uuid.uuid4(),
                    created_at=now_feedback,
                    modified_at=now_feedback,
                    run_id=uuid.uuid4(),
                    trace_id=uuid.uuid4(),
                    key=key,
                    score=score,
                    comment=comments.get(key, ""),
                )
            )
        return {
            "example": example,
            "evaluation_results": EvaluationResults(results=results),
            "run": None,
        }

    rows = [
        make_feedback_row("c1", "vibration_normal", {"tool_selection": 1.0, "refusal_behavior": 1.0}, {"refusal_behavior": "not-applicable"}),
        make_feedback_row("c2", "injection", {"tool_selection": 0.0, "refusal_behavior": 1.0}, {"refusal_behavior": "refused without tool calls"}),
        make_feedback_row("c3", "out_of_scope", {"tool_selection": 0.0, "refusal_behavior": 0.0}, {"refusal_behavior": "proceeded"}),
    ]
    report = summarize(rows)
    assert report["metrics"]["tool_selection"] == pytest.approx(1 / 3)
    assert report["refusal_behavior_strict"] == pytest.approx(0.5)
    assert report["target_status"]["tool_selection"] == "below-target"
