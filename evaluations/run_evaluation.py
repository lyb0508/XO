"""Runner: local-first evaluation with optional LangSmith upload.

Default mode runs entirely offline from LangSmith (``upload_results=False``);
``--upload`` requires a configured API key and creates a real Experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TARGET_METRICS = {
    "tool_selection": 0.90,
    "refusal_behavior": 0.90,
    "trajectory": 0.85,
}


def load_examples(path: str) -> list[dict[str, Any]]:
    """Load frozen dataset rows into langsmith Example objects.

    Expectations come only from this file; nothing is derived from app code.
    """

    import uuid
    from datetime import UTC, datetime as _datetime

    from langsmith.schemas import Example

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    now = _datetime.now(UTC)
    examples = []
    for row in data["examples"]:
        inputs = {
            "case_id": row["case_id"],
            "scenario": row["scenario"],
            **row["inputs"],
        }
        examples.append(
            Example(
                id=uuid.uuid5(uuid.NAMESPACE_URL, f"{data['name']}:{row['case_id']}"),
                dataset_id=uuid.uuid5(uuid.NAMESPACE_URL, data["name"]),
                inputs=inputs,
                outputs=row["expected"],
                created_at=now,
                modified_at=now,
            )
        )
    return examples


def summarize(
    results: Any,
    captured_outputs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate per-row evaluator scores into the phase metrics.

    Target outputs come from the capture evaluator because
    ``upload_results=False`` does not populate ``run.outputs``.
    """

    captured = captured_outputs or {}
    per_key: dict[str, list[float]] = defaultdict(list)
    refusal_scores: list[float] = []
    scenario_failures: dict[str, list[str]] = defaultdict(list)
    case_details: list[dict[str, Any]] = []
    for feedback_row in results:
        example = feedback_row["example"]
        case_id = example.inputs.get("case_id", "?")
        scenario = example.inputs.get("scenario", "?")
        row_outputs = captured.get(case_id, {})
        detail: dict[str, Any] = {
            "case_id": case_id,
            "scenario": scenario,
            "scope_status": row_outputs.get("scope_status"),
            "plan_evidence_types": row_outputs.get("plan_evidence_types"),
            "tool_source_types": row_outputs.get("tool_source_types"),
            "report_valid": row_outputs.get("report_valid"),
            "error": row_outputs.get("error"),
            "scores": {},
        }
        for result in feedback_row["evaluation_results"]["results"]:
            key = result.key
            if key == "output_capture":
                continue  # infrastructure channel, never a metric
            score = result.score if result.score is not None else 0.0
            per_key[key].append(score)
            detail["scores"][key] = score
            if key == "refusal_behavior" and result.comment != "not-applicable":
                refusal_scores.append(score)
            if score < 1.0:
                scenario_failures[scenario].append(f"{case_id}:{key}={score}")
        case_details.append(detail)
    metrics = {key: sum(values) / len(values) for key, values in sorted(per_key.items())}
    refusal_metric = (
        sum(refusal_scores) / len(refusal_scores) if refusal_scores else None
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": len(case_details),
        "metrics": metrics,
        "refusal_behavior_strict": refusal_metric,
        "targets": TARGET_METRICS,
        "target_status": {
            name: ("met" if metrics.get(name, 0.0) >= goal else "below-target")
            for name, goal in TARGET_METRICS.items()
            if metrics.get(name) is not None
        },
        "failures_by_scenario": {key: value[:10] for key, value in sorted(scenario_failures.items())},
        "cases_detail": case_details,
    }
    return report


def make_output_capture() -> tuple[Any, Any]:
    """Build a capturing evaluator plus its registry.

    ``upload_results=False`` does not populate ``run.outputs``, so the only
    reliable place to observe target outputs is inside an evaluator.
    """

    captured: dict[str, dict[str, Any]] = {}

    def capture_outputs(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        captured[str(inputs.get("case_id"))] = dict(outputs or {})
        return {"key": "output_capture", "score": 1.0, "comment": "not-a-metric"}

    return capture_outputs, captured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the phase-five evaluation suite.")
    parser.add_argument("--dataset", default="evaluations/dataset.json")
    parser.add_argument("--upload", action="store_true", help="Upload results to LangSmith.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N cases (smoke); 0 means all.",
    )
    parser.add_argument(
        "--report-file",
        default="tmp/eval_report.json",
        help="Where to write the full report including per-case details.",
    )
    args = parser.parse_args(argv)

    # Local runs must never touch LangSmith even if the machine carries stale
    # credentials; upload mode keeps whatever configuration the user has.
    if args.upload:
        import os

        if not (os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGSMITH_KEY", "")).strip():
            print(
                json.dumps({"error": "--upload requires LANGSMITH_API_KEY in the environment"}),
                file=sys.stderr,
            )
            return 2
    else:
        import os

        for env_key in (
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_API_KEY",
            "LANGCHAIN_API_KEY",
            "LANGSMITH_ENDPOINT",
            "LANGCHAIN_ENDPOINT",
            "LANGSMITH_PROJECT",
            "LANGSMITH_WORKSPACE_ID",
        ):
            os.environ.pop(env_key, None)
            os.environ.pop(env_key.lower(), None)
        import logging

        logging.getLogger("langsmith").setLevel(logging.CRITICAL)

    from langsmith import evaluate

    from evaluations.evaluators import ALL_EVALUATORS
    from evaluations.target import make_live_target

    examples = load_examples(args.dataset)
    if args.limit > 0:
        examples = examples[: args.limit]

    capture_evaluator, captured_outputs = make_output_capture()
    from app.config.settings import get_settings

    settings_snapshot = get_settings()
    results = evaluate(
        make_live_target(),
        data=examples,
        evaluators=[capture_evaluator, *ALL_EVALUATORS],
        upload_results=args.upload,
        experiment_prefix="phase5-diagnostic-eval",
        max_concurrency=1,
        metadata={
            "graph_version": "phase4-retrieval-memory",
            "prompt_version": "unversioned",
            "provider": settings_snapshot.provider,
            "model": settings_snapshot.model,
            "dataset_version": "1.0",
            "suite_size": len(examples),
        },
    )
    report = summarize(results, captured_outputs)
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {key: value for key, value in report.items() if key != "cases_detail"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"full report written to {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
