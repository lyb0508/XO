"""Runner: local-first evaluation with optional LangSmith upload.

本模块是评测运行器，把前三个环节串成一条流水线：
加载固定数据集（load_examples）→ 用目标函数跑每个样本（make_live_target）
→ 六个确定性 Evaluator 打分 → summarize 汇总成分阶段指标报告。

默认模式（不带 ``--upload``）完全离线于 LangSmith 运行：启动前主动清除所有
LangSmith 环境变量并压掉其日志，即使机器上残留旧凭据也不会外发任何数据。
只有显式传入 ``--upload`` 且环境配置了 LANGSMITH_API_KEY 时，才会创建真实
Experiment 并上传结果；缺 key 时以退出码 2 明确报错，而不是静默降级。

完整逐样本报告写入 ``tmp/eval_report.json``，终端只打印去掉明细的摘要。
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

    把 dataset.json 的每一行转成 langsmith ``Example``：inputs 是给目标函数
    的输入，outputs 就是手写的期望值。Example 的 id 与 dataset_id 都用
    ``uuid5`` 从数据集名和 case_id 确定性生成——同一份数据集重复加载得到
    相同的标识，便于跨 Experiment 对齐比较。

    期望值只来自这个 JSON 文件，绝不从 app 代码推导，保证评测独立性。
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

    把 langsmith 返回的逐行评测结果聚合成阶段报告：每个 Evaluator 维度的
    平均分、严格口径的拒答分（剔除 not-applicable 样本）、按场景归类的失败
    清单，以及与 TARGET_METRICS 目标线的达标状态。

    逐样本的目标函数输出来自 capture evaluator 的旁路记录——本地模式下
    ``run.outputs`` 不会被填充（见 make_output_capture），这是唯一可靠的
    观测位置。
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
                continue  # 基础设施通道，永远不算指标
            score = result.score if result.score is not None else 0.0
            per_key[key].append(score)
            detail["scores"][key] = score
            # 拒答指标用严格口径：not-applicable 样本不计入均值
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

    构造一个“旁路记录器”形态的 Evaluator：它对每个样本原样记下目标函数的
    输出并恒返 1 分，本身不参与任何指标（summarize 会跳过 output_capture
    键）。之所以需要它：本地模式下 ``upload_results=False`` 不会填充
    ``run.outputs``，Evaluator 是唯一稳定能拿到目标输出的位置。

    返回值是二元组 ``(capture 函数, captured 字典)``——运行器持有这个字典
    引用，评测结束后把它交给 summarize 生成逐样本明细。
    """

    captured: dict[str, dict[str, Any]] = {}

    def capture_outputs(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        captured[str(inputs.get("case_id"))] = dict(outputs or {})
        return {"key": "output_capture", "score": 1.0, "comment": "not-a-metric"}

    return capture_outputs, captured


def main(argv: list[str] | None = None) -> int:
    """评测入口：解析参数、清理环境、跑完整流水线并写出报告。

    返回 0 表示成功；``--upload`` 缺少 LANGSMITH_API_KEY 时返回 2。
    环境处理分两条路：上传模式保留用户现有配置；本地模式主动删除所有
    LangSmith 相关变量，防止残留凭据造成意外的数据外发。
    """
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

    # 本地运行绝不触碰 LangSmith：即使机器上残留旧凭据也要清掉；
    # 上传模式则保留用户自己的配置不动。
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
