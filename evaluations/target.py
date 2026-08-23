"""Evaluation target: run one diagnosis and normalize observable outcomes.

本模块回答“怎么跑一次完整诊断”。评测体系里只有这里知道如何驱动 LangGraph
诊断 Graph；其余 Evaluator 都只面对本模块归一化后的输出字典，因此它们保持为
纯函数，可以在不启动 Graph、不连模型的情况下离线测试。

数据流：数据集一行输入 → 构建真实 Graph 并 invoke → 遇到 Interrupt 时按
scope 自动批准/拒绝并恢复 → 把最终状态投影成小而稳定的字段集合。
任何环节抛异常都会被包装成带 ``error`` 字段的结果，而不是让整个评测中断，
这样汇总统计能把崩溃明确计为失败，而不是悄悄漏掉一个样本。
"""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings


def make_live_target():
    """Build a target that runs the real graph against the configured model.

    工厂函数：返回一个可直接交给 langsmith ``evaluate`` 的目标函数。目标函数
    接收数据集一行的 ``inputs``（device_id、question 等），跑完一次真实 Graph
    后返回归一化结果。

    为什么捕获所有异常：评测中某个样本崩溃（模型超时、Schema 校验失败等）
    不应拖垮整场 Experiment。这里把异常转成一个全空字段加 ``error`` 说明的
    可比较结果，Evaluator 与汇总报告就能把它当作显式失败计数。
    注意：这层兜底只服务评测，不代表生产路径可以吞异常。
    """

    settings = get_settings()

    def live_target(inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            return _run_once(inputs, settings)
        except Exception as error:
            return {
                "scope_status": None,
                "plan_evidence_types": [],
                "tool_source_types": [],
                "report_valid": False,
                "risk_level": None,
                "evidence_sufficient": None,
                "evidence_count": 0,
                "requires_human_review": None,
                "limitations_count": 0,
                "recommended_actions": [],
                "approval_decision": None,
                "error": f"{error.__class__.__name__}: {error}",
            }

    return live_target


def _run_once(inputs: dict[str, Any], settings: Any) -> dict[str, Any]:
    """跑一次完整诊断并返回归一化结果；异常直接向上抛，由 live_target 兜底。

    延迟导入（函数体内 import）是有意的：评测入口希望先完成环境清理与
    参数解析，再加载较重的 langgraph 与业务模块。
    """

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from app.graphs.builder import GRAPH_RECURSION_LIMIT, build_diagnosis_graph
    from app.memory.ledger import LongTermLedger
    from app.models.factory import create_chat_model
    from app.retrieval.retriever import create_manual_store

    model = create_chat_model(settings)
    graph = build_diagnosis_graph(
        model,
        structured_output_method=settings.structured_output_method,
        checkpointer=InMemorySaver(),
        manual_store=create_manual_store(settings),
        manual_top_k=settings.manual_retrieval_top_k,
        manual_min_score=settings.manual_retrieval_min_score,
        # 评测绝不能碰生产长期记忆：ledger 写入一次性临时文件，随进程消亡。
        ledger=LongTermLedger(_throwaway_ledger_path()),
    )
    config = {
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "configurable": {"thread_id": f"eval-{inputs.get('case_id', 'unknown')}"},
    }
    graph_input = {
        "request_id": f"eval-{inputs.get('case_id', 'unknown')}",
        "device_id": inputs["device_id"],
        "question": inputs["question"],
    }
    result = graph.invoke(graph_input, config=config)
    # 最多恢复三次审批中断：评测运行器代替人工审批，in_scope 自动批准、
    # 其余 scope 自动拒绝，保证非 in_scope 报告永远拿不到已批准的动作。
    for _ in range(3):
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if not interrupts:
            break
        plan_scope = (result.get("query_plan") or {}).get("scope_status")
        if plan_scope == "in_scope":
            resume_value = {
                "decision": "approved",
                "decided_by": "eval-runner",
                "reason": "automatic approval during evaluation",
            }
        else:
            # 即使在评测夹具里，非 in_scope 的报告也绝不能拿到 approved 动作。
            resume_value = {
                "decision": "rejected",
                "decided_by": "eval-runner",
                "reason": f"evaluation auto-reject for scope={plan_scope}",
            }
        result = graph.invoke(Command(resume=resume_value), config=config)
    return _normalize(result)


def _throwaway_ledger_path() -> str:
    """One-shot ledger location inside the ignored tmp directory.

    返回系统临时目录下一个带时间戳的 ledger 文件路径。每次评测运行都拿到
    独立文件：既避免并发运行互相污染，也保证评测结束后无需清理生产数据。
    """

    import tempfile
    from pathlib import Path

    handle = Path(tempfile.gettempdir()) / "industrial-agent-eval-ledgers"
    handle.mkdir(parents=True, exist_ok=True)
    return str(handle / f"eval-{datetime_now_stamp()}.jsonl")


def datetime_now_stamp() -> str:
    """生成 UTC 时间戳字符串，用于给一次性 ledger 文件命名。"""

    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _normalize(result: dict[str, Any]) -> dict[str, Any]:
    """Project graph state onto the small surface evaluators may inspect.

    把庞大的 Graph 最终状态投影成 Evaluator 唯一允许查看的字段集合：
    计划的 scope 与证据类型、工具实际触达的来源类型、报告有效性与风险字段、
    审批决定以及错误说明。投影面刻意保持小而稳定——Evaluator 依赖的面越窄，
    生产实现越容易重构而不破坏评测契约。
    """

    plan = result.get("query_plan") or {}
    payloads = result.get("tool_payloads") or []
    source_types = sorted({str(item.get("source_type")) for item in payloads if item.get("source_type")})
    report = result.get("report")
    return {
        "scope_status": plan.get("scope_status"),
        "plan_evidence_types": sorted(plan.get("requested_evidence_types", [])),
        "tool_source_types": source_types,
        "report_valid": isinstance(report, dict) and bool(report),
        "risk_level": (report or {}).get("risk_level"),
        "evidence_sufficient": (report or {}).get("evidence_sufficient"),
        "evidence_count": len((report or {}).get("evidence", []) or []),
        "requires_human_review": (report or {}).get("requires_human_review"),
        "limitations_count": len((report or {}).get("limitations", []) or []),
        "recommended_actions": (report or {}).get("recommended_actions", []),
        "approval_decision": (result.get("approval") or {}).get("decision"),
        "error": result.get("error"),
    }


def make_offline_target(plan_by_scenario: dict[str, dict[str, Any]]):
    """Build a deterministic pipeline-smoke target for offline runs.

    确定性的冒烟目标：不启动真实模型与 Graph，直接按 scenario 查表返回固定
    输出。当前测试未使用它，但作为文档化的接缝保留——需要在不依赖外部模型
    的情况下验证“运行器 → Evaluator → 汇总”整条管线时，用这个目标替换
    live target 即可，评测契约不受影响。
    """

    def offline_target(inputs: dict[str, Any]) -> dict[str, Any]:
        plan = plan_by_scenario.get(inputs.get("scenario", ""), {"requested_evidence_types": []})
        return {
            "scope_status": plan.get("scope_status", "in_scope"),
            "plan_evidence_types": sorted(plan.get("requested_evidence_types", [])),
            "tool_source_types": [],
            "report_valid": True,
            "risk_level": "medium",
            "evidence_sufficient": True,
            "evidence_count": 1,
            "requires_human_review": False,
            "limitations_count": 0,
            "recommended_actions": [],
            "approval_decision": None,
            "error": None,
        }

    return offline_target
