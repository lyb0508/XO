"""Command-line entry point for one read-only industrial diagnosis.

CLI 入口（第四阶段起提供）：解析一次诊断请求的参数，组装模型、记忆、手册
检索与诊断 Graph，执行完整的"研判 ->（如遇高风险动作）人工审批 -> 报告"
流程，最后向 stdout 打印一份脱敏后的 JSON 报告。

数据流：argv -> argparse -> Settings -> ChatModel + 诊断 Graph -> invoke
（遇审批中断则在 stderr 上交互式收集人工决策，再用 Command(resume) 恢复）
-> stdout 输出 JSON 结果。

安全边界：
* 一切输出（成功报告与错误消息）都先经 redact_payload 脱敏再离开进程；
* 参数解析失败走机器可读的 JSON 错误路径（CliUsageError），不裸吐 traceback；
* 审批决策必须来自真实 stdin 输入，CLI 绝不伪造"批准"；
* stdin 无输入（EOF）时进程失败退出，被中断的线程仍留在 checkpointer 的
  生命周期内可供恢复，而不是默默替人拍板。

失败行为：任何异常都在 main 最外层捕获、脱敏后以 {"error": ...} 写到
stderr 并返回退出码 1；成功返回 0。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from uuid import uuid4

from app.config.settings import get_settings
from app.graphs.builder import GRAPH_RECURSION_LIMIT, build_diagnosis_graph
from app.models.factory import create_chat_model
from app.observability.tracing import RunContext, TraceMetadata, redact_payload, tracing_run
from app.schemas.approval import ApprovalDecision
from app.schemas.diagnostics import DiagnosisReport

AGENT_VERSION = "phase4-retrieval-memory"
DEFAULT_DEVICE_ID = "PUMP-003"
# 审批交互轮数硬上限：既限制用户反复输错的次数，也杜绝无限审批循环。
MAX_APPROVAL_ROUNDS = 3


class CliUsageError(ValueError):
    """受控的用法错误：交给 CLI 的安全 JSON 错误路径处理，而非裸 traceback。"""


class _JsonArgumentParser(argparse.ArgumentParser):
    """让参数解析失败保持机器可读，同时不改变正常的 --help 行为。

    argparse 默认在出错时直接打印 usage 并 SystemExit(2)；包装成异常后，
    main 才能统一捕获、脱敏并输出 JSON 错误。
    """

    def error(self, message: str) -> None:
        raise CliUsageError(str(redact_payload(message)))


def _parser() -> argparse.ArgumentParser:
    """声明 CLI 参数：问题必填；设备/请求/线程/会话标识可选且有默认行为。"""
    parser = _JsonArgumentParser(description="Run one read-only industrial diagnosis.")
    parser.add_argument("--question", required=True, help="Diagnostic question for the mock device.")
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID, help="Mock device identifier.")
    parser.add_argument("--request-id", help="Optional safe request identifier; generated when omitted.")
    parser.add_argument("--thread-id", help="Checkpoint thread identifier; generated when omitted.")
    parser.add_argument(
        "--session",
        dest="session_id",
        help="Session identifier enabling bounded short-term recall for this run.",
    )
    return parser


def _identifier(value: str | None) -> str:
    """空白输入回退为随机 UUID，确保 request_id / thread_id 始终非空可用。"""
    return value.strip() if value and value.strip() else str(uuid4())


def _read_line(prompt: str) -> str:
    """从 stdin 读取一行；读到 EOF（如管道关闭）时抛 EOFError 而非静默通过。

    提示语写往 stderr，保证 stdout 只承载最终的 JSON 报告，便于管道消费。
    """
    print(prompt, file=sys.stderr, end="", flush=True)
    line = sys.stdin.readline()
    if line == "":
        raise EOFError("no human input available for the approval decision")
    return line.strip()


def _prompt_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """从 stdin 收集一次结构化的审批决策。

    CLI 绝不伪造人工决策：没有可用输入时进程直接失败，被中断的线程仍留在
    checkpointer 生命周期内、随时可以恢复。approve/modify/reject 会被映射为
    approved/modified/rejected；选择 modify 时还需逐行录入修改后的动作
    （空行结束）；非法输入最多重试 MAX_APPROVAL_ROUNDS 次。
    """

    print(json.dumps({"approval_required": payload}, ensure_ascii=False), file=sys.stderr)
    last_error: Exception | None = None
    for _ in range(MAX_APPROVAL_ROUNDS):
        try:
            choice = _read_line("decision [approve|modify|reject]: ").lower()
            decided_by = _read_line("decided_by: ")
            reason = _read_line("reason: ")
            modified_actions: list[str] | None = None
            if choice == "modify":
                modified_actions = []
                print("modified actions (one per line, empty line to finish):", file=sys.stderr)
                while True:
                    action = sys.stdin.readline()
                    if action == "":
                        raise EOFError("no human input available for the approval decision")
                    if not action.strip():
                        break
                    modified_actions.append(action.strip())
            decision = ApprovalDecision.model_validate(
                {
                    "decision": {"approve": "approved", "modify": "modified", "reject": "rejected"}.get(choice),
                    "decided_by": decided_by,
                    "reason": reason,
                    "modified_actions": modified_actions,
                }
            )
            return decision.model_dump(mode="json")
        except (ValueError, KeyError) as error:
            last_error = error
            print(f"invalid approval input ({error}); please retry.", file=sys.stderr)
    raise CliUsageError(f"approval input was rejected too many times: {last_error}")


def _run_with_approval(graph: Any, initial_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """驱动 Graph 直到产出最终结果，途中每次审批中断都与真人交互后恢复。

    与 HTTP 同步端点的"自动拒绝"不同：CLI 场景确有真人在终端旁，所以这里
    真实等待人类输入；轮数受 MAX_APPROVAL_ROUNDS 硬性封顶，防止无限循环。
    """
    from langgraph.types import Command

    result = graph.invoke(initial_input, config=config)
    for _ in range(MAX_APPROVAL_ROUNDS):
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if not interrupts:
            return result
        payload = interrupts[0].value
        if not isinstance(payload, dict):
            raise RuntimeError("unexpected approval interrupt payload shape")
        resume_value = _prompt_decision(redact_payload(payload))
        result = graph.invoke(Command(resume=resume_value), config=config)
    raise CliUsageError("too many approval rounds")


def main(argv: list[str] | None = None) -> int:
    """一次完整 CLI 诊断的主流程，返回进程退出码（0 成功 / 1 失败）。

    组装顺序：Settings -> Trace 元数据 -> 模型 -> 会话记忆（传入 --session
    时才创建）-> 长期台账 -> 手册检索库 -> 诊断 Graph；整次 invoke 包裹在
    tracing_run 里，LangSmith 上能看到完整轨迹。
    """
    try:
        args = _parser().parse_args(argv)
        if not args.question.strip():
            raise CliUsageError("question must not be empty")
        # 空 session id 决不能流入 Graph 或记忆写入方：在此提前归一化，
        # 保证报告与会话副作用的行为一致。
        args.session_id = (args.session_id or "").strip() or None
        settings = get_settings()
        context = RunContext(
            request_id=_identifier(args.request_id),
            thread_id=_identifier(args.thread_id),
        )
        metadata = TraceMetadata.from_run_context(
            context,
            agent_version=AGENT_VERSION,
            environment=settings.environment,
            provider=settings.provider,
            model_alias=settings.model,
        )
        model = create_chat_model(settings)
        from langgraph.checkpoint.memory import InMemorySaver

        session_memory = None
        if args.session_id:
            from app.memory.session import SessionMemory

            session_memory = SessionMemory(max_turns=settings.session_memory_max_turns)
        from app.memory.ledger import LongTermLedger

        ledger = LongTermLedger(settings.memory_ledger_path)

        try:
            from app.retrieval.retriever import create_manual_store

            manual_store = create_manual_store(settings)
        except Exception as error:
            raise CliUsageError(
                f"manual retrieval store is unavailable: {redact_payload(str(error))}"
            ) from error

        graph = build_diagnosis_graph(
            model,
            structured_output_method=settings.structured_output_method,
            checkpointer=InMemorySaver(),
            manual_store=manual_store,
            manual_top_k=settings.manual_retrieval_top_k,
            manual_min_score=settings.manual_retrieval_min_score,
            session_memory=session_memory,
            ledger=ledger,
        )
        invoke_config = {
            "run_name": "diagnosis_graph",
            "tags": ["diagnosis_graph"],
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "configurable": {"thread_id": context.thread_id},
        }
        graph_input = {
            "request_id": context.request_id,
            "device_id": args.device_id,
            "question": args.question.strip(),
        }
        if args.session_id:
            graph_input["session_id"] = args.session_id
        with tracing_run(settings, metadata):
            result = _run_with_approval(graph, graph_input, invoke_config)
        if result.get("error") or not result.get("report"):
            safe_message = redact_payload(str(result.get("error", "diagnosis produced no report")))
            print(json.dumps({"error": safe_message}, ensure_ascii=False), file=sys.stderr)
            return 1
        report = DiagnosisReport.model_validate(result["report"])
        if session_memory is not None:
            session_memory.append_turn(
                args.session_id,
                question=args.question.strip(),
                device_id=args.device_id,
                risk_level=report.risk_level,
                summary=report.summary,
            )
        outcome = {
            "report": report.model_dump(mode="json"),
            "approval": result.get("approval"),
            "action_audit": result.get("action_audit"),
        }
        safe_outcome = redact_payload(outcome)
        print(json.dumps(safe_outcome, ensure_ascii=False))
        return 0
    except Exception as error:
        safe_message = redact_payload(str(error))
        print(json.dumps({"error": safe_message}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
