"""Command-line entry point for one read-only industrial diagnosis."""

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

AGENT_VERSION = "phase3-approval"
DEFAULT_DEVICE_ID = "PUMP-003"
MAX_APPROVAL_ROUNDS = 3


class CliUsageError(ValueError):
    """A controlled parse failure that can use the CLI's safe JSON error path."""


class _JsonArgumentParser(argparse.ArgumentParser):
    """Keep parse failures machine-readable without changing normal --help behavior."""

    def error(self, message: str) -> None:
        raise CliUsageError(str(redact_payload(message)))


def _parser() -> argparse.ArgumentParser:
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
    return value.strip() if value and value.strip() else str(uuid4())


def _read_line(prompt: str) -> str:
    print(prompt, file=sys.stderr, end="", flush=True)
    line = sys.stdin.readline()
    if line == "":
        raise EOFError("no human input available for the approval decision")
    return line.strip()


def _prompt_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Collect one structured approval decision from stdin.

    The CLI never fabricates a human decision: when no input is available the
    process fails and the thread stays resumable within its checkpointer's
    lifetime instead.
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
    try:
        args = _parser().parse_args(argv)
        if not args.question.strip():
            raise CliUsageError("question must not be empty")
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
        ledger = None
        try:
            from app.memory.ledger import LongTermLedger

            ledger = LongTermLedger(settings.memory_ledger_path)
        except Exception:
            ledger = None

        from app.retrieval.retriever import create_manual_store

        graph = build_diagnosis_graph(
            model,
            structured_output_method=settings.structured_output_method,
            checkpointer=InMemorySaver(),
            manual_store=create_manual_store(settings),
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
