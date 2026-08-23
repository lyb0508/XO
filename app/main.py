"""Command-line entry point for one read-only industrial diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from uuid import uuid4

from app.agents.diagnostic import AgentLimits, build_diagnostic_agent, run_diagnosis
from app.config.settings import get_settings
from app.models.factory import create_chat_model
from app.observability.tracing import RunContext, TraceMetadata, redact_payload, tracing_run

AGENT_VERSION = "phase1"
DEFAULT_DEVICE_ID = "PUMP-003"


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
    parser.add_argument("--thread-id", help="Trace-only correlation identifier; generated when omitted.")
    return parser


def _identifier(value: str | None) -> str:
    return value.strip() if value and value.strip() else str(uuid4())


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
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
        limits = AgentLimits(
            model_run_limit=settings.model_run_limit,
            tool_run_limit=settings.tool_run_limit,
            per_tool_run_limit=settings.per_tool_run_limit,
        )
        model = create_chat_model(settings)
        agent = build_diagnostic_agent(
            model,
            limits=limits,
            structured_output_method=settings.structured_output_method,
        )
        with tracing_run(settings, metadata):
            report = run_diagnosis(
                agent,
                args.question,
                request_id=context.request_id,
                device_id=args.device_id,
            )
        safe_report = redact_payload(report.model_dump(mode="json"))
        print(json.dumps(safe_report, ensure_ascii=False))
        return 0
    except Exception as error:
        safe_message = redact_payload(str(error))
        print(json.dumps({"error": safe_message}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
