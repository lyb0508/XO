from __future__ import annotations

import io
import json
import re

import pytest

from app import main as cli
from app.config.settings import Settings
from app.schemas.diagnostics import DiagnosisReport


def _report() -> DiagnosisReport:
    return DiagnosisReport.model_validate(
        {
            "request_id": "request-001",
            "device_id": "PUMP-003",
            "scope_status": "out_of_scope",
            "risk_level": "unknown",
            "summary": "请求不在范围内。",
            "evidence_sufficient": False,
            "likely_causes": [],
            "evidence": [],
            "recommended_actions": [],
            "requires_human_review": False,
            "limitations": ["这是离线 CLI 注入测试。"],
        }
    )


def test_cli_prints_exactly_one_outcome_json_to_stdout_and_passes_structured_method(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(
            _env_file=None,
            tracing_enabled=False,
            provider="deepseek",
            model="deepseek-v4-flash",
            structured_output_method="function_calling",
        ),
    )
    monkeypatch.setattr(cli, "create_chat_model", lambda settings: object())
    received: dict[str, object] = {}

    def build_stub(model, **kwargs):
        from types import SimpleNamespace

        received["method"] = kwargs["structured_output_method"]
        assert kwargs.get("checkpointer") is not None, "the CLI must always run with a checkpointer"
        assert kwargs.get("manual_store") is not None, "the CLI must enable manual retrieval"
        return SimpleNamespace(
            invoke=lambda state, config: {
                "report": _report().model_dump(mode="json"),
                "error": "",
                "configurable_thread_id": config["configurable"]["thread_id"],
            }
        )

    monkeypatch.setattr(cli, "build_diagnosis_graph", build_stub)

    exit_code = cli.main(["--question", "无关问题", "--request-id", "request-001", "--thread-id", "thread-001"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    expected = {
        "report": _report().model_dump(mode="json"),
        "approval": None,
        "action_audit": None,
    }
    assert json.loads(captured.out) == expected
    assert captured.out.count("\n") == 1
    assert received == {"method": "function_calling"}


def test_cli_fail_closed_branch_prints_redacted_error_to_stderr(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, tracing_enabled=False))
    monkeypatch.setattr(cli, "create_chat_model", lambda settings: object())
    monkeypatch.setattr(
        cli,
        "build_diagnosis_graph",
        lambda model, **kwargs: SimpleNamespace(
            invoke=lambda state, config: {
                "report": None,
                "error": "query_sensor_history: Authorization: Bearer top-secret",
            }
        ),
    )

    exit_code = cli.main(["--question", "研判振动", "--request-id", "request-001"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "top-secret" not in captured.err
    assert json.loads(captured.err)["error"].count("[REDACTED") >= 1


def _stderr_error(captured) -> dict:
    """Extract the final redacted error object from mixed stderr output."""

    matches = re.findall(r'\{"error".*?\}', captured.err)
    assert matches, captured.err
    return json.loads(matches[-1])


def test_cli_approval_flow_approves_with_audit(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    report_payload = _report().model_dump(mode="json")
    calls: list[object] = []

    def build_stub(model, **kwargs):
        graph = SimpleNamespace()
        graph.invoke = lambda command, config: calls.append(command) or (
            {
                "__interrupt__": [SimpleNamespace(value={"proposed_action": {"action_type": "schedule_maintenance"}})],
            }
            if len(calls) == 1
            else {
                "report": report_payload,
                "approval": {
                    "decision": "approved",
                    "decided_by": "duty-officer",
                    "modified_actions": None,
                },
                "action_audit": {"status": "executed", "ticket_id": "MNT-request-001"},
            }
        )

        return graph

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, tracing_enabled=False))
    monkeypatch.setattr(cli, "create_chat_model", lambda settings: object())
    monkeypatch.setattr(cli, "build_diagnosis_graph", build_stub)
    monkeypatch.setattr("sys.stdin", io.StringIO("approve\nduty-officer\n现场已确认\n"))

    exit_code = cli.main(["--question", "研判振动", "--request-id", "request-001"])
    captured = capsys.readouterr()

    assert exit_code == 0
    outcome = json.loads(captured.out)
    assert outcome["approval"]["decision"] == "approved"
    assert outcome["action_audit"]["ticket_id"] == "MNT-request-001"


def test_cli_modify_flow_rewrites_report_actions(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    report_payload = _report().model_dump(mode="json")
    calls: list[object] = []

    def build_stub(model, **kwargs):
        graph = SimpleNamespace()

        def invoke(command, config):
            calls.append(command)
            if len(calls) == 1:
                return {
                    "__interrupt__": [SimpleNamespace(value={"proposed_action": {"action_type": "schedule_maintenance"}})]
                }
            assert command.resume["decision"] == "modified"
            return {
                "report": {**report_payload, "recommended_actions": ["复测后再定检修。"]},
                "approval": command.resume,
                "action_audit": {"status": "executed", "ticket_id": "MNT-request-001"},
            }

        graph.invoke = invoke
        return graph

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, tracing_enabled=False))
    monkeypatch.setattr(cli, "create_chat_model", lambda settings: object())
    monkeypatch.setattr(cli, "build_diagnosis_graph", build_stub)
    monkeypatch.setattr("sys.stdin", io.StringIO("modify\nofficer-2\n按复测结果决定。\n复测振动。\n\n"))

    exit_code = cli.main(["--question", "研判振动", "--request-id", "request-001"])
    captured = capsys.readouterr()

    assert exit_code == 0
    outcome = json.loads(captured.out)
    assert outcome["approval"]["modified_actions"] == ["复测振动。"]
    assert outcome["report"]["recommended_actions"] == ["复测后再定检修。"]


def test_cli_eof_during_approval_fails_without_fabricating_a_decision(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    def build_stub(model, **kwargs):
        return SimpleNamespace(
            invoke=lambda command, config: {
                "__interrupt__": [SimpleNamespace(value={"proposed_action": {"action_type": "schedule_maintenance"}})]
            }
        )

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, tracing_enabled=False))
    monkeypatch.setattr(cli, "create_chat_model", lambda settings: object())
    monkeypatch.setattr(cli, "build_diagnosis_graph", build_stub)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    exit_code = cli.main(["--question", "研判振动", "--request-id", "request-001"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "no human input" in _stderr_error(captured)["error"]
    assert '"report"' not in captured.out


def test_cli_error_is_nonzero_and_redacts_stderr(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, tracing_enabled=False))

    def raises_secret(settings):
        raise RuntimeError("Authorization: Bearer top-secret alice@example.com 13800138000")

    monkeypatch.setattr(cli, "create_chat_model", raises_secret)

    exit_code = cli.main(["--question", "test", "--request-id", "request-001", "--thread-id", "thread-001"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "top-secret" not in captured.err
    assert "alice@example.com" not in captured.err
    assert "13800138000" not in captured.err
    assert json.loads(captured.err)["error"].count("[REDACTED") >= 1


@pytest.mark.parametrize("argv", [[], ["--question", "x", "--unknown"], ["--question", "   "]])
def test_cli_parse_errors_are_one_line_json_stderr(argv, capsys) -> None:
    exit_code = cli.main(argv)
    captured = capsys.readouterr()
    assert exit_code == 1 and captured.out == ""
    assert captured.err.count("\n") == 1
    assert isinstance(json.loads(captured.err)["error"], str)


def test_cli_help_remains_normal_successful_text(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    captured = capsys.readouterr()
    assert exit_info.value.code == 0
    assert "Run one read-only industrial diagnosis." in captured.out
    assert captured.err == ""
