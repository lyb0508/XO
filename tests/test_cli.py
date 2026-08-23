from __future__ import annotations

import json

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


def test_cli_prints_exactly_one_report_json_to_stdout_and_passes_structured_method(monkeypatch, capsys) -> None:
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

    def build_stub(model, *, structured_output_method):
        from types import SimpleNamespace

        received["method"] = structured_output_method
        return SimpleNamespace(
            invoke=lambda state, config: {
                "report": _report().model_dump(mode="json"),
                "error": "",
            }
        )

    monkeypatch.setattr(cli, "build_diagnosis_graph", build_stub)

    exit_code = cli.main(["--question", "无关问题", "--request-id", "request-001", "--thread-id", "thread-001"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == _report().model_dump(mode="json")
    assert captured.out.count("\n") == 1
    assert received == {"method": "function_calling"}


def test_cli_fail_closed_branch_prints_redacted_error_to_stderr(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, tracing_enabled=False))
    monkeypatch.setattr(cli, "create_chat_model", lambda settings: object())
    monkeypatch.setattr(
        cli,
        "build_diagnosis_graph",
        lambda model, *, structured_output_method: SimpleNamespace(
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
