"""Explicit, opt-in live smoke tests. They never run in the default suite."""

from __future__ import annotations

import os

import pytest

from app.agents.diagnostic import AgentLimits, build_diagnostic_agent, run_diagnosis
from app.config.settings import Settings
from app.models.factory import create_chat_model


def _run_live(settings: Settings, question: str):
    model = create_chat_model(settings)
    agent = build_diagnostic_agent(model, limits=AgentLimits(model_run_limit=8, tool_run_limit=8, per_tool_run_limit=2), structured_output_method=settings.structured_output_method)
    return run_diagnosis(agent, question, request_id="live-smoke-001", device_id="PUMP-003")


@pytest.mark.live
def test_ollama_live_canonical_device_and_vibration_gate() -> None:
    if os.getenv("INDUSTRIAL_AGENT_RUN_LIVE_OLLAMA") != "1":
        pytest.skip("set INDUSTRIAL_AGENT_RUN_LIVE_OLLAMA=1 to opt into real Ollama")
    settings = Settings(_env_file=None, provider="ollama", structured_output_method="json_schema")
    report = _run_live(
        settings,
        "查询 PUMP-003 在 2026-08-22T00:00:00+00:00 至 "
        "2026-08-22T02:00:00+00:00 的 metric vibration_mm_s 历史，"
        "查询设备振动报警阈值，并给出风险研判。",
    )
    assert any(item.evidence_type == "device" and item.source_id == "asset:PUMP-003" for item in report.evidence)
    text = " ".join(item.summary for item in report.evidence)
    assert "7.1" in text and "8.2" in text and report.risk_level != "low"
    assert report.risk_level not in {"high", "critical"} or report.requires_human_review


@pytest.mark.live
def test_deepseek_live_canonical_device_and_vibration_gate() -> None:
    if os.getenv("INDUSTRIAL_AGENT_RUN_LIVE_DEEPSEEK") != "1" or not os.getenv("INDUSTRIAL_AGENT_DEEPSEEK_API_KEY", "").strip():
        pytest.skip("set DeepSeek live flag and project API key to opt into paid provider")
    settings = Settings(_env_file=None, provider="deepseek", model="deepseek-v4-flash", structured_output_method="function_calling")
    report = _run_live(
        settings,
        "查询 PUMP-003 在 2026-08-22T00:00:00+00:00 至 "
        "2026-08-22T02:00:00+00:00 的 metric vibration_mm_s 历史，"
        "查询设备振动报警阈值，并给出风险研判。",
    )
    assert any(item.evidence_type == "device" and item.source_id == "asset:PUMP-003" for item in report.evidence)
    text = " ".join(item.summary for item in report.evidence)
    assert "7.1" in text and "8.2" in text and report.risk_level != "low"
    assert report.risk_level not in {"high", "critical"} or report.requires_human_review
