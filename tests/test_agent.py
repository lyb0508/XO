from __future__ import annotations

import copy
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.diagnostic import AgentLimits, TwoStageDiagnosticAgent, build_diagnostic_agent, run_diagnosis
from app.schemas.diagnostics import DiagnosisDraft
from conftest import ToolCapableFakeChatModel, scripted_tool_call


def _end(summary: str = "private evidence-agent summary") -> AIMessage:
    return AIMessage(content=summary)


def _out_of_scope_draft() -> dict[str, object]:
    return {"request_id": "request-001", "device_id": "PUMP-003", "scope_status": "out_of_scope", "risk_level": "unknown", "summary": "请求不在范围内。", "evidence_sufficient": False, "likely_causes": [], "evidence_ids": [], "recommended_actions": [], "requires_human_review": False, "limitations": ["不执行控制或无关任务。"]}


def _device_draft() -> dict[str, object]:
    return {"request_id": "request-001", "device_id": "PUMP-003", "scope_status": "in_scope", "risk_level": "low", "summary": "设备资产记录存在。", "evidence_sufficient": True, "likely_causes": [], "evidence_ids": ["asset:PUMP-003"], "recommended_actions": ["继续核查。"], "requires_human_review": False, "limitations": []}


def test_draft_formatter_selects_id_and_program_generates_canonical_evidence() -> None:
    model = ToolCapableFakeChatModel(evidence_responses=[scripted_tool_call("get_device_info", {"device_id": "PUMP-003"}, "device"), _end()], formatter_responses=[_device_draft()])
    report = run_diagnosis(build_diagnostic_agent(model), "查询设备", request_id="request-001")
    assert report.evidence[0].evidence_type == "device"
    assert report.evidence[0].source_id == "asset:PUMP-003"
    assert "7.1" in report.evidence[0].summary and report.evidence[0].version == "2026.08.mock.1"
    assert (model.evidence_call_count, model.formatter_call_count) == (2, 1)


def test_formatter_receives_only_draft_schema_and_no_private_ai_summary() -> None:
    private_summary = "MUST_NOT_REACH_FORMATTER"
    model = ToolCapableFakeChatModel(evidence_responses=[scripted_tool_call("get_device_info", {"device_id": "PUMP-003"}, "device"), _end(private_summary)], formatter_responses=[_device_draft()])
    run_diagnosis(build_diagnostic_agent(model), "查询设备", request_id="request-001")
    messages = model.formatter_inputs[0]
    assert len(messages) == 2 and isinstance(messages[0], SystemMessage) and isinstance(messages[1], HumanMessage)
    assert private_summary not in messages[1].content and '"canonical_evidence"' in messages[1].content
    assert model.formatter_schema is DiagnosisDraft
    assert model.formatter_kwargs == {"method": "json_schema", "include_raw": False}
    assert model.bind_calls and all("get_device_info" in names for names, _ in model.bind_calls)


def test_old_evidence_or_fabricated_fact_fields_are_rejected_by_draft_schema() -> None:
    old = _device_draft(); old["evidence"] = [{"evidence_id": "asset:PUMP-003"}]
    model = ToolCapableFakeChatModel(evidence_responses=[_end()], formatter_responses=[old])
    with pytest.raises(Exception, match="Extra inputs are not permitted"):
        run_diagnosis(build_diagnostic_agent(model), "无关", request_id="request-001")
    forged = _device_draft(); forged["evidence_type"] = "alarm"
    model = ToolCapableFakeChatModel(evidence_responses=[_end()], formatter_responses=[forged])
    with pytest.raises(Exception, match="Extra inputs are not permitted"):
        run_diagnosis(build_diagnostic_agent(model), "无关", request_id="request-001")


def test_one_ai_message_with_two_distinct_tool_calls_is_processed() -> None:
    both = AIMessage(content="", tool_calls=[{"name": "get_device_info", "args": {"device_id": "PUMP-003"}, "id": "device"}, {"name": "search_manual", "args": {"device_id": "PUMP-003", "query": "振动"}, "id": "manual"}])
    draft = _device_draft(); draft["evidence_ids"] = ["asset:PUMP-003", "manual:circulation-pump-v2:section-4.2"]
    model = ToolCapableFakeChatModel(evidence_responses=[both, _end()], formatter_responses=[draft])
    report = run_diagnosis(build_diagnostic_agent(model), "查询设备手册", request_id="request-001")
    assert [item.evidence_type for item in report.evidence] == ["device", "manual"]
    assert any(isinstance(message, ToolMessage) for message in model.seen_messages[1])


def _vibration_responses(start_at: datetime, end_at: datetime) -> list[AIMessage]:
    return [AIMessage(content="", tool_calls=[{"name": "get_device_info", "args": {"device_id": "PUMP-003"}, "id": "device"}, {"name": "query_sensor_history", "args": {"device_id": "PUMP-003", "start_at": start_at, "end_at": end_at, "metric": "vibration_mm_s"}, "id": "sensor"}]), _end()]


def _vibration_ids() -> list[str]:
    return ["asset:PUMP-003", "sensor:PUMP-003:2026-08-22T01:00:00Z", "sensor:PUMP-003:2026-08-22T01:10:00Z", "sensor:PUMP-003:2026-08-22T01:20:00Z"]


def test_vibration_gate_selects_all_points_and_threshold_then_generates_canonical_report(valid_draft_payload: dict[str, object], utc_window: tuple[datetime, datetime]) -> None:
    start_at, end_at = utc_window
    draft = copy.deepcopy(valid_draft_payload); draft["evidence_ids"] = _vibration_ids(); draft["likely_causes"][0]["evidence_ids"] = ["sensor:PUMP-003:2026-08-22T01:10:00Z"]
    model = ToolCapableFakeChatModel(evidence_responses=_vibration_responses(start_at, end_at), formatter_responses=[draft])
    report = run_diagnosis(build_diagnostic_agent(model), "研判振动", request_id="request-001")
    text = " ".join(item.summary for item in report.evidence)
    assert report.risk_level == "high" and report.requires_human_review is True
    assert "8.2" in text and "7.1" in text
    assert len([item for item in report.evidence if item.evidence_type == "sensor"]) == 3


@pytest.mark.parametrize("mutation", ["missing_threshold", "missing_point"])
def test_vibration_repair_completes_missing_threshold_or_omitted_point(valid_draft_payload: dict[str, object], utc_window: tuple[datetime, datetime], mutation: str) -> None:
    start_at, end_at = utc_window
    draft = copy.deepcopy(valid_draft_payload); ids = _vibration_ids()
    if mutation == "missing_threshold": ids.remove("asset:PUMP-003")
    if mutation == "missing_point": ids.pop()
    draft["evidence_ids"] = ids; draft["likely_causes"][0]["evidence_ids"] = ["sensor:PUMP-003:2026-08-22T01:10:00Z"]
    model = ToolCapableFakeChatModel(evidence_responses=_vibration_responses(start_at, end_at), formatter_responses=[draft])
    report = run_diagnosis(build_diagnostic_agent(model), "研判振动", request_id="request-001")
    selected_types = sorted(item.evidence_type for item in report.evidence)
    assert selected_types == ["device", "sensor", "sensor", "sensor"]
    assert any(item.evidence_type == "device" for item in report.evidence)


def test_vibration_gate_still_rejects_low_risk_with_above_threshold_evidence(valid_draft_payload: dict[str, object], utc_window: tuple[datetime, datetime]) -> None:
    start_at, end_at = utc_window
    draft = copy.deepcopy(valid_draft_payload)
    draft["risk_level"] = "low"; draft["requires_human_review"] = False
    draft["evidence_ids"] = _vibration_ids(); draft["likely_causes"][0]["evidence_ids"] = ["sensor:PUMP-003:2026-08-22T01:10:00Z"]
    model = ToolCapableFakeChatModel(evidence_responses=_vibration_responses(start_at, end_at), formatter_responses=[draft])
    with pytest.raises(RuntimeError, match="risk_level=low"):
        run_diagnosis(build_diagnostic_agent(model), "研判振动", request_id="request-001")


def test_limits_and_empty_question_do_not_run_formatter() -> None:
    model = ToolCapableFakeChatModel(evidence_responses=[scripted_tool_call("get_device_info", {"device_id": "PUMP-003"}, "one"), scripted_tool_call("get_device_info", {"device_id": "PUMP-003"}, "two"), scripted_tool_call("get_device_info", {"device_id": "PUMP-003"}, "three")], formatter_responses=[_device_draft()])
    with pytest.raises(Exception, match="[Tt]ool call limit.*exceeded"):
        run_diagnosis(build_diagnostic_agent(model, limits=AgentLimits(per_tool_run_limit=2)), "循环", request_id="request-001")
    assert model.formatter_call_count == 0
    model = ToolCapableFakeChatModel(evidence_responses=[_end()], formatter_responses=[_out_of_scope_draft()])
    with pytest.raises(ValueError, match="question must not be empty"):
        run_diagnosis(build_diagnostic_agent(model), "  ", request_id="request-001")
    assert model.total_call_count == 0


def test_model_and_total_tool_limits_fail_before_formatter_runs() -> None:
    model = ToolCapableFakeChatModel(
        evidence_responses=[scripted_tool_call("get_device_info", {"device_id": "PUMP-003"}, "one"), _end()],
        formatter_responses=[_device_draft()],
    )
    with pytest.raises(Exception, match="Model call limits exceeded"):
        run_diagnosis(build_diagnostic_agent(model, limits=AgentLimits(model_run_limit=1)), "查询", request_id="request-001")
    assert model.formatter_call_count == 0
    model = ToolCapableFakeChatModel(
        evidence_responses=[AIMessage(content="", tool_calls=[{"name": "get_device_info", "args": {"device_id": "PUMP-003"}, "id": "one"}, {"name": "search_manual", "args": {"device_id": "PUMP-003", "query": "振动"}, "id": "two"}])],
        formatter_responses=[_device_draft()],
    )
    with pytest.raises(Exception, match="[Tt]ool call limit.*exceeded"):
        run_diagnosis(build_diagnostic_agent(model, limits=AgentLimits(tool_run_limit=1)), "查询", request_id="request-001")
    assert model.formatter_call_count == 0


class _StaticEvidenceAgent:
    """Test-only evidence stage returning a fixed ToolMessage sequence."""

    def __init__(self, messages: list[ToolMessage]) -> None:
        self._messages = messages

    def invoke(self, input: object, config: dict[str, object] | None = None) -> dict[str, list[ToolMessage]]:
        return {"messages": self._messages}


def test_unresolved_natural_language_tool_error_blocks_formatter_with_stable_error() -> None:
    model = ToolCapableFakeChatModel(evidence_responses=[_end()], formatter_responses=[_device_draft()])
    formatter = model.with_structured_output(DiagnosisDraft, method="json_schema", include_raw=False)
    agent = TwoStageDiagnosticAgent(
        evidence_agent=_StaticEvidenceAgent([
            ToolMessage(
                content="缺少带时区的 start_at 参数，请修正工具调用。",
                name="query_sensor_history",
                tool_call_id="invalid-sensor-call",
                status="error",
            )
        ]),
        report_formatter=formatter,
    )

    errors: list[str] = []
    for _ in range(2):
        with pytest.raises(RuntimeError) as raised:
            run_diagnosis(agent, "查询振动", request_id="request-001")
        errors.append(str(raised.value))

    assert errors == [
        "evidence collection contains unresolved tool errors; report formatting is blocked",
        "evidence collection contains unresolved tool errors; report formatting is blocked",
    ]
    assert model.formatter_call_count == 0


def _entry(eid: str, etype: str, facts: dict[str, object]) -> object:
    from types import MappingProxyType

    from app.agents.evidence import RegistryEntry
    from app.schemas.diagnostics import EvidenceItem

    return RegistryEntry(
        evidence=EvidenceItem(evidence_id=eid, evidence_type=etype, source_id=eid, summary="s"),
        device_id="PUMP-003",
        tool_name="t",
        facts=MappingProxyType(dict(facts)),
    )


def test_repair_vibration_selection_is_noop_without_vibration_reference(valid_draft_payload: dict[str, object]) -> None:
    from types import MappingProxyType

    from app.agents.diagnostic import repair_vibration_selection
    from app.agents.evidence import EvidenceRegistry
    from app.schemas.diagnostics import DiagnosisDraft

    draft = DiagnosisDraft.model_validate(valid_draft_payload)
    entry = _entry(
        "sensor:PUMP-003:2026-08-22T01:10:00Z",
        "sensor",
        {"metric": "temperature_c", "value": 50.0},
    )
    registry = EvidenceRegistry(entries=MappingProxyType({entry.evidence.evidence_id: entry}), unresolved_tool_errors=frozenset())
    # The selected sensor point is a temperature metric, not vibration: no repair may fire.
    repaired = repair_vibration_selection(draft, registry)
    assert list(repaired.evidence_ids) == list(draft.evidence_ids)


def test_repair_skips_threshold_injection_when_registry_thresholds_conflict() -> None:
    from types import MappingProxyType

    from app.agents.diagnostic import repair_vibration_selection, validate_vibration_gate
    from app.agents.evidence import EvidenceRegistry
    from app.schemas.diagnostics import DiagnosisDraft

    entries = {
        "sensor-vib": _entry("sensor-vib", "sensor", {"metric": "vibration_mm_s", "value": 8.2}),
        "device-a": _entry("device-a", "device", {"vibration_alarm_threshold_mm_s": 7.1}),
        "device-b": _entry("device-b", "device", {"vibration_alarm_threshold_mm_s": 9.9}),
    }
    registry = EvidenceRegistry(entries=MappingProxyType(entries), unresolved_tool_errors=frozenset())
    draft = DiagnosisDraft.model_validate({
        "request_id": "request-001",
        "device_id": "PUMP-003",
        "scope_status": "in_scope",
        "risk_level": "medium",
        "summary": "振动超限。",
        "evidence_sufficient": True,
        "likely_causes": [{"cause": "轴承异常。", "confidence": 0.5, "evidence_ids": ["sensor-vib"]}],
        "evidence_ids": ["sensor-vib"],
        "recommended_actions": [],
        "requires_human_review": False,
        "limitations": [],
    })
    repaired = repair_vibration_selection(draft, registry)
    # Conflicting thresholds are never auto-selected; the gate still fails closed.
    assert "device-a" not in repaired.evidence_ids and "device-b" not in repaired.evidence_ids
    with pytest.raises(RuntimeError, match="threshold"):
        validate_vibration_gate(repaired, registry)
