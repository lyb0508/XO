"""Independent fixtures for the two-stage phase-one Agent contract."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class _ScriptedFormatter:
    """A schema-bound second-stage fake; it intentionally exposes no tools."""

    def __init__(self, model: "ToolCapableFakeChatModel", schema: object, kwargs: dict[str, Any]) -> None:
        self._model = model
        self._model.formatter_schema = schema
        self._model.formatter_kwargs = kwargs

    def invoke(self, messages: list[BaseMessage], config: dict[str, Any] | None = None) -> Any:
        self._model.formatter_call_count += 1
        self._model.formatter_inputs.append(list(messages))
        if not self._model.formatter_responses:
            raise AssertionError("test script has no formatter response")
        index = min(self._model._formatter_index, len(self._model.formatter_responses) - 1)
        self._model._formatter_index += 1
        return self._model.formatter_responses[index]


class ToolCapableFakeChatModel(BaseChatModel):
    """Deterministic fake for both Agent stages, without an external model call.

    It proves only wiring and enforcement boundaries. It is not evidence of real
    Ollama/DeepSeek tool-selection quality.
    """

    evidence_responses: list[BaseMessage]
    formatter_responses: list[Any]
    _evidence_index: int = PrivateAttr(default=0)
    _formatter_index: int = PrivateAttr(default=0)
    _evidence_call_count: int = PrivateAttr(default=0)
    _formatter_call_count: int = PrivateAttr(default=0)
    _seen_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _formatter_inputs: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bind_calls: list[tuple[list[str], dict[str, Any]]] = PrivateAttr(default_factory=list)
    _formatter_schema: object | None = PrivateAttr(default=None)
    _formatter_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def evidence_call_count(self) -> int:
        return self._evidence_call_count

    @property
    def formatter_call_count(self) -> int:
        return self._formatter_call_count

    @formatter_call_count.setter
    def formatter_call_count(self, value: int) -> None:
        self._formatter_call_count = value

    @property
    def total_call_count(self) -> int:
        return self._evidence_call_count + self._formatter_call_count

    @property
    def seen_messages(self) -> list[list[BaseMessage]]:
        return self._seen_messages

    @property
    def formatter_inputs(self) -> list[list[BaseMessage]]:
        return self._formatter_inputs

    @property
    def bind_calls(self) -> list[tuple[list[str], dict[str, Any]]]:
        return self._bind_calls

    @property
    def formatter_schema(self) -> object | None:
        return self._formatter_schema

    @formatter_schema.setter
    def formatter_schema(self, value: object) -> None:
        self._formatter_schema = value

    @property
    def formatter_kwargs(self) -> dict[str, Any]:
        return self._formatter_kwargs

    @formatter_kwargs.setter
    def formatter_kwargs(self, value: dict[str, Any]) -> None:
        self._formatter_kwargs = value

    @property
    def _llm_type(self) -> str:
        return "phase-one-two-stage-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolCapableFakeChatModel":
        names = [getattr(item, "name", getattr(item, "__name__", str(item))) for item in tools]
        self._bind_calls.append((names, dict(kwargs)))
        return self

    def with_structured_output(self, schema: object, **kwargs: Any) -> _ScriptedFormatter:
        return _ScriptedFormatter(self, schema, dict(kwargs))

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        self._evidence_call_count += 1
        self._seen_messages.append(list(messages))
        if not self.evidence_responses:
            raise AssertionError("test script has no evidence-agent response")
        index = min(self._evidence_index, len(self.evidence_responses) - 1)
        self._evidence_index += 1
        return ChatResult(generations=[ChatGeneration(message=self.evidence_responses[index])])


def scripted_tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class _ScriptedBound:
    """Schema-bound invokable returning scripted responses in order."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls = 0

    def invoke(self, messages: list[Any], config: dict[str, Any] | None = None) -> Any:
        if not self._responses:
            raise AssertionError("no scripted response configured")
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]


class GraphFakeChatModel(BaseChatModel):
    """Deterministic offline stand-in exposing two schema-bound invokables.

    The raw model raises if called directly: the graph must only ever reach the
    model through its two structured wrappers. Call counters live on the
    ``_planner``/``_formatter`` bound objects (``calls``).
    """

    plan_responses: list[Any]
    draft_responses: list[Any]

    _planner: _ScriptedBound | None = PrivateAttr(default=None)
    _formatter: _ScriptedBound | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "phase-three-graph-fake"

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        raise AssertionError("the diagnosis graph must never call the raw model directly")

    def with_structured_output(self, schema: object, **kwargs: Any) -> "_ScriptedBound":
        from app.schemas.diagnostics import DiagnosisDraft
        from app.schemas.query_plan import QueryPlan

        if schema is QueryPlan:
            self._planner = _ScriptedBound(list(self.plan_responses))
            return self._planner  # type: ignore[return-value]
        if schema is DiagnosisDraft:
            self._formatter = _ScriptedBound(list(self.draft_responses))
            return self._formatter  # type: ignore[return-value]
        raise AssertionError(f"unexpected structured schema {schema!r}")


@pytest.fixture
def utc_window() -> tuple[datetime, datetime]:
    return (
        datetime(2026, 8, 14, 0, 0, tzinfo=UTC),
        datetime(2026, 8, 23, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def valid_report_payload() -> dict[str, Any]:
    """Frozen independently from production field lists, router tables, and constants."""

    return {
        "request_id": "request-001",
        "device_id": "PUMP-003",
        "scope_status": "in_scope",
        "risk_level": "high",
        "summary": "振动持续超限，需要由现场负责人复核。",
        "evidence_sufficient": True,
        "likely_causes": [
            {
                "cause": "联轴器找正偏差或轴承状态异常。",
                "confidence": 0.72,
                "evidence_ids": ["sensor-1", "manual-1"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "sensor-1",
                "evidence_type": "sensor",
                "source_id": "mock_sensor_store:PUMP-003",
                "summary": "振动读数高于报警阈值。",
                "observed_at": "2026-08-22T01:20:00+00:00",
                "version": None,
            },
            {
                "evidence_id": "manual-1",
                "evidence_type": "manual",
                "source_id": "manual:circulation-pump-v2:section-4.2",
                "summary": "手册要求核查轴承和联轴器。",
                "observed_at": None,
                "version": "2.0",
            },
        ],
        "recommended_actions": ["安排现场复核并记录复测结果。"],
        "requires_human_review": True,
        "limitations": [],
    }


@pytest.fixture
def valid_draft_payload() -> dict[str, Any]:
    """Frozen model-writable payload; evidence facts are intentionally absent."""

    return {
        "request_id": "request-001",
        "device_id": "PUMP-003",
        "scope_status": "in_scope",
        "risk_level": "high",
        "summary": "振动连续超限，需要现场复核。",
        "evidence_sufficient": True,
        "likely_causes": [
            {"cause": "需要排查振动原因。", "confidence": 0.72, "evidence_ids": ["sensor:PUMP-003:2026-08-22T01:10:00Z"]}
        ],
        "evidence_ids": ["sensor:PUMP-003:2026-08-22T01:10:00Z"],
        "recommended_actions": ["安排现场复核。"],
        "requires_human_review": True,
        "limitations": [],
    }
