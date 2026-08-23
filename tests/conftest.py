"""Independent fixtures for the two-stage phase-one Agent contract.

本文件是 pytest 自动加载的共享夹具层，为各测试模块提供确定性的 fake 模型与
冻结的契约数据，核心回答一个问题：单元测试为什么以及如何完全不依赖真实模型。

工作方式：
1. ``ToolCapableFakeChatModel`` 服务一阶段两段式 Agent——证据 Agent 通过
   ``bind_tools``/``_generate`` 走脚本化 tool_calls；格式化阶段通过
   ``with_structured_output`` 返回脚本化结构化结果。两个阶段各自维护响应
   队列与调用计数，按顺序回放，队列耗尽时抛 AssertionError 让测试立刻暴露。
2. ``GraphFakeChatModel`` 服务三阶段的 LangGraph 诊断 Graph：原始模型一旦被
   直接调用就抛 AssertionError，Graph 只允许通过 QueryPlan / DiagnosisDraft
   两个结构化包装器触达模型。
3. 数据类 fixture（报告/草稿 payload）在测试侧独立手写并冻结，不 import
   生产代码的字段列表或常量，避免“实现改了测试也跟着改”的假阳性。

API 层测试复用这里的 ``GraphFakeChatModel``：通过 monkeypatch 替换应用内的
模型工厂后，FastAPI 测试客户端即可在不启动 Ollama/DeepSeek 的情况下走完整
请求 → Graph → 响应链路。

用 fake 而不用真实模型的原因：单元测试要的是可复现、零成本、毫秒级的反馈；
fake 只能证明“接线正确、边界被强制执行”，不能证明真实模型会选对工具——
那是 evaluations/ 里 Dataset + Evaluator 的职责，两者互补而非替代。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


class _ScriptedFormatter:
    """A schema-bound second-stage fake; it intentionally exposes no tools.

    格式化阶段的脚本化包装器：由 ``with_structured_output`` 创建，持有对宿主
    fake 模型的引用。每次 invoke 记录调用次数与输入消息，然后按顺序从
    formatter_responses 队列回放下一条响应；越界时停在最后一条（便于测试
    重复调用同一响应），队列本身为空则直接断言失败。

    刻意不提供任何工具接口：格式化阶段是纯结构化输出，若 Agent 在此阶段
    尝试调工具，测试会立即失败。
    """

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

    一阶段两段式 Agent 的确定性 fake，覆盖两个阶段：
    - 证据阶段：``_generate`` 按脚本回放 evidence_responses（通常是带
      tool_calls 的 AIMessage），并记录每次收到的消息序列供测试断言。
    - 格式化阶段：``with_structured_output`` 返回 _ScriptedFormatter，
      回放 formatter_responses。

    它只能证明接线与强制边界（工具是否被绑定、消息是否按预期流动、调用
    次数是否符合上限），不能证明真实 Ollama/DeepSeek 的工具选择质量——
    那属于 evaluations/ 评测体系的职责。
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
        # 证据阶段入口：记录消息、按脚本回放；脚本耗尽即断言失败，
        # 让测试在第一时间发现“模型被调用了比预期更多次”。
        self._evidence_call_count += 1
        self._seen_messages.append(list(messages))
        if not self.evidence_responses:
            raise AssertionError("test script has no evidence-agent response")
        index = min(self._evidence_index, len(self.evidence_responses) - 1)
        self._evidence_index += 1
        return ChatResult(generations=[ChatGeneration(message=self.evidence_responses[index])])


def scripted_tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """构造一条带单个 tool_calls 的 AIMessage，作为证据阶段的脚本响应。"""

    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class _ScriptedBound:
    """Schema-bound invokable returning scripted responses in order.

    GraphFakeChatModel 专用的结构化包装器：按顺序回放响应并计数。与
    _ScriptedFormatter 不同，它独立持有响应队列（不回写宿主），调用次数
    通过 ``calls`` 属性暴露给测试。
    """

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

    LangGraph 诊断 Graph 的确定性替身。设计要点：原始 ``_generate`` 一被
    调用就抛 AssertionError——诊断 Graph 必须且只能通过两个结构化包装器
    （QueryPlan 规划、DiagnosisDraft 报告）触达模型，任何绕过结构化输出的
    路径都会让测试立刻失败。

    ``with_structured_output`` 按 schema 类型分发：QueryPlan 得到 plan_
    responses 队列的包装器，DiagnosisDraft 得到 draft_responses 队列的包装器；
    其他 schema 直接断言失败。调用计数挂在返回的 ``_planner``/``_formatter``
    对象的 ``calls`` 属性上。

    API 测试也复用本类：monkeypatch 掉模型工厂后，FastAPI 测试客户端即可
    离线跑通完整请求链路（见模块 docstring）。
    """

    plan_responses: list[Any]
    draft_responses: list[Any]

    _planner: _ScriptedBound | None = PrivateAttr(default=None)
    _formatter: _ScriptedBound | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "phase-three-graph-fake"

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        # 原始模型通道被刻意封死：Graph 只允许走结构化包装器。
        raise AssertionError("the diagnosis graph must never call the raw model directly")

    def with_structured_output(self, schema: object, **kwargs: Any) -> "_ScriptedBound":
        from app.schemas.diagnostics import DiagnosisDraft
        from app.schemas.query_plan import QueryPlan

        # 按 schema 类型分发各自的脚本队列；未知 schema 视为测试脚本错误。
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
    """最终诊断报告的契约样例，在测试侧独立冻结。

    字段与取值全部手写，不 import 生产代码的字段列表、路由表或常量——
    若生产实现改动导致此 fixture 失配，测试会如实失败而不是静默跟随。
    """

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
    """模型可写字段的草稿契约样例，在测试侧独立冻结。

    与 valid_report_payload 的关键差异：草稿只携带 evidence_ids 引用，
    不含证据事实本身——事实由工具层补充，模型不得凭空编造。
    """

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
