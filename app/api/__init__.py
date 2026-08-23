"""HTTP API for the diagnosis graph.

本模块把诊断 LangGraph 包装成一个 FastAPI 服务，是第六阶段"服务化"的 HTTP 入口。

职责与数据流：
* 所有请求先经过 API key 认证（``X-API-Key`` 头）和滑动窗口限流，再进业务逻辑；
* ``POST /diagnoses``：同步跑完一次诊断，直接返回最终 JSON 报告；
* ``POST /diagnoses/stream``：以 SSE 流式推送节点进度；如果 Graph 在高风险动作前
  抛出审批中断（interrupt），流会发出一个 ``approval_required`` 事件后结束；
* ``POST /approvals/{thread_id}``：接收人工决策，借助共享 checkpointer 从 checkpoint
  恢复被中断的线程继续执行。三个端点合起来构成"中断 -> 审批 -> 恢复"的完整往返。

安全边界（fail-closed，宁可拒绝也不放行）：
* 未配置 API key 时服务直接拒绝启动；
* key 比较使用常量时间算法，防止通过响应耗时推测密钥；
* 限流按"客户端地址 + key"在进程内滑动窗口计数，超限返回 429 并附 Retry-After；
* 所有响应载荷与错误消息都经过统一脱敏层 redact_payload，避免密钥等敏感值外泄；
* 同步端点遇到审批中断会自动拒绝而非自动批准：无人值守的 HTTP 调用绝不能
  替人批准停机等高危动作。

失败行为：Graph 内部出错、报告缺失或校验失败都会转换成带脱敏消息的
4xx/5xx JSON 错误；SSE 路径则以 ``error`` 事件收尾，连接不会悬挂。
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from app.config.settings import get_settings
from app.graphs.builder import GRAPH_RECURSION_LIMIT, build_diagnosis_graph
from app.memory.ledger import LongTermLedger
from app.memory.session import SessionMemory
from app.models.factory import create_chat_model
from app.observability.tracing import _SAFE_VALUE, redact_payload
from app.retrieval.retriever import create_manual_store
from app.schemas.approval import ApprovalDecision
from app.schemas.diagnostics import DiagnosisReport


class DiagnosisRequest(BaseModel):
    """同步与流式诊断端点共用的请求体。

    字段的长度上下限是最基础的输入防线：过短的问题没有意义，超长输入则可能
    拖垮下游模型调用，因此在进入 Graph 之前就会被 pydantic 拒绝。
    """

    question: str = Field(min_length=1, max_length=2000)
    device_id: str = Field(min_length=1, max_length=128)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class ApprovalRequestBody(DiagnosisRequest):
    """审批端点请求体：继承诊断请求的全部字段，额外携带人工决策内容。"""

    decision: dict[str, Any]


class SlidingWindowLimiter:
    """按 key 统计的滑动窗口限流器；有意做成进程内实现。

    不引入 Redis 等外部存储是为了零依赖、低延迟；代价是多进程/多副本部署时
    各自独立计数，实际总配额等于"单副本限额 x 副本数"。窗口左端的过期记录
    采用惰性清理：只在请求到来时顺带淘汰，无需后台定时任务。
    """

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        if limit < 1:
            raise ValueError("rate limit must be at least one")
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """判断本次请求是否放行，返回（是否允许，建议等待秒数）。

        被拒时不写入记录，因此重试不会把窗口越推越远；读写都在锁内完成，
        保证并发下计数准确。
        """
        # 用 monotonic 时钟计量窗口：不受系统时间回拨或跳变的影响。
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                retry_after = int(self._window - (now - hits[0])) + 1
                return False, max(retry_after, 1)
            hits.append(now)
            return True, 0


def create_app(
    *,
    model: Any = None,
    settings: Any = None,
    api_key: str | None = None,
):
    """围绕诊断 Graph 组装 FastAPI 应用（factory 模式）。

    为什么做成工厂函数而不是模块级 app 单例：测试可以在这里注入 fake model
    与自定义 settings，生产环境则留 ``model=None``，由 provider factory 按
    配置创建真实模型；同时每个应用实例拥有独立的 checkpointer、限流器与
    记忆组件，多套应用互不串扰。

    失败行为：解析不到非空 API key 时立刻抛 RuntimeError 拒绝启动
    （fail-closed），绝不"先起来再说"；记忆台账或手册检索库初始化失败同样
    会让启动本身失败，而不是等到第一个请求才报错。
    """

    resolved_settings = settings or get_settings()
    raw_key = api_key if api_key is not None else resolved_settings.api_key
    # Settings 携带的 key 是 SecretStr，测试直注时可能是普通字符串；
    # 两种来源都必须先还原成明文，才能参与常量时间比较。
    resolved_key = (
        raw_key.get_secret_value() if hasattr(raw_key, "get_secret_value") else str(raw_key)
    ) if raw_key is not None else ""
    if not resolved_key.strip():
        raise RuntimeError(
            "INDUSTRIAL_AGENT_API_KEY must be configured before the HTTP API can start"
        )
    resolved_key = resolved_key.strip()

    app = FastAPI(title="Industrial Diagnostic Agent", version="phase6-api")
    limiter = SlidingWindowLimiter(resolved_settings.api_rate_limit)
    session_memory = SessionMemory(max_turns=settings_session_turns(resolved_settings))
    ledger = LongTermLedger(resolved_settings.memory_ledger_path)
    manual_store = create_manual_store(resolved_settings)

    def _authorized(x_api_key: str | None) -> None:
        """校验 ``X-API-Key`` 请求头，失败抛 401。

        用 secrets.compare_digest 做常量时间比较：普通的 ``==`` 在遇到第一个
        不同字符时就提前返回，攻击者可借响应耗时逐位猜出 key；常量时间
        比较让耗时与差异位置无关，堵住这类时序侧信道。
        """
        if not x_api_key or not secrets.compare_digest(x_api_key, resolved_key):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def _limited(request: Request, x_api_key: str | None) -> None:
        """滑动窗口限流检查，超限抛 429 并附 Retry-After 头。

        以"客户端地址:key"为计数维度：同一 IP 换 key、或同一 key 换 IP 都会
        分别计数，避免某一维度上的正常用户被他人占满配额。
        """
        client_host = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.check(f"{client_host}:{x_api_key}")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

    def _safe_identifier(value: str | None, field: str) -> str:
        """校验 request_id / thread_id / session_id 等标识符。

        空值回退为随机 UUID（省去调用方自行生成）；含白名单之外字符则抛
        400。这些值会进入日志、Trace 标签与 checkpoint 存储键，必须先用
        _SAFE_VALUE 正则约束，防止特殊字符注入到上述位置。
        """
        candidate = (value or "").strip()
        if not candidate:
            import uuid

            return str(uuid.uuid4())
        if not _SAFE_VALUE.fullmatch(candidate):
            raise HTTPException(status_code=400, detail=f"{field} contains unsupported characters")
        return candidate

    # checkpointer 必须是 create_app 级共享单例：/diagnoses/stream 可能把某个
    # thread 中断后直接结束本次 HTTP 连接，而 POST /approvals/{thread_id} 是
    # 一个全新的请求，只有两次请求共用同一个 InMemorySaver，后者才能凭
    # thread_id 找回中断现场并恢复执行。若每次请求各建一个 saver，中断状态
    # 随请求结束丢失，"审批后恢复"将永远无法完成。
    from langgraph.checkpoint.memory import InMemorySaver

    graph_checkpointer = InMemorySaver()

    def _build_graph(model_override: Any = None):
        """按"测试注入 > 工厂参数 > provider factory"的优先级确定模型并构建 Graph。

        每次请求都重新 build，但共享同一批会话记忆、长期台账、手册检索库与
        checkpointer，因此中断前后状态依然连续。
        """
        active_model = model_override or model or create_chat_model(resolved_settings)
        return build_diagnosis_graph(
            active_model,
            structured_output_method=resolved_settings.structured_output_method,
            checkpointer=graph_checkpointer,
            manual_store=manual_store,
            manual_top_k=resolved_settings.manual_retrieval_top_k,
            manual_min_score=resolved_settings.manual_retrieval_min_score,
            session_memory=session_memory,
            ledger=ledger,
        ), resolved_settings

    def _graph_input(body: DiagnosisRequest) -> dict[str, Any]:
        """把请求体清洗为 Graph 初始 State：校验标识符、去首尾空白、拒绝空问题。"""
        payload = {
            "request_id": _safe_identifier(body.request_id, "request_id"),
            "device_id": body.device_id.strip(),
            "question": body.question.strip(),
        }
        if not payload["question"]:
            raise HTTPException(status_code=400, detail="question must not be empty")
        if body.session_id:
            payload["session_id"] = _safe_identifier(body.session_id, "session_id")
        return payload

    def _invoke_with_approval(graph, graph_input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """同步执行 Graph，遇到审批中断一律"服务端自动拒绝"。

        为什么必须 fail-safe：同步 HTTP 调用没有真人在场，若这里自动批准，
        等于让"能发请求的一方"绕过 Human-in-the-loop 直接触发停机等高危
        动作；自动拒绝则保证受控动作只能经 /approvals 由真人裁决。
        最多重入 3 次，防御意外情况下陷入无限审批循环。
        """
        from langgraph.types import Command

        result = graph.invoke(graph_input, config=config)
        for _ in range(3):
            interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
            if not interrupts:
                break
            # 服务端自动拒绝维持 fail-safe：无人值守的同步调用永远无权批准动作。
            result = graph.invoke(
                Command(
                    resume={
                        "decision": "rejected",
                        "decided_by": "http-auto-reject",
                        "reason": "synchronous API calls cannot approve controlled actions",
                    }
                ),
                config=config,
            )
        return result

    def _outcome(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
        """把 Graph 最终状态整理为对外 JSON；所有字段出口前统一脱敏。

        出错或缺报告时抛 422，错误消息同样先经 redact_payload 清洗——异常
        文本里可能夹带内部 URL、路径甚至密钥片段，不能原样回给客户端。
        """
        if result.get("error") or not result.get("report"):
            message = redact_payload(str(result.get("error", "diagnosis produced no report")))
            raise HTTPException(status_code=422, detail=message)
        report = DiagnosisReport.model_validate(result["report"])
        return {
            "thread_id": thread_id,
            "report": redact_payload(report.model_dump(mode="json")),
            "approval": redact_payload(result.get("approval")),
            "action_audit": redact_payload(result.get("action_audit")),
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        """存活探针：不做认证与限流，供负载均衡或容器编排探活。"""
        return {"status": "ok"}

    @app.post("/diagnoses")
    def diagnose(body: DiagnosisRequest, request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        """同步诊断端点：一次请求内跑完整条链路并返回最终 JSON 报告。

        途中遇到审批中断时会被 _invoke_with_approval 自动拒绝后继续，
        因此响应里的 approval 字段呈现 rejected/http-auto-reject，
        请求不会挂起等待人工。
        """
        _authorized(x_api_key)
        _limited(request, x_api_key)
        graph, _ = _build_graph()
        thread_id = _safe_identifier(body.thread_id, "thread_id")
        config = {
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "configurable": {"thread_id": thread_id},
        }
        try:
            result = _invoke_with_approval(graph, _graph_input(body), config)
            return _outcome(result, thread_id)
        except HTTPException:
            raise
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=str(redact_payload(str(error))))
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(redact_payload(str(error))))

    @app.post("/diagnoses/stream")
    async def diagnose_stream(body: DiagnosisRequest, request: Request, x_api_key: str | None = Header(default=None)) -> StreamingResponse:
        """SSE 流式诊断端点：以 text/event-stream 边执行边推送进度。

        事件序列：若干 ``node``（单个节点完成）-> ``done``（含最终报告），
        或 ``approval_required``（流随即结束，等待人工决策），或 ``error``。
        """
        _authorized(x_api_key)
        _limited(request, x_api_key)
        thread_id = _safe_identifier(body.thread_id, "thread_id")

        async def event_stream() -> AsyncIterator[bytes]:
            """真正的 SSE 生产者：一条连接只驱动一段 Graph 执行。

            "中断即断开"的设计：发出 approval_required 后本连接结束，恢复由
            POST /approvals/{thread_id} 的下一个请求完成——每次 SSE 连接职责
            单一，也避免连接长期挂着等人审批。
            """
            graph, _ = _build_graph()
            config = {
                "recursion_limit": GRAPH_RECURSION_LIMIT,
                "configurable": {"thread_id": thread_id},
            }
            try:
                for update in graph.stream(_graph_input(body), config=config, stream_mode="updates"):
                    for node_name, node_update in update.items():
                        # stream_mode="updates" 下，中断不是普通节点更新，
                        # 而是以 Interrupt 对象元组挂在 "__interrupt__" 键下。
                        if node_name == "__interrupt__":
                            container = node_update if isinstance(node_update, (list, tuple)) else []
                            payload = container[0].value if container and hasattr(container[0], "value") else {}
                            yield _sse("approval_required", redact_payload(payload))
                            return
                        if isinstance(node_update, dict):
                            yield _sse("node", {"node": node_name})
                final_state = graph.get_state(config)
                values = final_state.values or {}
                if values.get("error") or not values.get("report"):
                    message = redact_payload(str(values.get("error", "diagnosis produced no report")))
                    yield _sse("error", {"error": message})
                    return
                report = DiagnosisReport.model_validate(values["report"])
                yield _sse(
                    "done",
                    {
                        "thread_id": thread_id,
                        "report": redact_payload(report.model_dump(mode="json")),
                        "approval": redact_payload(values.get("approval")),
                        "action_audit": redact_payload(values.get("action_audit")),
                    },
                )
            except Exception as error:
                yield _sse("error", {"error": str(redact_payload(str(error)))})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/approvals/{thread_id}")
    def approve(
        thread_id: str,
        body: ApprovalRequestBody,
        request: Request,
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """审批决策端点：从 checkpoint 恢复被中断的线程并续跑到出报告。

        Command(resume=...) 唤醒该 thread_id 上挂起的 interrupt 节点并交给它
        决策结果；决策结构先经 ApprovalDecision 校验，approve/modify/reject
        三种结果都如实透传，不存在默认放行。
        """
        _authorized(x_api_key)
        _limited(request, x_api_key)
        safe_thread = _safe_identifier(thread_id, "thread_id")
        try:
            decision = ApprovalDecision.model_validate(body.decision)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=str(redact_payload(str(error))))
        graph, _ = _build_graph()
        config = {
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "configurable": {"thread_id": safe_thread},
        }
        try:
            from langgraph.types import Command

            result = graph.invoke(Command(resume=decision.model_dump(mode="json")), config=config)
            return _outcome(result, safe_thread)
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(redact_payload(str(error))))

    return app


def settings_session_turns(settings: Any) -> int:
    """读取会话记忆轮数上限（缺省 5）；getattr 写法兼容测试用的精简 settings。"""
    return int(getattr(settings, "session_memory_max_turns", 5))


def _sse(event: str, data: Any) -> bytes:
    """把一条事件编码为 SSE 帧：event 行 + data 行 + 空行结尾。

    ensure_ascii=False 让中文内容按 UTF-8 原样输出，而不是转成转义序列。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
