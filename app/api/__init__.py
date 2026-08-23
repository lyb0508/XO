"""HTTP API for the diagnosis graph.

Security model:
* The service refuses to start without a configured API key (fail closed).
* Every diagnostic route requires the ``X-API-Key`` header, compared in
  constant time.
* Requests are rate limited per key with an in-process sliding window.
* All error payloads pass through the same redaction layer as the CLI.

Streaming model: one SSE connection drives exactly one graph execution
segment. When the graph raises an approval interrupt the stream emits
``approval_required`` and ends; the client submits a decision to
``POST /approvals/{thread_id}`` and opens a new stream (or reads the JSON
response) to continue from the checkpoint.
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
    """Request body shared by the synchronous and streaming endpoints."""

    question: str = Field(min_length=1, max_length=2000)
    device_id: str = Field(min_length=1, max_length=128)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class ApprovalRequestBody(DiagnosisRequest):
    decision: dict[str, Any]


class SlidingWindowLimiter:
    """Per-key sliding window limiter; process-local by design."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        if limit < 1:
            raise ValueError("rate limit must be at least one")
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
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
    """Build the FastAPI application around the diagnosis graph.

    Tests inject a fake model here; production leaves ``model=None`` so the
    configured provider factory runs. The service fails to start when no API
    key is configured.
    """

    resolved_settings = settings or get_settings()
    raw_key = api_key if api_key is not None else resolved_settings.api_key
    # Settings carries the key as SecretStr; direct injection may pass a plain
    # string. Both must resolve to the plaintext for constant-time comparison.
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
        if not x_api_key or not secrets.compare_digest(x_api_key, resolved_key):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def _limited(request: Request, x_api_key: str | None) -> None:
        client_host = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.check(f"{client_host}:{x_api_key}")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

    def _safe_identifier(value: str | None, field: str) -> str:
        candidate = (value or "").strip()
        if not candidate:
            import uuid

            return str(uuid.uuid4())
        if not _SAFE_VALUE.fullmatch(candidate):
            raise HTTPException(status_code=400, detail=f"{field} contains unsupported characters")
        return candidate

    # One shared in-process checkpointer: /diagnoses/stream may interrupt on a
    # thread, and POST /approvals/{thread_id} must resume that same thread.
    # A per-call saver would lose the interrupt state between requests.
    from langgraph.checkpoint.memory import InMemorySaver

    graph_checkpointer = InMemorySaver()

    def _build_graph(model_override: Any = None):
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
        from langgraph.types import Command

        result = graph.invoke(graph_input, config=config)
        for _ in range(3):
            interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
            if not interrupts:
                break
            # Server-side auto-reject keeps the HTTP path fail-safe: an
            # unattended synchronous call can never approve an action.
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
        return {"status": "ok"}

    @app.post("/diagnoses")
    def diagnose(body: DiagnosisRequest, request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
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
        _authorized(x_api_key)
        _limited(request, x_api_key)
        thread_id = _safe_identifier(body.thread_id, "thread_id")

        async def event_stream() -> AsyncIterator[bytes]:
            graph, _ = _build_graph()
            config = {
                "recursion_limit": GRAPH_RECURSION_LIMIT,
                "configurable": {"thread_id": thread_id},
            }
            try:
                for update in graph.stream(_graph_input(body), config=config, stream_mode="updates"):
                    for node_name, node_update in update.items():
                        # In updates mode the interrupt arrives as a tuple of
                        # Interrupt objects under the "__interrupt__" key.
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
    return int(getattr(settings, "session_memory_max_turns", 5))


def _sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
