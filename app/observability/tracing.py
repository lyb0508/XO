"""Explicit, redacted LangSmith tracing for the first learning milestone.

Tracing is deliberately opt-in: importing this module and entering a disabled
context neither constructs a LangSmith client nor makes a network request.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from langsmith import Client, tracing_context
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from app.config.settings import Settings


REDACTED_SECRET = "[REDACTED_SECRET]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|"
    r"authorization|cookie|secret|credential|bearer)",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|"
    r"authorization|cookie|secret|credential)\s*[:=]\s*[^\s,;]+"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_API_KEY_SECRET = re.compile(r"\b(?:sk|lsv2|pk)[_-][A-Za-z0-9_-]{8,}\b", re.IGNORECASE)
_EXPLICIT_SECRET_SENTINEL = re.compile(
    r"(?i)(?:<\s*secret\s*>|\[\s*secret\s*\]|__secret__|secret[_-]sentinel|"
    r"do[_-]?not[_-]?log)"
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_CN_MOBILE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9._:-]+$")


class _StrictTraceModel(BaseModel):
    """Trace metadata is intentionally small and rejects future free-form fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


class RunContext(_StrictTraceModel):
    """Identifiers that associate a single CLI invocation with a trace."""

    request_id: str = Field(min_length=1, max_length=128)
    thread_id: str = Field(min_length=1, max_length=128)

    @field_validator("request_id", "thread_id")
    @classmethod
    def identifiers_are_safe(cls, value: str) -> str:
        if not _SAFE_VALUE.fullmatch(value):
            raise ValueError("trace identifiers may contain only letters, digits, . _ : or -")
        return value


class TraceMetadata(_StrictTraceModel):
    """The complete allow-listed metadata that may leave the local process."""

    request_id: str = Field(min_length=1, max_length=128)
    thread_id: str = Field(min_length=1, max_length=128)
    agent_version: str = Field(min_length=1, max_length=64)
    environment: str = Field(min_length=1, max_length=64)
    provider: str = Field(min_length=1, max_length=32)
    model_alias: str = Field(min_length=1, max_length=128)

    @field_validator(
        "request_id", "thread_id", "agent_version", "environment", "provider", "model_alias"
    )
    @classmethod
    def values_are_safe(cls, value: str) -> str:
        if not _SAFE_VALUE.fullmatch(value):
            raise ValueError("trace metadata values may contain only letters, digits, . _ : or -")
        return value

    @classmethod
    def from_run_context(
        cls,
        context: RunContext,
        *,
        agent_version: str,
        environment: str,
        provider: str,
        model_alias: str,
    ) -> "TraceMetadata":
        return cls(
            request_id=context.request_id,
            thread_id=context.thread_id,
            agent_version=agent_version,
            environment=environment,
            provider=provider,
            model_alias=model_alias,
        )

    def tags(self) -> list[str]:
        """Build tags only from fixed, validated configuration values."""

        return [
            f"environment:{self.environment}",
            f"provider:{self.provider}",
            f"agent:{self.agent_version}",
        ]


def _redact_text(value: str) -> str:
    """Mask common secret and personal-data formats without flattening structure."""

    value = _BEARER_SECRET.sub(REDACTED_SECRET, value)
    # Process Bearer credentials first. Otherwise an ``Authorization: Bearer``
    # assignment would mask only the word "Bearer" and leave its token behind.
    value = _ASSIGNMENT_SECRET.sub(REDACTED_SECRET, value)
    value = _API_KEY_SECRET.sub(REDACTED_SECRET, value)
    value = _EXPLICIT_SECRET_SENTINEL.sub(REDACTED_SECRET, value)
    value = _EMAIL.sub(REDACTED_EMAIL, value)
    return _CN_MOBILE.sub(REDACTED_PHONE, value)


def redact_payload(payload: Any) -> Any:
    """Return a structurally equivalent payload with known sensitive values masked.

    This function is used for LangSmith inputs, outputs, metadata, and client
    anonymization. It never calls ``SecretStr.get_secret_value()``.
    """

    if isinstance(payload, SecretStr):
        return REDACTED_SECRET
    if isinstance(payload, str):
        return _redact_text(payload)
    if isinstance(payload, Mapping):
        return {
            str(key): REDACTED_SECRET
            if _SENSITIVE_KEY.search(str(key))
            else redact_payload(value)
            for key, value in payload.items()
        }
    if isinstance(payload, tuple):
        return tuple(redact_payload(value) for value in payload)
    if isinstance(payload, list):
        return [redact_payload(value) for value in payload]
    if isinstance(payload, set):
        return {redact_payload(value) for value in payload}
    if isinstance(payload, BaseModel):
        return redact_payload(payload.model_dump(mode="python"))
    return payload


def _safe_error_message(error: BaseException) -> str:
    return str(redact_payload(str(error))) or error.__class__.__name__


class TraceDeliveryFailure(RuntimeError):
    """A local, explicit signal that an enabled trace was not delivered."""


def _close_client(client: Client, *, preserve_business_error: bool) -> None:
    """Flush and close while never replacing an Agent/business exception."""

    errors: list[BaseException] = []
    for operation in (client.flush, client.close):
        try:
            operation()
        except BaseException as error:  # SDK cleanup must not hide a diagnosis failure.
            errors.append(error)

    if not errors:
        return

    message = "trace_delivery_failure: " + "; ".join(
        _safe_error_message(error) for error in errors
    )
    if preserve_business_error:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return
    raise TraceDeliveryFailure(message) from errors[0]


@contextmanager
def tracing_run(settings: Settings, metadata: TraceMetadata) -> Iterator[None]:
    """Enable one safe LangSmith trace only when configuration explicitly permits it.

    A disabled context is a no-op. When enabled, an API key is mandatory and a
    client is supplied explicitly so environment variables cannot silently
    change the trace destination. Remote delivery failures are raised unless an
    Agent exception is already in flight, in which case they are surfaced as a
    warning while preserving that primary exception.
    """

    if not settings.tracing_enabled:
        # ``enabled=False`` explicitly overrides LANGSMITH_TRACING and the
        # legacy LANGCHAIN_TRACING_V2 host settings. ``parent=False`` prevents
        # an enclosing third-party trace from being inherited. The SDK restores
        # the previous context when this block exits.
        with tracing_context(enabled=False, parent=False):
            yield
        return

    if settings.langsmith_api_key is None:
        raise RuntimeError("LangSmith tracing is enabled but no API key is configured")

    # Keep the plaintext value in this local only long enough to reject blank
    # configuration and construct the explicitly configured SDK client.
    api_key = settings.langsmith_api_key.get_secret_value()
    if not api_key.strip():
        raise RuntimeError("LangSmith tracing is enabled but its API key is blank")

    client = Client(
        api_url=str(settings.tracing_endpoint),
        api_key=api_key,
        timeout_ms=int(settings.timeout_seconds * 1000),
        anonymizer=redact_payload,
        hide_inputs=redact_payload,
        hide_outputs=redact_payload,
        hide_metadata=redact_payload,
    )
    try:
        with tracing_context(
            project_name=settings.tracing_project,
            tags=metadata.tags(),
            metadata=metadata.model_dump(mode="json"),
            enabled=True,
            client=client,
        ):
            yield
    except BaseException:
        _close_client(client, preserve_business_error=True)
        raise
    else:
        _close_client(client, preserve_business_error=False)
