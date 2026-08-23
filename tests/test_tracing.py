from __future__ import annotations

from contextlib import contextmanager

import pytest
from pydantic import SecretStr

from app.config.settings import Settings
from app.observability import tracing


class _FakeClient:
    instances: list["_FakeClient"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.flush_calls = 0
        self.close_calls = 0
        self.instances.append(self)

    def flush(self) -> None:
        self.flush_calls += 1

    def close(self) -> None:
        self.close_calls += 1


@contextmanager
def _fake_tracing_context(**kwargs: object):
    _fake_tracing_context.calls.append(kwargs)
    yield


_fake_tracing_context.calls = []


def _metadata() -> tracing.TraceMetadata:
    return tracing.TraceMetadata(
        request_id="request-001",
        thread_id="thread-001",
        agent_version="phase1",
        environment="test",
        provider="ollama",
        model_alias="qwen2.5:7b",
    )


def test_disabled_tracing_does_not_construct_client(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed = 0

    def should_not_construct(**kwargs: object) -> object:
        nonlocal constructed
        constructed += 1
        raise AssertionError("disabled tracing must not create a client")

    monkeypatch.setattr(tracing, "Client", should_not_construct)
    with tracing.tracing_run(Settings(_env_file=None, tracing_enabled=False), _metadata()):
        pass

    assert constructed == 0


def test_disabled_context_overrides_global_trace_environment_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tracing_context.calls.clear()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: pytest.fail("disabled must not construct Client"))
    monkeypatch.setattr(tracing, "tracing_context", _fake_tracing_context)
    with tracing.tracing_run(Settings(_env_file=None, tracing_enabled=False), _metadata()):
        pass
    assert _fake_tracing_context.calls == [{"enabled": False, "parent": False}]
    assert __import__("os").environ["LANGSMITH_TRACING"] == "true"


@pytest.mark.parametrize("key", [None, "", "   "])
def test_enabled_trace_rejects_none_empty_and_blank_keys_before_client(
    monkeypatch: pytest.MonkeyPatch, key: str | None
) -> None:
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: pytest.fail("Client was constructed"))
    settings = Settings(_env_file=None, tracing_enabled=True, langsmith_api_key=key)
    with pytest.raises(RuntimeError, match="API key"):
        with tracing.tracing_run(settings, _metadata()):
            pass


def test_enabled_tracing_without_key_fails_before_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "Client", lambda **kwargs: pytest.fail("Client was constructed"))

    with pytest.raises(RuntimeError, match="no API key"):
        with tracing.tracing_run(Settings(_env_file=None, tracing_enabled=True), _metadata()):
            pass


def test_enabled_trace_uses_only_allowlisted_metadata_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeClient.instances.clear()
    _fake_tracing_context.calls.clear()
    monkeypatch.setattr(tracing, "Client", _FakeClient)
    monkeypatch.setattr(tracing, "tracing_context", _fake_tracing_context)
    settings = Settings(_env_file=None, tracing_enabled=True, langsmith_api_key="lsv2_only-test-value")

    with tracing.tracing_run(settings, _metadata()):
        pass

    client = _FakeClient.instances[0]
    assert client.flush_calls == 1
    assert client.close_calls == 1
    assert set(_fake_tracing_context.calls[0]["metadata"]) == {
        "request_id", "thread_id", "agent_version", "environment", "provider", "model_alias"
    }
    assert _fake_tracing_context.calls[0]["metadata"]["request_id"] == "request-001"
    assert _fake_tracing_context.calls[0]["enabled"] is True
    assert _fake_tracing_context.calls[0]["client"] is client


def test_redactor_masks_keys_bearer_email_china_mobile_and_secretstr() -> None:
    redacted = tracing.redact_payload(
        {
            "api_key": "will-not-leak",
            "note": "Bearer very.secret-token alpha@example.com 13800138000 sk-secretzzzz sk_secretzzzz lsv2_secretzzzz pk_secretzzzz",
            "nested": [SecretStr("hidden"), {"authorization": "Bearer nested-token"}],
        }
    )

    assert redacted["api_key"] == "[REDACTED_SECRET]"
    assert "very.secret-token" not in redacted["note"]
    assert "alpha@example.com" not in redacted["note"]
    assert "13800138000" not in redacted["note"]
    assert all(token not in redacted["note"] for token in ("sk-secretzzzz", "sk_secretzzzz", "lsv2_secretzzzz", "pk_secretzzzz"))
    assert redacted["nested"] == ["[REDACTED_SECRET]", {"authorization": "[REDACTED_SECRET]"}]


@pytest.mark.parametrize("operation", ["flush", "close"])
def test_trace_cleanup_failure_raises_stable_delivery_error_without_business_error(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    class FailingClient(_FakeClient):
        def flush(self) -> None:
            super().flush()
            if operation == "flush":
                raise RuntimeError("Bearer trace-token")
        def close(self) -> None:
            super().close()
            if operation == "close":
                raise RuntimeError("sk_secretzzzz")

    monkeypatch.setattr(tracing, "Client", FailingClient)
    monkeypatch.setattr(tracing, "tracing_context", _fake_tracing_context)
    with pytest.raises(tracing.TraceDeliveryFailure, match="^trace_delivery_failure:") as error:
        with tracing.tracing_run(Settings(_env_file=None, tracing_enabled=True, langsmith_api_key="lsv2_test-key"), _metadata()):
            pass
    assert "trace-token" not in str(error.value) and "sk_secretzzzz" not in str(error.value)


def test_trace_cleanup_preserves_business_error_and_warns_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingClient(_FakeClient):
        def flush(self) -> None:
            raise RuntimeError("Bearer cleanup-token")

    monkeypatch.setattr(tracing, "Client", FailingClient)
    monkeypatch.setattr(tracing, "tracing_context", _fake_tracing_context)
    with pytest.warns(RuntimeWarning, match="trace_delivery_failure") as warning:
        with pytest.raises(RuntimeError, match="primary business error"):
            with tracing.tracing_run(Settings(_env_file=None, tracing_enabled=True, langsmith_api_key="lsv2_test-key"), _metadata()):
                raise RuntimeError("primary business error")
    assert "cleanup-token" not in str(warning[0].message)
