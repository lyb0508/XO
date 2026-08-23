from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.models import factory


def test_defaults_and_project_environment_mapping_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDUSTRIAL_AGENT_PROVIDER", "deepseek")
    monkeypatch.setenv("INDUSTRIAL_AGENT_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("INDUSTRIAL_AGENT_STRUCTURED_OUTPUT_METHOD", "function_calling")
    monkeypatch.setenv("INDUSTRIAL_AGENT_DEEPSEEK_API_KEY", "env-placeholder-secret")
    settings = Settings(_env_file=None)
    assert settings.provider == "deepseek" and settings.structured_output_method == "function_calling"
    assert "env-placeholder-secret" not in repr(settings)


@pytest.mark.parametrize("key", [None, "", "   "])
def test_deepseek_missing_key_fails_closed_before_provider_constructs(monkeypatch: pytest.MonkeyPatch, key: str | None) -> None:
    monkeypatch.setattr(factory, "ChatDeepSeek", lambda **kwargs: pytest.fail("must not construct"))
    settings = Settings(_env_file=None, provider="deepseek", model="deepseek-v4-flash", structured_output_method="function_calling", deepseek_api_key=key)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is required"):
        factory.create_chat_model(settings)


def test_provider_factories_are_monkeypatched_network_free_and_use_isolated_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, dict[str, object]] = {}
    def ollama(**kwargs: object) -> object: captured["ollama"] = kwargs; return object()
    def deepseek(**kwargs: object) -> object: captured["deepseek"] = kwargs; return object()
    monkeypatch.setattr(factory, "ChatOllama", ollama)
    monkeypatch.setattr(factory, "ChatDeepSeek", deepseek)
    factory.create_chat_model(Settings(_env_file=None, provider="ollama", structured_output_method="json_schema"))
    factory.create_chat_model(Settings(_env_file=None, provider="deepseek", model="deepseek-v4-flash", structured_output_method="function_calling", deepseek_api_key="placeholder-secret"))
    assert str(captured["ollama"]["base_url"]).startswith("http://127.0.0.1")
    assert captured["ollama"]["validate_model_on_init"] is False
    assert str(captured["deepseek"]["api_base"]).startswith("https://api.deepseek.com")
    assert str(captured["deepseek"]["api_key"]) == "**********"


@pytest.mark.parametrize("kwargs", [
    {"provider": "ollama", "structured_output_method": "function_calling"},
    {"provider": "deepseek", "structured_output_method": "json_schema"},
    {"ollama_base_url": "http://10.0.0.4:11434"},
    {"deepseek_base_url": "http://api.deepseek.com/v1"},
    {"deepseek_base_url": "https://proxy.example.com/v1"},
    {"tracing_endpoint": "http://api.smith.langchain.com"},
    {"tracing_endpoint": "https://trace.example.com"},
])
def test_cross_provider_methods_and_untrusted_endpoints_are_rejected(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


@pytest.mark.unit
def test_deepseek_thinking_disabled_by_default_for_structured_output() -> None:
    """v4 思考模式拒绝强制 tool_choice：默认必须携带 thinking=disabled。"""

    model = factory.create_chat_model(
        Settings(_env_file=None, provider="deepseek", model="deepseek-v4-flash",
                 structured_output_method="function_calling",
                 deepseek_api_key="placeholder-secret")
    )
    assert model.extra_body == {"thinking": {"type": "disabled"}}


@pytest.mark.unit
def test_deepseek_thinking_can_be_reenabled_explicitly() -> None:
    model = factory.create_chat_model(
        Settings(_env_file=None, provider="deepseek", model="deepseek-v4-flash",
                 structured_output_method="function_calling",
                 deepseek_api_key="placeholder-secret", deepseek_thinking_disabled=False)
    )
    assert not model.extra_body
