from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.models.factory import create_chat_model


def test_defaults_load_without_key_and_tracing_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INDUSTRIAL_AGENT_LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("INDUSTRIAL_AGENT_TRACING_ENABLED", raising=False)

    settings = Settings(_env_file=None)

    assert settings.provider == "ollama"
    assert settings.tracing_enabled is False
    assert settings.langsmith_api_key is None


def test_secret_value_is_not_in_settings_representation() -> None:
    settings = Settings(langsmith_api_key="lsv2_this-must-not-appear")

    assert "this-must-not-appear" not in repr(settings)
    assert "SecretStr" in repr(settings)


def test_ollama_factory_returns_real_provider_and_never_fake() -> None:
    model = create_chat_model(Settings(_env_file=None, model="qwen2.5:7b"))

    assert model.__class__.__name__ == "ChatOllama"
    assert "fake" not in model.__class__.__name__.lower()


def test_unsupported_provider_is_rejected_before_model_construction() -> None:
    settings = Settings(_env_file=None)
    object.__setattr__(settings, "provider", "not-a-provider")

    with pytest.raises(ValueError, match="Unsupported model provider"):
        create_chat_model(settings)
