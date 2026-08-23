"""Create the configured model without performing a network request."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from app.config.settings import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Build the configured real provider model.

    Fake models are deliberately supplied by tests through dependency injection
    to ``build_diagnostic_agent``; this factory never substitutes a provider.
    Ollama validation remains disabled, and neither branch invokes a model during
    construction.
    """

    if settings.provider == "ollama":
        return ChatOllama(
            model=settings.model,
            base_url=str(settings.ollama_base_url),
            temperature=settings.temperature,
            client_kwargs={"timeout": settings.timeout_seconds},
            validate_model_on_init=False,
        )

    if settings.provider == "deepseek":
        if (
            settings.deepseek_api_key is None
            or not settings.deepseek_api_key.get_secret_value().strip()
        ):
            raise ValueError("INDUSTRIAL_AGENT_DEEPSEEK_API_KEY is required when provider=deepseek")
        return ChatDeepSeek(
            model=settings.model,
            api_base=str(settings.deepseek_base_url),
            api_key=settings.deepseek_api_key,
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
            max_retries=settings.deepseek_max_retries,
        )

    raise ValueError(f"Unsupported model provider: {settings.provider}")
