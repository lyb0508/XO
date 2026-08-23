"""Central, side-effect-free configuration for the first milestone."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings without reading environment variables in business modules.

    Tracing values are retained here so later observability code has a single
    configuration source. This module neither enables tracing nor sends data.
    """

    model_config = SettingsConfigDict(
        env_prefix="INDUSTRIAL_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["ollama", "deepseek"] = "ollama"
    model: str = Field(default="qwen2.5:7b", min_length=1, max_length=128)
    ollama_base_url: AnyHttpUrl = "http://127.0.0.1:11434"
    deepseek_base_url: AnyHttpUrl = "https://api.deepseek.com/v1"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    deepseek_max_retries: int = Field(default=2, ge=0, le=10)
    structured_output_method: Literal["json_schema", "function_calling"] = "json_schema"

    # Evidence collection may use up to five sequential business tools plus a
    # short closing turn. Report formatting has a separate fixed single call.
    model_run_limit: int = Field(default=8, ge=1, le=20)
    tool_run_limit: int = Field(default=8, ge=1, le=50)
    per_tool_run_limit: int = Field(default=2, ge=1, le=10)

    # Manual retrieval. The default provider is the deterministic hash
    # embedding so offline use needs no model; "ollama" enables live semantic
    # retrieval with a locally pulled embedding model.
    embeddings_provider: Literal["deterministic", "ollama"] = "deterministic"
    embeddings_model: str = Field(default="nomic-embed-text", min_length=1, max_length=128)
    manual_retrieval_top_k: int = Field(default=3, ge=1, le=10)
    # 0.0 accepts every non-negative cosine score. The calibration samples in
    # tests/test_retrieval.py measured related queries at ~0.23-0.40 and
    # unrelated ones at <=0.08 for the deterministic embedding on the mock
    # corpus; recalibrate whenever the corpus or embeddings change.
    manual_retrieval_min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Session memory keeps at most this many recent turns per session id in
    # process memory; long-term records go to an append-only JSONL ledger.
    session_memory_max_turns: int = Field(default=5, ge=1, le=50)
    memory_ledger_path: str = Field(default="data/memory/approved_actions.jsonl", min_length=1, max_length=256)

    tracing_enabled: bool = False
    tracing_project: str = Field(
        default="industrial-diagnostic-agent-dev", min_length=1, max_length=128
    )
    environment: str = Field(default="development", min_length=1, max_length=64)
    tracing_endpoint: AnyHttpUrl = "https://api.smith.langchain.com"
    # SecretStr prevents accidental values appearing in repr(), validation output,
    # or normal logs. Only this project-prefixed environment variable is read.
    langsmith_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None

    @field_validator("ollama_base_url")
    @classmethod
    def ollama_must_use_loopback(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.scheme not in {"http", "https"} or value.host not in {"localhost", "127.0.0.1", "::1", "[::1]"}:
            raise ValueError("ollama_base_url must use http/https with a loopback host")
        return value

    @field_validator("deepseek_base_url")
    @classmethod
    def deepseek_must_use_official_https(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.scheme != "https" or value.host != "api.deepseek.com":
            raise ValueError("deepseek_base_url must use https://api.deepseek.com")
        return value

    @field_validator("tracing_endpoint")
    @classmethod
    def tracing_must_use_official_https(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.scheme != "https" or value.host != "api.smith.langchain.com":
            raise ValueError("tracing_endpoint must use https://api.smith.langchain.com")
        return value

    @model_validator(mode="after")
    def provider_and_structured_method_must_match(self) -> "Settings":
        required_method = "json_schema" if self.provider == "ollama" else "function_calling"
        if self.structured_output_method != required_method:
            raise ValueError(
                f"provider={self.provider} requires structured_output_method={required_method}"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached validated settings object for dependency injection."""

    return Settings()
