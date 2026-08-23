"""Central, side-effect-free configuration for the first milestone.

全项目唯一的配置中心：用 pydantic-settings 从 ``INDUSTRIAL_AGENT_`` 前缀的
环境变量与根目录 ``.env`` 文件读取配置，加载时即刻完成类型与取值校验
（URL 白名单、数值范围等），坏配置在启动瞬间暴露，而不是运行中途才失败。

设计要点：
* 业务模块不许散落读取 os.environ，一律通过 get_settings() 注入；
* 本模块零副作用：不启用追踪、不发网络请求、不写任何文件；
* 密钥类字段一律用 SecretStr 包装，避免明文意外出现在 repr、日志或校验报错里。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时配置的唯一来源：业务模块从这里取值，而不是各自读环境变量。

    Trace 相关配置也集中于此，让后续可观测性代码拥有单一配置源；
    本模块本身既不开启追踪，也不会向外发送任何数据。
    """

    model_config = SettingsConfigDict(
        env_prefix="INDUSTRIAL_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        # extra="ignore"：未在本类声明的环境变量一律忽略，
        # 其他项目留下的 INDUSTRIAL_AGENT_* 变量不会导致启动失败。
        extra="ignore",
    )

    provider: Literal["ollama", "deepseek"] = "ollama"
    model: str = Field(default="qwen2.5:7b", min_length=1, max_length=128)
    ollama_base_url: AnyHttpUrl = "http://127.0.0.1:11434"
    deepseek_base_url: AnyHttpUrl = "https://api.deepseek.com/v1"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    deepseek_max_retries: int = Field(default=2, ge=0, le=10)
    # deepseek-v4 默认思考模式拒绝强制 tool_choice（结构化输出必需），默认关闭。
    deepseek_thinking_disabled: bool = True
    structured_output_method: Literal["json_schema", "function_calling"] = "json_schema"

    # 证据采集最多串联五个业务工具外加一小段收尾回合；报告排版另有固定的一次
    # 单独调用。这几个上限共同封顶单次诊断的模型/工具调用量，防止 Agent 失控循环。
    model_run_limit: int = Field(default=8, ge=1, le=20)
    tool_run_limit: int = Field(default=8, ge=1, le=50)
    per_tool_run_limit: int = Field(default=2, ge=1, le=10)

    # 手册检索配置。默认 provider 是确定性 hash embedding，离线使用无需模型；
    # 选 "ollama" 则用本地拉取的 embedding 模型做真正的语义检索。
    embeddings_provider: Literal["deterministic", "ollama"] = "deterministic"
    embeddings_model: str = Field(default="nomic-embed-text", min_length=1, max_length=128)
    manual_retrieval_top_k: int = Field(default=3, ge=1, le=10)
    # 0.0 表示接受所有非负余弦分数。tests/test_retrieval.py 的标定样例显示：
    # 确定性 embedding 在模拟语料上相关查询约 0.23~0.40、无关查询 <=0.08；
    # 语料或 embedding 更换后必须重新标定阈值。
    manual_retrieval_min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # 会话记忆按 session id 只在进程内保留最近 N 轮；长期记录写入只追加的
    # JSONL 台账文件。
    session_memory_max_turns: int = Field(default=5, ge=1, le=50)
    memory_ledger_path: str = Field(default="data/memory/approved_actions.jsonl", min_length=1, max_length=256)

    # HTTP API 配置。未配置 API key 时服务拒绝启动（fail-closed）。
    api_key: SecretStr | None = None
    api_rate_limit: int = Field(default=30, ge=1, le=10_000)

    tracing_enabled: bool = False
    tracing_project: str = Field(
        default="industrial-diagnostic-agent-dev", min_length=1, max_length=128
    )
    environment: str = Field(default="development", min_length=1, max_length=64)
    tracing_endpoint: AnyHttpUrl = "https://api.smith.langchain.com"
    # SecretStr 防止密钥意外出现在 repr()、校验错误或常规日志里；
    # 这里只会读取本项目前缀（INDUSTRIAL_AGENT_）的环境变量。
    langsmith_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None

    @field_validator("ollama_base_url")
    @classmethod
    def ollama_must_use_loopback(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        """Ollama 地址必须是 http(s) 回环地址，防止开发流量打到陌生主机。"""
        if value.scheme not in {"http", "https"} or value.host not in {"localhost", "127.0.0.1", "::1", "[::1]"}:
            raise ValueError("ollama_base_url must use http/https with a loopback host")
        return value

    @field_validator("deepseek_base_url")
    @classmethod
    def deepseek_must_use_official_https(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        """DeepSeek 地址强制官方 HTTPS 域名，防止 API key 被发给仿冒端点。"""
        if value.scheme != "https" or value.host != "api.deepseek.com":
            raise ValueError("deepseek_base_url must use https://api.deepseek.com")
        return value

    @field_validator("tracing_endpoint")
    @classmethod
    def tracing_must_use_official_https(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        """Trace 上报地址限定官方 LangSmith HTTPS 域名，防止 Trace（可能含提示词）外流。"""
        if value.scheme != "https" or value.host != "api.smith.langchain.com":
            raise ValueError("tracing_endpoint must use https://api.smith.langchain.com")
        return value

    @model_validator(mode="after")
    def provider_and_structured_method_must_match(self) -> "Settings":
        """结构化输出方式必须与 provider 匹配：Ollama 走 json_schema，DeepSeek 走 function_calling。

        在配置加载期就拦下不兼容组合，避免运行期才出现难以定位的解析失败。
        """
        required_method = "json_schema" if self.provider == "ollama" else "function_calling"
        if self.structured_output_method != required_method:
            raise ValueError(
                f"provider={self.provider} requires structured_output_method={required_method}"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的已校验 Settings 单例，供依赖注入复用。

    lru_cache 保证整个进程只解析一次环境变量与 .env：配置视图全局一致，
    重复调用也没有开销；测试需要隔离配置时可用 cache_clear() 重置。
    """

    return Settings()
