"""Explicit, redacted LangSmith tracing for the first learning milestone.

本模块提供显式、经过脱敏的 LangSmith 追踪封装。

核心设计（fail-closed）：追踪默认关闭——只有 ``settings.tracing_enabled``
明确为真时才创建 Client；关闭时连 LangSmith client 都不会构建，更不会有
任何网络请求。启用时强制要求 API key 且必须显式传入配置好的 Client，
环境变量无法悄悄改变 trace 的目的地。

脱敏边界：所有上传的 inputs / outputs / metadata 都经过 ``redact_payload``
（redaction），Client 级 anonymizer 也指向同一函数；敏感键名命中 allowlist
之外的黑名单模式即整值替换为占位符。

flush 时机：``tracing_run`` 上下文退出时立即 flush 并 close——业务路径上
投递失败会抛 ``TraceDeliveryFailure``；若业务异常已在飞行中，则降级为
warning，绝不掩盖原始异常。
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


# 脱敏后的固定占位符：替换后的文本仍能看出"这里曾有敏感值"及其类型，
# 便于在 Trace 里发现泄露企图，又不暴露真实内容。
REDACTED_SECRET = "[REDACTED_SECRET]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"
# 以下正则按"键名黑名单 -> 赋值语句 -> Bearer -> 常见 key 格式 -> 显式哨兵
# -> 邮箱 -> 手机号"的层次覆盖最常见的密钥与个人信息形态。
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
# metadata/tags 只允许这种安全字符集，防止换行、引号等被塞进上传内容。
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9._:-]+$")


class _StrictTraceModel(BaseModel):
    """Trace 元数据刻意保持极小集合，并拒绝一切未来新增的自由字段。

    ``extra="forbid"`` 是 allowlist 策略的代码化：任何想往 trace 里多塞一个
    字段的改动都必须先修改这里的模型定义并接受审查。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


class RunContext(_StrictTraceModel):
    """把一次 CLI 调用关联到一条 trace 所需的标识符。"""

    request_id: str = Field(min_length=1, max_length=128)
    thread_id: str = Field(min_length=1, max_length=128)

    @field_validator("request_id", "thread_id")
    @classmethod
    def identifiers_are_safe(cls, value: str) -> str:
        if not _SAFE_VALUE.fullmatch(value):
            raise ValueError("trace identifiers may contain only letters, digits, . _ : or -")
        return value


class TraceMetadata(_StrictTraceModel):
    """允许离开本进程的 trace metadata 的完整 allowlist。

    只有这六个字段可以上传，且每个值都通过字符集校验——保证不会把用户
    输入、密钥或其他敏感数据借道 metadata/tags 泄露到 LangSmith。
    """

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
        """只从固定的、已校验的配置值构建 tags（环境/提供商/Agent 版本）。"""

        return [
            f"environment:{self.environment}",
            f"provider:{self.provider}",
            f"agent:{self.agent_version}",
        ]


def _redact_text(value: str) -> str:
    """按已知格式掩掉密钥与个人信息，同时不破坏文本整体结构。"""

    value = _BEARER_SECRET.sub(REDACTED_SECRET, value)
    # 必须先处理 Bearer 凭据：否则 ``Authorization: Bearer xxx`` 会被赋值
    # 规则只遮住 "Bearer" 这个词，把真正的 token 留在原文里。
    value = _ASSIGNMENT_SECRET.sub(REDACTED_SECRET, value)
    value = _API_KEY_SECRET.sub(REDACTED_SECRET, value)
    value = _EXPLICIT_SECRET_SENTINEL.sub(REDACTED_SECRET, value)
    value = _EMAIL.sub(REDACTED_EMAIL, value)
    return _CN_MOBILE.sub(REDACTED_PHONE, value)


def redact_payload(payload: Any) -> Any:
    """递归脱敏，返回结构等价但已知敏感值被掩掉的副本。

    用于 LangSmith 的 inputs、outputs、metadata 以及 Client 级 anonymizer。
    关键规则：遇到 ``SecretStr`` 直接替换为占位符，绝不调用
    ``get_secret_value()``；Mapping 的键名命中敏感模式时整个值一起替换，
    而不是只递归处理值。
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
    """错误消息先脱敏再外传，避免异常文本把密钥带进日志或 Trace。"""

    return str(redact_payload(str(error))) or error.__class__.__name__


class TraceDeliveryFailure(RuntimeError):
    """本地显式信号：一条已启用的 trace 未能成功投递到 LangSmith。"""


def _close_client(client: Client, *, preserve_business_error: bool) -> None:
    """flush 并关闭 client，同时绝不替换（掩盖）Agent/业务异常。

    ``preserve_business_error=True`` 表示调用方有业务异常正在向外传播：
    此时投递失败只降级为 warning，让原始异常继续抛出；否则投递失败本身
    就是需要暴露的问题，包装成 ``TraceDeliveryFailure`` 抛出。
    """

    errors: list[BaseException] = []
    for operation in (client.flush, client.close):
        try:
            operation()
        except BaseException as error:  # SDK 清理失败不能掩盖诊断主流程的异常。
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
    """仅当配置明确允许时，启用一次安全的 LangSmith trace。

    关闭路径：``settings.tracing_enabled`` 为假时本上下文是 no-op——不建
    client、不发请求。这是 fail-closed 默认。

    启用路径：必须配置 API key（缺失或空白都直接报错），并且显式传入按
    配置构建的 Client，环境变量无法悄悄改变 trace 目的地；inputs/outputs/
    metadata 与 anonymizer 全部挂上 ``redact_payload``。远端投递失败默认
    抛出；若业务异常已在飞行中则降级为 warning 并保留原异常。
    """

    if not settings.tracing_enabled:
        # ``enabled=False`` 显式覆盖 LANGSMITH_TRACING 以及旧版
        # LANGCHAIN_TRACING_V2 环境变量；``parent=False`` 防止继承外层
        # 第三方库开启的 trace。离开代码块时 SDK 自动恢复先前上下文。
        with tracing_context(enabled=False, parent=False):
            yield
        return

    if settings.langsmith_api_key is None:
        raise RuntimeError("LangSmith tracing is enabled but no API key is configured")

    # 明文 key 只在这个局部变量里停留到"拒绝空配置 + 构建显式配置的 SDK
    # Client"为止，之后不再持有。
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
        # flush 时机：上下文退出即 flush+close。except 分支说明业务异常在
        # 飞行中，投递失败降级为 warning；else 分支说明业务正常，投递失败
        # 必须作为 TraceDeliveryFailure 暴露出来。
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
