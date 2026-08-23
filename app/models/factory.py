"""Create the configured model without performing a network request.

模型 factory：根据配置中的 provider 构造对应的 LangChain 聊天模型对象
（ChatOllama 或 ChatDeepSeek）。关键约定：构造过程绝不发起网络请求，也绝不
偷偷换成假模型——假模型只应由测试通过依赖注入显式提供。这样"单元测试用
stub、生产用真实 provider"的边界才不会被 factory 悄悄打破。

失败行为：provider 不受支持、或 DeepSeek 缺少 API key 时直接抛 ValueError
快速失败——不降级、不回退到另一个 provider，把配置问题暴露在启动阶段，
而不是拖到第一次 invoke 才报错。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

from app.config.settings import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    """按配置构建真实的 provider 模型。

    假模型刻意由测试经依赖注入传给 build_diagnostic_agent，本 factory 绝不
    替换 provider。Ollama 分支关闭构造期的模型名校验
    （validate_model_on_init=False）；两个分支在构造阶段都不联网——第一次
    真正的网络调用发生在首次 invoke 时。
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
        # deepseek-v4 系列默认开启思考模式，该模式拒绝强制 tool_choice
        # （HTTP 400 "Thinking mode does not support this tool_choice"），
        # 而 function_calling 结构化输出依赖它。默认显式关闭思考：
        # 兼容性必需，且实测更快、更省 token。
        return ChatDeepSeek(
            model=settings.model,
            api_base=str(settings.deepseek_base_url),
            api_key=settings.deepseek_api_key,
            temperature=settings.temperature,
            timeout=settings.timeout_seconds,
            max_retries=settings.deepseek_max_retries,
            **(
                {}
                if not getattr(settings, "deepseek_thinking_disabled", False)
                else {"extra_body": {"thinking": {"type": "disabled"}}}
            ),
        )

    raise ValueError(f"Unsupported model provider: {settings.provider}")
