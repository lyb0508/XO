"""Retrieval entry point wiring settings, store, and citation metadata.

本模块是检索能力的组装入口：把 ``Settings`` 中的配置、embedding provider
和内存向量库串起来，对外暴露"建库"与"带引用检索"两个函数。

数据来源：语料来自 ``app.tools.mock_data`` 的固定手册；provider 与阈值等
参数来自 ``app.config.settings``。

副作用边界：建库对象本身无网络请求；但 provider=ollama 时的第一次嵌入
调用会真实连接 Ollama 服务。

失败行为：Ollama 不可达时在摄取阶段大声抛错；查询为空或 top_k 非法时
返回空列表而不是异常。
"""

from __future__ import annotations

from app.config.settings import Settings
from app.retrieval.store import ManualVectorStore, RetrievedChunk, build_default_store


def create_manual_store(settings: Settings) -> ManualVectorStore:
    """按配置构建手册向量库并摄取 mock 语料。

    注意网络时点：构建 embeddings 对象不做任何 I/O，但随后的语料摄取会
    调用一次 ``embed_documents``——provider=ollama 时这一步会真实连接所
    配置的 Ollama 端点，不可达时直接失败，绝不静默降级。
    """

    embeddings = _create_embeddings(settings)
    return build_default_store(embeddings)


def retrieve_manual_citations(
    store: ManualVectorStore,
    query: str,
    *,
    device_id: str,
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    """按给定阈值检索一次手册内容，返回可引用的 chunk 列表。

    top_k 与 min_score 由调用方显式传入（通常取自 Settings），让检索行为
    在每个调用点都清晰可见，而不是藏在库的默认值里；min_score 必须来自
    正负样例校准的结果，不能随意指定。
    """

    return store.retrieve(query, device_id=device_id, top_k=top_k, min_score=min_score)


def _create_embeddings(settings: Settings):
    """从 Settings 读出 provider/model/base_url 并构建 embedding 实例。"""

    from app.retrieval.embeddings import create_embeddings

    return create_embeddings(
        settings.embeddings_provider,
        model=settings.embeddings_model,
        base_url=str(settings.ollama_base_url),
    )
