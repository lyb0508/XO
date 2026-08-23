"""In-memory vector store and manual ingestion with source metadata.

本模块实现一个极简的内存向量库，负责手册语料的摄取（ingest）与检索。

数据来源：调用 ``build_default_store`` 时从 ``app.tools.mock_data`` 摄取
固定的 mock 手册；不依赖任何外部向量数据库。

设计要点：每个 chunk 都强制携带 doc_id、section、version、device_id 等
引用元数据，保证检索结果在诊断报告里可溯源；检索按 device_id 过滤，
避免把别的设备的手册内容混进证据。

失败行为：摄取时字段缺失会抛 ValueError；嵌入后端失败发生在任何状态
变更之前，因此不会留下"有 chunk 无向量"的半成品库。min_score 阈值不是
拍脑袋定的，必须用正负样例校准（见 evaluations 阶段）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from langchain_core.embeddings import Embeddings


def _cosine(left: list[float], right: list[float]) -> float:
    """计算余弦相似度；维度不一致或任一向量为零向量时返回 0.0。

    返回 0.0 而不是抛错，是为了让"无法比较"统一走低分路径被 min_score
    过滤掉，而不是让整个检索请求失败。
    """
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


@dataclass(frozen=True)
class RetrievedChunk:
    """一条检索命中及其完整引用元数据。

    doc_id / section / version / device_id 让报告里的每段引用都能追溯到
    具体文档、章节和版本；frozen 保证命中结果一旦返回就不会被调用方篡改。
    """

    doc_id: str
    section: str
    title: str
    content: str
    version: str
    device_id: str
    score: float


class ManualVectorStore:
    """基于余弦相似度的极简内存向量库。

    chunks 与向量两条平行列表按下标一一对应，仅用于学习里程碑；
    不追求性能、过滤能力和持久化，接入真实向量库时整体替换即可。
    """

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._chunks: list[dict[str, str]] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def ingest(self, sections: list[Mapping[str, Any]]) -> int:
        """摄取手册章节；每个章节必须携带自己的 device_id 与 version 标识。

        注意：这里只校验标识非空，不查重、不去重——同一章节重复摄取会
        产生重复条目。执行顺序刻意安排为"先校验、再嵌入、最后入库"：
        嵌入调用发生在任何状态变更之前，这样嵌入后端失败时不会留下有
        chunk 却没有向量的半成品状态。
        """

        new_chunks = []
        for section in sections:
            doc_id = _required(section.get("source_id"), "source_id")
            title = _required(section.get("title"), "title")
            content = _required(section.get("content"), "content")
            version = _required(section.get("version"), "version")
            device_id = _required(section.get("device_id"), "device_id")
            new_chunks.append(
                {
                    "doc_id": doc_id,
                    "section": doc_id.rsplit(":", 1)[-1],
                    "title": title,
                    "content": content,
                    "version": version,
                    "device_id": device_id,
                }
            )
        vectors = self._embeddings.embed_documents(
            [f"{chunk['title']} {chunk['content']}" for chunk in new_chunks]
        )
        self._chunks.extend(new_chunks)
        self._vectors.extend(vectors)
        return len(new_chunks)

    def retrieve(
        self,
        query: str,
        *,
        device_id: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """按阈值检索指定设备的手册 chunk，最多返回 ``top_k`` 条。

        结果按分数降序排列，并以 doc_id 作为确定性 tiebreaker，保证相同
        输入必然得到相同顺序——这是轨迹评测可复现的前提。

        ``min_score`` 阈值必须用本项目正负样例校准后由调用方显式传入，
        不能凭感觉设一个"通用相似度阈值"：不同 embedding 的分数量纲完全
        不同，未校准的阈值要么漏掉正确证据，要么放进无关噪声。
        """

        if not query.strip() or top_k < 1:
            return []
        query_vector = self._embeddings.embed_query(query)
        if not any(query_vector):
            return []  # 零向量没有方向信息，余弦相似度无意义，直接视为无结果。
        scored = [
            (_cosine(query_vector, vector), chunk)
            for vector, chunk in zip(self._vectors, self._chunks)
            if chunk["device_id"] == device_id
        ]
        hits = [
            RetrievedChunk(
                doc_id=chunk["doc_id"],
                section=chunk["section"],
                title=chunk["title"],
                content=chunk["content"],
                version=chunk["version"],
                device_id=chunk["device_id"],
                score=round(score, 6),
            )
            for score, chunk in scored
            if score >= min_score
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return hits[:top_k]

    def snapshot(self) -> Mapping[str, Any]:
        """返回浅只读视图，供诊断与测试检查库内状态。

        只防止增删条目：外层映射与 chunks 元组不可变，但元组内单个
        chunk 仍是普通 dict，其字段内容技术上仍可被调用方改写。
        """

        return MappingProxyType({"chunks": tuple(self._chunks), "count": len(self._chunks)})


def _required(value: Any, field: str) -> str:
    """校验并清洗单个必填字段；缺失或空白即抛 ValueError。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manual section field {field} must be a non-empty string")
    return value.strip()


def build_default_store(embeddings: Embeddings) -> ManualVectorStore:
    """把固定的 mock 手册摄取进一个全新 store，作为默认检索库。"""

    from app.tools.mock_data import MANUAL_SECTIONS

    store = ManualVectorStore(embeddings)
    store.ingest(list(MANUAL_SECTIONS))
    return store
