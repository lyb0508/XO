"""Phase-four retrieval: manual RAG with citation metadata.

第四阶段检索模块的对外出口：向量库、检索结果与 embedding 工厂都从这里
导出；上层只需 import 本包即可搭建带引用元数据的手册 RAG。
"""

from app.retrieval.embeddings import DeterministicCharacterEmbeddings, create_embeddings
from app.retrieval.store import ManualVectorStore, RetrievedChunk, build_default_store

__all__ = [
    "DeterministicCharacterEmbeddings",
    "ManualVectorStore",
    "RetrievedChunk",
    "build_default_store",
    "create_embeddings",
]
