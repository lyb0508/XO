"""Embedding providers for manual retrieval.

本模块提供两种实现同一个标准 LangChain ``Embeddings`` 接口的向量器：

* ``DeterministicCharacterEmbeddings`` 把重叠的字符 bigram 哈希进定长向量。
  它完全没有任何语义质量，存在的意义是让离线测试能在不依赖模型的情况下
  验证排序、阈值和元数据契约；绝不能对外宣称它是"语义搜索"。

* ``OllamaEmbeddings`` 由工厂函数按配置构建，供真实运行使用。构建对象本身
  不产生任何网络请求；第一次真正的 embed 调用（摄取或查询）才会连接所
  配置的 Ollama 端点，连不上时大声失败而不是静默降级。
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings


class DeterministicCharacterEmbeddings(Embeddings):
    """基于字符 bigram hash 的确定性向量器；仅限离线测试使用。

    为什么这样设计：真实 embedding 模型需要网络或本地推理，会让单元测试
    变慢且不可复现；这个实现的输出只由输入文本决定，同一文本永远得到同一
    向量，适合验证"检索契约"（排序、阈值过滤、引用元数据），但它的向量
    不携带语义——近义词不会相近，别用它评估检索质量。

    失败行为：dimensions < 32 时构造即抛 ValueError，避免桶太少导致哈希
    碰撞把所有向量挤成几乎相同的形状。
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32:
            raise ValueError("embedding dimensions must be at least 32")
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        """把文本转成 L2 归一化后的特征哈希向量（hashing trick 思路）。

        流程：小写并去掉全部空白 -> 取相邻字符 bigram -> 每个 bigram 用
        sha256 映射到一个固定桶，并用摘要的奇偶性决定 +1/-1（正负号能抵消
        碰撞带来的系统性偏置）-> 最后除以模长归一化。归一化之后余弦相似度
        就等价于点积。空串返回全零向量，表示"无信息"。
        """
        normalized = "".join(text.lower().split())
        vector = [0.0] * self.dimensions
        if not normalized:
            return vector
        for index in range(len(normalized) - 1):
            bigram = normalized[index : index + 2]
            digest = hashlib.sha256(bigram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档语料（ingest 阶段调用）。"""

        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询（retrieve 阶段调用）。"""

        return self._vector(text)


def create_embeddings(provider: str, *, model: str, base_url: str) -> Embeddings:
    """按配置构建 embedding provider；工厂本身不发起任何网络请求。

    失败行为：provider 不在支持列表时抛 ValueError；Ollama 端点不可达的
    错误会延迟到第一次 embed 调用时才出现。
    """

    # 延迟导入：未安装 langchain-ollama 时仍可使用 deterministic provider。
    from langchain_ollama import OllamaEmbeddings

    if provider == "ollama":
        return OllamaEmbeddings(model=model, base_url=base_url)
    if provider == "deterministic":
        return DeterministicCharacterEmbeddings()
    raise ValueError(f"Unsupported embeddings provider: {provider}")
