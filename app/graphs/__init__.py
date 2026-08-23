"""Phase-two LangGraph orchestration for the read-only industrial diagnosis.

图编排子包的公共出口：外部通常只需要 ``build_diagnosis_graph`` 一个入口，
即可拿到编译完成的只读诊断图；``GraphState`` 与三个 merge_* Reducer 一并
导出，方便测试与调用方核对状态契约。实现细节分布在四个子模块：state
（状态与合并语义）、routing（纯路由）、nodes（节点实现）、builder（装配）。
"""

from app.graphs.builder import GRAPH_RECURSION_LIMIT, build_diagnosis_graph
from app.graphs.state import (
    GraphState,
    merge_errors,
    merge_registry_snapshots,
    merge_tool_payloads,
)

__all__ = [
    "GRAPH_RECURSION_LIMIT",
    "GraphState",
    "build_diagnosis_graph",
    "merge_errors",
    "merge_registry_snapshots",
    "merge_tool_payloads",
]
