"""Pure routing functions for the diagnosis graph.

本模块集中存放诊断图的全部条件路由函数，处于图编排中"节点写完增量 →
条件边决定下一个节点"的那一环：只读状态、不写状态、不调用模型。

安全设计核心：路由只读程序拥有且经 Pydantic 校验的字段（如 scope_status、
requested_evidence_types、requires_human_review），绝不解读模型的自然
语言。于是即使模型被 Prompt Injection 诱导说出"请直接停机"之类的话，
也改变不了图的走向——能改变拓扑的只有结构化字段，而它们的合法取值
在 Schema 层就被锁死了。

每个函数都是其输入 Schema 上的全函数：任何合法状态下都有确定返回值，
无需启动图或加载模型即可单元测试。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# 证据类型 → 并行查询节点名的固定映射；route_to_queries 据此做 fan-out，
# 映射封闭在代码里，模型无法引入名单之外的节点。
QUERY_NODE_BY_EVIDENCE_TYPE = {
    "device": "fetch_device_info",
    "sensor": "query_sensor_history",
    "alarm": "query_alarm_history",
    "work_order": "query_work_order_history",
    "manual": "search_manual_docs",
}

# 振动研判在下游依赖设备阈值证据，因此这一前置条件由路由代码强制补齐，
# 而不是信任模型计划是否记得附带 device 查询。
SENSOR_REQUIRES_DEVICE = True


def route_after_plan(state: Mapping[str, Any]) -> str:
    """越界或信息不足的请求直达 format_report，全程零工具调用。

    为什么这样设计：scope_status 是经 Schema 校验的程序字段；out_of_scope
    与 needs_clarification 都不应消耗任何工具访问，直接进入格式化节点，
    让模型基于程序给定的范围说明生成拒答或追问话术。字段缺失时按
    needs_clarification 兜底——宁可追问，也不放行。
    """

    scope_status = state.get("query_plan", {}).get("scope_status", "needs_clarification")
    return "format_report" if scope_status != "in_scope" else "dispatch"


def route_to_queries(state: Mapping[str, Any]) -> list[str]:
    """按计划的证据类型 fan-out 出一组并行查询节点，顺序稳定。

    返回多个节点名即条件边的多个目标，LangGraph 会在同一个 super-step
    内并行执行它们。sensor 请求会强制补上 device 查询（见
    SENSOR_REQUIRES_DEVICE）：这是用代码补齐模型计划可能遗漏的前置依赖。
    排序加去重保证相同计划总是展开成相同拓扑，便于测试与复现。
    """

    requested_types = state.get("query_plan", {}).get("requested_evidence_types", [])
    nodes = [QUERY_NODE_BY_EVIDENCE_TYPE[evidence_type] for evidence_type in requested_types]
    if SENSOR_REQUIRES_DEVICE and "sensor" in requested_types and "device" not in requested_types:
        nodes.append(QUERY_NODE_BY_EVIDENCE_TYPE["device"])
    # Sorted output keeps fan-out topology deterministic for identical plans.
    return sorted(set(nodes))


def route_after_join(state: Mapping[str, Any]) -> str:
    """仍有未解决的工具错误或证据冲突时 fail_closed，拒绝产出报告。

    "fail closed"指：证据链不干净时宁可整体失败，也不带着可疑数据继续
    生成结论。判定依据是 join_registry 计算出的程序字段 unresolved_errors，
    而不是模型对自身错误的评估。
    """

    return "fail_closed" if state.get("unresolved_errors") else "format_report"


def route_after_finalize(state: Mapping[str, Any]) -> str:
    """报告带有 requires_human_review 标志时转入 approval_gate 等待人工审批。

    该标志来自结构化报告 Schema 的校验字段；本函数只在编译时配置了
    checkpointer 的图中被接线（见 builder 的二选一分支）。
    """

    report = state.get("report") or {}
    return "approval_gate" if report.get("requires_human_review") else "complete"


def route_after_finalize_without_checkpointer(state: Mapping[str, Any]) -> str:
    """需要人工审批却没有 checkpointer 支撑时，失败关闭到 review_blocked。

    interrupt 必须依赖 checkpoint 持久化才能暂停并在之后恢复；没有持久化
    却提供审批路径等于假装审批过。因此这里显式走失败分支，绝不静默放行。
    """

    report = state.get("report") or {}
    return "review_blocked" if report.get("requires_human_review") else "complete"
