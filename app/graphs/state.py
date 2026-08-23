"""Serializable LangGraph state for the phase-two diagnosis graph.

本模块定义整张诊断图唯一的共享状态 GraphState，以及三个供并行合并使用的
Reducer 函数。可以把这里理解为所有节点之间传递数据的"总线"：

- 状态如何流转：LangGraph 按 super-step 推进——同一批并行节点各自返回
  字段增量，框架再用字段 Annotated 元数据里指定的 Reducer 把增量合并进
  总状态；不带 Reducer 的字段则是"最后写入者获胜"的直接覆盖；
- 副作用边界：这里只存放原始、可复用、JSON 可序列化的数据。拼接好的
  Prompt 文本与模型对象一律不进状态，因此未来接入 checkpoint 持久化时
  无需改动 Schema；
- 失败行为：Reducer 本身是纯排序合并，不会失败；工具失败以字符串形式
  进入 tool_errors 通道，是否整体失败由后续 join 节点与路由决定。
"""

from __future__ import annotations

import json
from typing import Annotated, Any, NotRequired, TypedDict


def merge_tool_payloads(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """并行 fan-out 结果的确定性合并（Reducer）。

    同一 super-step 内多个查询节点都会往 tool_payloads 写入一份增量列表，
    框架按完成顺序两两调用本函数合并。按 canonical JSON 排序后，通道的
    最终内容与节点执行顺序无关——同样一批增量永远合并出同样的序列，
    这对 checkpoint 回放与评测稳定性至关重要。

    载荷来自程序调用的只读工具，本身就是 JSON 安全的；本函数是纯函数，
    不会抛错，坏数据由下游 join_registry 节点检测。
    """

    combined = [*left, *right]
    return sorted(combined, key=_canonical_json)


def merge_errors(left: list[str], right: list[str]) -> list[str]:
    """工具失败的确定性收集（Reducer）。

    先用 set 去重再排序：无论哪个并行查询先报错，tool_errors 的内容与
    顺序都一致。这里只负责忠实记录"发生过什么错误"；要不要因此终止
    流程，由 join_registry 与 route_after_join 决定。
    """

    return sorted({*left, *right})


def merge_registry_snapshots(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """证据注册表快照的确定性合并（Reducer）。

    与 merge_tool_payloads 相同，按 canonical JSON 排序保证顺序无关。
    内容冲突的重复快照在这里故意保留、不去重：检测并上报 canonical
    冲突是 join_registry 节点的单一职责，Reducer 层只做忠实收集。
    """

    combined = [*left, *right]
    return sorted(combined, key=_canonical_json)


def _canonical_json(payload: Any) -> str:
    """生成按键排序的紧凑 JSON 文本，作为跨嵌套结构的稳定比较键。"""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class GraphState(TypedDict):
    """诊断流程的共享状态；所有字段均可 JSON 序列化。

    字段分三段阅读：

    - 输入（必填）：request_id 全局唯一请求号；device_id 程序侧确定的
      目标设备号（模型无权改写）；question 用户原始问题；session_id
      可选，用于关联短期会话记忆；
    - 中间产物（NotRequired）：query_plan 是校验后的查询计划字典；
      tool_payloads / tool_errors / registry_entries 带 Annotated
      Reducer，并行节点各写一份增量、由框架自动排序合并；
      unresolved_errors 由 join_registry 整体覆盖写入；draft 是格式化
      模型的结构化草稿；report 是最终报告（失败分支显式写 None）；
      error 是终态错误消息；
    - 审批阶段（NotRequired）：proposed_action 待审动作提案、approval
      人工决策、action_audit 动作执行审计结果。

    合并语义小结：带 Reducer 的字段在同一 super-step 内多次写入时排序
    合并、顺序无关；不带 Reducer 的字段遵循 LangGraph"最后写入者获胜"
    的覆盖语义，因此约定每条执行路径上每个字段只由一个节点写入一次，
    消除覆盖歧义。
    """

    request_id: str
    device_id: str
    question: str
    session_id: NotRequired[str]

    query_plan: NotRequired[dict[str, Any]]
    tool_payloads: NotRequired[Annotated[list[dict[str, Any]], merge_tool_payloads]]
    tool_errors: NotRequired[Annotated[list[str], merge_errors]]
    registry_entries: NotRequired[Annotated[list[dict[str, Any]], merge_registry_snapshots]]
    unresolved_errors: NotRequired[list[str]]
    draft: NotRequired[dict[str, Any]]
    report: NotRequired[dict[str, Any] | None]
    error: NotRequired[str]

    # 第三阶段 human-in-the-loop 字段：每条执行路径上每个键只由一个节点
    # 写入（两个审计写入节点位于互斥分支上），因此无需 Reducer，直接
    # 覆盖即为确定性行为。
    proposed_action: NotRequired[dict[str, Any]]
    approval: NotRequired[dict[str, Any]]
    action_audit: NotRequired[dict[str, Any]]
