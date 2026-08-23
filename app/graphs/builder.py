"""Compose the phase-two/three diagnosis graph.

本模块是图编排的"施工图"：把 nodes.py 的节点与 routing.py 的路由装配成
一张可编译运行的 StateGraph，并按需接入 checkpointer、RAG 手册库与记忆
组件。理解本文件即可看懂整条链路的物理形状。

拓扑（除注明外，同一行内的分支属于同一个 super-step）:

    START -> plan_queries --> dispatch  -+-> fetch_device_info     ----+
                          |             +-> query_sensor_history    |
                          |             +-> query_alarm_history     | parallel
                          |             +-> query_work_order_history| fan-out
                          |             +-> search_manual_docs    ----+
                          |                  all -> join_registry
                          +-> format_report (out-of-scope shortcut)
    join_registry -> fail_closed            when errors/conflicts remain
                   -> format_report          otherwise
    format_report -> finalize_report
    finalize_report -> complete                when no human review is required
                    -> approval_gate           otherwise (phase three, needs checkpointer)
    approval_gate -> execute_approved_action   approved / modified decisions
                   -> record_rejection          rejected decision

调用上限在结构上就已锁定：模型调用恰好两次（规划一次、格式化一次），
工具调用次数由五路 fan-out 决定；唯一受控副作用位于
execute_approved_action，且必须先经 interrupt 取得合法人工决策。
GRAPH_RECURSION_LIMIT 则作为 invoke 时的最后一道步数保险丝。

审批分支只在传入 checkpointer 时接线：没有持久化就无法恢复 interrupt，
与其提供一条注定无法完成的路径，不如在编译期就拒绝提供。

失败行为：装配错误（如未知节点名）会在 compile 阶段立即暴露；运行期
失败由 fail_closed / review_blocked 节点接管，绝不带病产出报告。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents.diagnostic import Invokable
from app.graphs import nodes, routing
from app.graphs.state import GraphState
from app.schemas.diagnostics import DiagnosisDraft
from app.schemas.query_plan import QueryPlan

# invoke 时施加的图步数上限：作为结构性调用上限之外的最后防线，
# 防止任何意外循环无限执行。
GRAPH_RECURSION_LIMIT = 25


def build_diagnosis_graph(
    model: BaseChatModel,
    *,
    structured_output_method: str = "json_schema",
    checkpointer: BaseCheckpointSaver | None = None,
    manual_store: Any = None,
    manual_top_k: int = 3,
    manual_min_score: float = 0.0,
    session_memory: Any = None,
    ledger: Any = None,
):
    """围绕受 Schema 约束的规划/格式化调用，编译出完整的诊断图。

    关键参数语义：
    - checkpointer：传入后会额外接通 human-approval 分支；此后调用方
      每次 invoke 都必须提供 ``configurable.thread_id``，否则中断无法
      定位到具体会话；
    - manual_store：提供时手册证据节点从关键词匹配切换为 embedding 检索；
    - session_memory / ledger：分别启用有界的短期会话召回与受控的
      长期动作台账。

    未传 checkpointer 时，凡需要人工审批的报告都会走 review_blocked
    失败关闭，而不是悄悄完成。
    """

    planner = _schema_bound(model, QueryPlan, structured_output_method)
    formatter = _schema_bound(model, DiagnosisDraft, structured_output_method)

    builder = StateGraph(GraphState)
    builder.add_node(
        "plan_queries",
        nodes.make_plan_queries(planner, session_memory=session_memory, ledger=ledger),
    )
    builder.add_node("dispatch", nodes.dispatch)
    for node_name in set(routing.QUERY_NODE_BY_EVIDENCE_TYPE.values()):
        if node_name == "search_manual_docs" and manual_store is not None:
            builder.add_node(
                node_name,
                nodes.make_manual_retrieval_node(
                    manual_store,
                    top_k=manual_top_k,
                    min_score=manual_min_score,
                ),
            )
        else:
            builder.add_node(node_name, nodes.make_query_node(_tool_name_for(node_name)))
    builder.add_node("join_registry", nodes.join_registry)
    builder.add_node("format_report", nodes.make_format_report(formatter))
    builder.add_node("finalize_report", nodes.finalize_report)
    builder.add_node("fail_closed", nodes.fail_closed)
    builder.add_node("review_blocked", nodes.review_blocked)
    builder.add_node("approval_gate", nodes.approval_gate)
    builder.add_node(
        "execute_approved_action",
        nodes.make_execute_approved_action(ledger=ledger),
    )
    builder.add_node("record_rejection", nodes.record_rejection)
    # complete 是终点前的空节点：让三条成功路径（免审批完成、执行后、
    # 拒绝后）统一汇到这里再进入 END，便于观测与统计。
    builder.add_node("complete", lambda state: {})

    builder.add_edge(START, "plan_queries")
    builder.add_conditional_edges(
        "plan_queries",
        routing.route_after_plan,
        {"dispatch": "dispatch", "format_report": "format_report"},
    )
    builder.add_conditional_edges("dispatch", routing.route_to_queries)
    for node_name in set(routing.QUERY_NODE_BY_EVIDENCE_TYPE.values()):
        builder.add_edge(node_name, "join_registry")
    builder.add_conditional_edges(
        "join_registry",
        routing.route_after_join,
        {"fail_closed": "fail_closed", "format_report": "format_report"},
    )
    builder.add_edge("format_report", "finalize_report")
    if checkpointer is None:
        # 没有持久化就无法恢复 interrupt，因此需要人工审批的报告直接
        # 失败关闭，而不是被悄悄当作已完成。
        builder.add_conditional_edges(
            "finalize_report",
            routing.route_after_finalize_without_checkpointer,
            {"review_blocked": "review_blocked", "complete": "complete"},
        )
        builder.add_edge("complete", END)
    else:
        builder.add_conditional_edges(
            "finalize_report",
            routing.route_after_finalize,
            {"approval_gate": "approval_gate", "complete": "complete"},
        )
        builder.add_edge("execute_approved_action", "complete")
        builder.add_edge("record_rejection", "complete")
        builder.add_edge("complete", END)
    builder.add_edge("fail_closed", END)
    compile_kwargs = {} if checkpointer is None else {"checkpointer": checkpointer}
    return builder.compile(**compile_kwargs)


def _tool_name_for(node_name: str) -> str:
    """节点名 → 工具名的静态映射；查询节点据此从白名单取真实工具函数。"""
    return {
        "fetch_device_info": "get_device_info",
        "query_sensor_history": "query_sensor_history",
        "query_alarm_history": "query_alarm_history",
        "query_work_order_history": "query_work_order_history",
        "search_manual_docs": "search_manual",
    }[node_name]


def _schema_bound(model: BaseChatModel, schema: type, method: str) -> Invokable:
    """把裸模型包装成"产出符合 Schema 的结构化结果"的可调用体。

    include_raw=True 让解析失败不再以异常形式逃逸：调用方拿到
    ``{"raw": 原始消息, "parsed": 校验实例或 None, "parsing_error": 说明}``，
    由节点层的规范化与重试逻辑接管修复，而不是直接输掉整次诊断。
    """
    return model.with_structured_output(schema, method=method, include_raw=True)  # type: ignore[arg-type]
