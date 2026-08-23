"""Compose the phase-two/three diagnosis graph.

Topology (one super-step unless noted):

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

Model calls are bounded by construction: one planner call, one formatter
call. Tool calls are bounded by the five-node fan-out. The only controlled
side effect runs in ``execute_approved_action`` after a validated human
decision. A recursion limit is applied at invoke time as a final backstop.

The approval branch is wired only when a checkpointer is supplied: without
persistence there is no way to resume an interrupt, so the graph refuses to
offer a path that could require one.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agents.diagnostic import Invokable
from app.graphs import nodes, routing
from app.graphs.state import GraphState
from app.schemas.diagnostics import DiagnosisDraft
from app.schemas.query_plan import QueryPlan

GRAPH_RECURSION_LIMIT = 25


def build_diagnosis_graph(
    model: BaseChatModel,
    *,
    structured_output_method: str = "json_schema",
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compile the diagnosis graph around schema-bound planner/formatter calls.

    Passing a checkpointer additionally wires the human-approval branch; the
    caller must then always provide ``configurable.thread_id``.
    """

    planner = _schema_bound(model, QueryPlan, structured_output_method)
    formatter = _schema_bound(model, DiagnosisDraft, structured_output_method)

    builder = StateGraph(GraphState)
    builder.add_node("plan_queries", nodes.make_plan_queries(planner))
    builder.add_node("dispatch", nodes.dispatch)
    for node_name in set(routing.QUERY_NODE_BY_EVIDENCE_TYPE.values()):
        builder.add_node(node_name, nodes.make_query_node(_tool_name_for(node_name)))
    builder.add_node("join_registry", nodes.join_registry)
    builder.add_node("format_report", nodes.make_format_report(formatter))
    builder.add_node("finalize_report", nodes.finalize_report)
    builder.add_node("fail_closed", nodes.fail_closed)
    builder.add_node("review_blocked", nodes.review_blocked)
    builder.add_node("approval_gate", nodes.approval_gate)
    builder.add_node("execute_approved_action", nodes.execute_approved_action)
    builder.add_node("record_rejection", nodes.record_rejection)
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
        # Without persistence an interrupt could never be resumed, so a
        # review-required report fails closed instead of silently completing.
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
    return {
        "fetch_device_info": "get_device_info",
        "query_sensor_history": "query_sensor_history",
        "query_alarm_history": "query_alarm_history",
        "query_work_order_history": "query_work_order_history",
        "search_manual_docs": "search_manual",
    }[node_name]


def _schema_bound(model: BaseChatModel, schema: type, method: str) -> Invokable:
    return model.with_structured_output(schema, method=method, include_raw=False)  # type: ignore[arg-type]
