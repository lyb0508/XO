"""Pure routing functions for the diagnosis graph.

Routing never inspects model prose: it reads only validated, program-owned
state fields. Every function here is a total function over its declared input
schema and is unit-testable without a graph or a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

QUERY_NODE_BY_EVIDENCE_TYPE = {
    "device": "fetch_device_info",
    "sensor": "query_sensor_history",
    "alarm": "query_alarm_history",
    "work_order": "query_work_order_history",
    "manual": "search_manual_docs",
}

# Vibration diagnosis is gated on device threshold evidence downstream, so the
# router enforces that prerequisite in code instead of trusting plan contents.
SENSOR_REQUIRES_DEVICE = True


def route_after_plan(state: Mapping[str, Any]) -> str:
    """Send out-of-scope work straight to formatting with zero tool calls."""

    scope_status = state.get("query_plan", {}).get("scope_status", "needs_clarification")
    return "format_report" if scope_status != "in_scope" else "dispatch"


def route_to_queries(state: Mapping[str, Any]) -> list[str]:
    """Fan out one parallel node per requested evidence type, in stable order."""

    requested_types = state.get("query_plan", {}).get("requested_evidence_types", [])
    nodes = [QUERY_NODE_BY_EVIDENCE_TYPE[evidence_type] for evidence_type in requested_types]
    if SENSOR_REQUIRES_DEVICE and "sensor" in requested_types and "device" not in requested_types:
        nodes.append(QUERY_NODE_BY_EVIDENCE_TYPE["device"])
    # Sorted output keeps fan-out topology deterministic for identical plans.
    return sorted(set(nodes))


def route_after_join(state: Mapping[str, Any]) -> str:
    """Fail closed when any tool error or canonical conflict remains unresolved."""

    return "fail_closed" if state.get("unresolved_errors") else "format_report"


def route_after_finalize(state: Mapping[str, Any]) -> str:
    """Send reports flagged for human review to the approval gate."""

    report = state.get("report") or {}
    return "approval_gate" if report.get("requires_human_review") else "complete"
