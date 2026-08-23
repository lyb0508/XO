"""Deterministic merge and routing behavior for the phase-two graph."""

from __future__ import annotations

import json

import pytest

from app.graphs.state import merge_errors, merge_registry_snapshots, merge_tool_payloads
from app.graphs.routing import route_after_join, route_after_plan, route_to_queries

# Frozen literal expectations, independent of the production routing table.
EXPECTED_NODE_NAMES = {
    "device": "fetch_device_info",
    "sensor": "query_sensor_history",
    "alarm": "query_alarm_history",
    "work_order": "query_work_order_history",
    "manual": "search_manual_docs",
}


@pytest.mark.unit
def test_payload_merge_is_order_insensitive_and_stable() -> None:
    import json

    first = {"source_id": "b", "points": [1]}
    second = {"source_id": "a", "alarms": []}
    left_to_right = merge_tool_payloads([first], [second])
    right_to_left = merge_tool_payloads([second], [first])
    assert left_to_right == right_to_left
    assert left_to_right == sorted(
        [first, second],
        key=lambda payload: json.dumps(payload, sort_keys=True, ensure_ascii=False),
    )


@pytest.mark.unit
def test_error_merge_deduplicates_and_sorts() -> None:
    merged = merge_errors(["b: x", "a: y"], ["b: x"])
    assert merged == ["a: y", "b: x"]


def _snapshot(evidence_id: str, summary: str) -> dict[str, object]:
    return {
        "evidence": {
            "evidence_id": evidence_id,
            "evidence_type": "sensor",
            "source_id": "s1",
            "summary": summary,
            "observed_at": None,
            "version": None,
        },
        "device_id": "PUMP-003",
        "tool_name": "query_sensor_history",
        "facts": {"metric": "vibration_mm_s", "value": 1.0},
    }


@pytest.mark.unit
def test_registry_snapshot_merge_is_order_insensitive() -> None:
    first = _snapshot("e-2", "later")
    second = _snapshot("e-1", "earlier")
    assert merge_registry_snapshots([first], [second]) == merge_registry_snapshots([second], [first])


@pytest.mark.unit
def test_join_reports_conflicting_facts_for_same_evidence_id() -> None:
    from app.graphs.nodes import join_registry

    conflicting = _snapshot("e-1", "different facts")
    conflicting["facts"] = {"metric": "vibration_mm_s", "value": 99.0}
    state = {"registry_entries": [_snapshot("e-1", "original"), conflicting, _snapshot("e-1", "original")]}
    update = join_registry(state)
    # The canonical (lexicographically first) snapshot survives exactly once.
    assert len(update["registry_entries"]) == 1
    assert update["unresolved_errors"] == ["conflicting evidence id e-1"]
    assert route_after_join({"unresolved_errors": update["unresolved_errors"]}) == "fail_closed"


@pytest.mark.unit
def test_join_deduplicates_identical_snapshots_without_conflict() -> None:
    from app.graphs.nodes import join_registry

    same = _snapshot("e-1", "original")
    update = join_registry({"registry_entries": [same, dict(same)]})
    assert update["registry_entries"] == [same]
    assert update["unresolved_errors"] == []


@pytest.mark.unit
def test_route_after_plan_shortcuts_non_in_scope() -> None:
    in_scope = {"query_plan": {"scope_status": "in_scope"}}
    off_scope = {"query_plan": {"scope_status": "out_of_scope"}}
    unclear = {"query_plan": {}}
    assert route_after_plan(in_scope) == "dispatch"
    assert route_after_plan(off_scope) == "format_report"
    assert route_after_plan(unclear) == "format_report"


@pytest.mark.unit
def test_route_to_queries_fans_out_requested_types_in_stable_order() -> None:
    state = {
        "query_plan": {
            "requested_evidence_types": ["manual", "alarm", "sensor"],
        }
    }
    nodes = route_to_queries(state)
    # Sensor requests force the device threshold lookup as a code-level gate prerequisite.
    assert nodes == [
        "fetch_device_info",
        "query_alarm_history",
        "query_sensor_history",
        "search_manual_docs",
    ]


@pytest.mark.unit
def test_every_evidence_type_maps_to_its_own_frozen_node_name() -> None:
    from app.graphs.routing import QUERY_NODE_BY_EVIDENCE_TYPE

    assert QUERY_NODE_BY_EVIDENCE_TYPE == EXPECTED_NODE_NAMES


@pytest.mark.unit
def test_route_to_queries_without_sensor_keeps_exact_types() -> None:
    nodes = route_to_queries({"query_plan": {"requested_evidence_types": ["device", "manual"]}})
    assert nodes == [EXPECTED_NODE_NAMES["device"], EXPECTED_NODE_NAMES["manual"]]


@pytest.mark.unit
def test_route_after_join_fails_closed_only_on_unresolved_errors() -> None:
    assert route_after_join({"unresolved_errors": ["x"]}) == "fail_closed"
    assert route_after_join({}) == "format_report"
