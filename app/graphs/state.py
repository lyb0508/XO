"""Serializable LangGraph state for the phase-two diagnosis graph.

The state stores raw, reusable, JSON-safe data only. Prompt-specific text and
model objects never enter the graph state so a later checkpointing stage can
persist it without schema changes.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, NotRequired, TypedDict


def merge_tool_payloads(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministic merge for parallel fan-out results.

    Payloads come from program-invoked read-only tools and are already
    JSON-safe. Sorting by canonical JSON makes the merged channel independent
    of node execution order within the same super-step.
    """

    combined = [*left, *right]
    return sorted(combined, key=_canonical_json)


def merge_errors(left: list[str], right: list[str]) -> list[str]:
    """Collect tool failures deterministically regardless of completion order."""

    return sorted({*left, *right})


def merge_registry_snapshots(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministic merge for parallel registry snapshots.

    Conflicting duplicates are intentionally preserved here; the join node is
    the single place that detects and reports canonical conflicts.
    """

    combined = [*left, *right]
    return sorted(combined, key=_canonical_json)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class GraphState(TypedDict):
    """Shared diagnosis state; every field is JSON-serializable."""

    request_id: str
    device_id: str
    question: str

    query_plan: NotRequired[dict[str, Any]]
    tool_payloads: NotRequired[Annotated[list[dict[str, Any]], merge_tool_payloads]]
    tool_errors: NotRequired[Annotated[list[str], merge_errors]]
    registry_entries: NotRequired[Annotated[list[dict[str, Any]], merge_registry_snapshots]]
    unresolved_errors: NotRequired[list[str]]
    draft: NotRequired[dict[str, Any]]
    report: NotRequired[dict[str, Any] | None]
    error: NotRequired[str]
