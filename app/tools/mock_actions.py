"""Simulated, idempotent maintenance action; no external side effects exist.

This is the only "business action" the project can take. It writes nothing
outside process memory and exists so approval, execution, idempotency, and
rejection semantics can be exercised end to end before any real integration.
"""

from __future__ import annotations

import threading
from typing import Any

_ACTION_TYPE = "schedule_maintenance"
_EXECUTION_LEDGER: dict[tuple[str, str], dict[str, Any]] = {}
# LangGraph executes node work on background threads even for sync invoke, so
# guard the process-wide ledger against concurrent check-and-set windows.
_LEDGER_LOCK = threading.Lock()


def execute_maintenance_action(request_id: str, device_id: str) -> dict[str, Any]:
    """Record one simulated maintenance scheduling with an idempotency key.

    The key is (action_type, request_id): retrying the same request can never
    create a second ticket. A repeated call reports ``already_executed`` while
    keeping the original ticket reference stable.
    """

    if not request_id.strip():
        raise ValueError("request_id must not be empty")
    if not device_id.strip():
        raise ValueError("device_id must not be empty")
    key = (_ACTION_TYPE, request_id)
    with _LEDGER_LOCK:
        previous = _EXECUTION_LEDGER.get(key)
        if previous is not None:
            return {
                "status": "already_executed",
                "ticket_id": previous["ticket_id"],
                "action_type": _ACTION_TYPE,
                "request_id": request_id,
                "device_id": previous["device_id"],
            }
        record = {
            "status": "executed",
            "ticket_id": f"MNT-{request_id}",
            "action_type": _ACTION_TYPE,
            "request_id": request_id,
            "device_id": device_id,
        }
        _EXECUTION_LEDGER[key] = record
    return dict(record)


def reset_execution_ledger() -> None:
    """Test-only cleanup so unit tests never observe each other's tickets."""

    with _LEDGER_LOCK:
        _EXECUTION_LEDGER.clear()
