"""Controlled append-only long-term memory.

Only approved maintenance actions may enter the ledger, and only through a
whitelist of structured program-derived fields. Free-form model output, user
questions, and rejected proposals are never persisted. The file is JSONL so
each line can be inspected independently; malformed lines are skipped on read
instead of poisoning the whole history.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ALLOWED_FIELDS = (
    "recorded_at",
    "action_type",
    "request_id",
    "device_id",
    "risk_level",
    "ticket_id",
    "decided_by",
)


class LongTermLedger:
    """Append-only record of executed, human-approved maintenance actions."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def record_approved_action(
        self,
        *,
        request_id: str,
        device_id: str,
        risk_level: str,
        ticket_id: str,
        decided_by: str,
        action_type: str = "schedule_maintenance",
    ) -> dict[str, Any]:
        if not request_id.strip() or not device_id.strip():
            raise ValueError("request_id and device_id must not be empty")
        record = {
            "recorded_at": datetime.now(UTC).isoformat(),
            "action_type": action_type,
            "request_id": request_id,
            "device_id": device_id,
            "risk_level": risk_level,
            "ticket_id": ticket_id,
            "decided_by": decided_by,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return dict(record)

    def history_for_device(self, device_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return the most recent whitelisted records for one device."""

        if limit < 1:
            return []
        records: list[dict[str, Any]] = []
        if not self._path.exists():
            return records
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue  # one damaged line must not poison the history
                if not isinstance(parsed, dict):
                    continue
                if parsed.get("device_id") != device_id:
                    continue
                records.append({field: parsed.get(field) for field in _ALLOWED_FIELDS})
        return records[-limit:]

    def total_records(self) -> int:
        """Count parseable records; damaged lines are not valid history."""

        if not self._path.exists():
            return 0
        count = 0
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue
                count += 1
        return count
