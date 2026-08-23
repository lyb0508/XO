"""Memory contracts: bounded session recall and the controlled long-term ledger."""

from __future__ import annotations

import json

import pytest

from app.memory.ledger import LongTermLedger
from app.memory.session import SessionMemory


@pytest.mark.unit
def test_session_memory_is_bounded_and_isolated() -> None:
    memory = SessionMemory(max_turns=2)
    for index in range(4):
        memory.append_turn(
            "session-a",
            question=f"问题{index}",
            device_id="PUMP-003",
            risk_level="medium",
            summary=f"摘要{index}",
        )
    turns = memory.recent_turns("session-a")
    assert len(turns) == 2
    assert turns[-1]["question"] == "问题3"
    assert memory.recent_turns("session-b") == []
    memory.forget("session-a")
    assert memory.snapshot() == {}


@pytest.mark.unit
def test_session_memory_clips_and_copies() -> None:
    memory = SessionMemory()
    long_question = "问" * 500
    memory.append_turn(
        "s",
        question=long_question,
        device_id="D",
        risk_level="low",
        summary="总结" * 300,
    )
    stored = memory.recent_turns("s")[0]
    assert len(stored["question"]) == 200 and len(stored["summary"]) == 300
    stored["question"] = "mutated"
    assert memory.recent_turns("s")[0]["question"] != "mutated"
    with pytest.raises(ValueError, match="session_id"):
        memory.append_turn("  ", question="q", device_id="d", risk_level="low", summary="s")


@pytest.mark.unit
def test_ledger_records_whitelisted_fields_only(tmp_path) -> None:
    ledger = LongTermLedger(tmp_path / "memory" / "actions.jsonl")
    record = ledger.record_approved_action(
        request_id="req-1",
        device_id="PUMP-003",
        risk_level="high",
        ticket_id="MNT-req-1",
        decided_by="officer",
    )
    assert set(record) == {
        "recorded_at",
        "action_type",
        "request_id",
        "device_id",
        "risk_level",
        "ticket_id",
        "decided_by",
    }
    lines = (tmp_path / "memory" / "actions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["request_id"] == "req-1"


@pytest.mark.unit
def test_ledger_history_filters_by_device_and_tolerates_damaged_lines(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = LongTermLedger(path)
    for index in range(3):
        ledger.record_approved_action(
            request_id=f"req-{index}",
            device_id="PUMP-003" if index != 1 else "PUMP-999",
            risk_level="medium",
            ticket_id=f"MNT-req-{index}",
            decided_by="officer",
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{damaged json line\n")
    history = ledger.history_for_device("PUMP-003", limit=5)
    assert [item["request_id"] for item in history] == ["req-0", "req-2"]
    recent = ledger.history_for_device("PUMP-003", limit=1)
    assert len(recent) == 1 and recent[0]["request_id"] == "req-2"
    assert ledger.total_records() == 3


@pytest.mark.unit
def test_ledger_rejects_empty_identifiers(tmp_path) -> None:
    ledger = LongTermLedger(tmp_path / "l.jsonl")
    with pytest.raises(ValueError):
        ledger.record_approved_action(
            request_id=" ",
            device_id="PUMP-003",
            risk_level="low",
            ticket_id="t",
            decided_by="o",
        )


@pytest.mark.unit
def test_missing_ledger_file_reads_as_empty(tmp_path) -> None:
    ledger = LongTermLedger(tmp_path / "absent.jsonl")
    assert ledger.history_for_device("PUMP-003") == []
    assert ledger.total_records() == 0
