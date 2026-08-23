"""Short-term in-process session memory.

Sessions are keyed by an explicit session id and keep only a bounded number
of recent turns. Stored text is truncated and is always treated as untrusted
context downstream: it can never change system rules.
"""

from __future__ import annotations

from collections import deque


def _clip(value: str, limit: int) -> str:
    cleaned = " ".join(str(value).split())
    return cleaned[:limit]


class SessionMemory:
    """Bounded per-session recall of recent questions and outcome summaries."""

    def __init__(self, max_turns: int = 5) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least one")
        self._max_turns = max_turns
        self._sessions: dict[str, deque[dict[str, str]]] = {}

    def append_turn(
        self,
        session_id: str,
        *,
        question: str,
        device_id: str,
        risk_level: str,
        summary: str,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        turns = self._sessions.setdefault(session_id, deque(maxlen=self._max_turns))
        turns.append(
            {
                "question": _clip(question, 200),
                "device_id": _clip(device_id, 128),
                "risk_level": _clip(risk_level, 32),
                "summary": _clip(summary, 300),
            }
        )

    def recent_turns(self, session_id: str) -> list[dict[str, str]]:
        """Return copies so callers cannot mutate stored history."""

        return [dict(turn) for turn in self._sessions.get(session_id, ())]

    def forget(self, session_id: str | None = None) -> None:
        """Drop one session or everything; sessions are independently clearable."""

        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)

    def snapshot(self) -> dict[str, int]:
        """Turn counts per session; useful for diagnostics and tests."""

        return {session: len(turns) for session, turns in sorted(self._sessions.items())}
