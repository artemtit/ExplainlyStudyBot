from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class SessionState:
    user_id: int
    last_seen: datetime
    last_topic: str | None = None
    last_mode: str | None = None


class SessionService:
    def __init__(self) -> None:
        self._sessions: dict[int, SessionState] = {}

    def touch(self, user_id: int) -> SessionState:
        state = self._sessions.get(user_id)
        now = datetime.utcnow()
        if state is None:
            state = SessionState(user_id=user_id, last_seen=now)
            self._sessions[user_id] = state
            return state
        state.last_seen = now
        return state

    def set_last_topic(self, *, user_id: int, topic: str, mode: str) -> None:
        state = self.touch(user_id)
        state.last_topic = topic
        state.last_mode = mode

    def get_last_topic(self, user_id: int) -> str | None:
        state = self._sessions.get(user_id)
        return state.last_topic if state else None
