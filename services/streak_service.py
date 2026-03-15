from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class StreakState:
    current: int = 0
    last_date: date | None = None


class StreakService:
    def __init__(self) -> None:
        self._states: dict[int, StreakState] = {}

    def update(self, user_id: int) -> int:
        state = self._states.get(user_id)
        today = date.today()
        if state is None:
            state = StreakState(current=1, last_date=today)
            self._states[user_id] = state
            return state.current
        if state.last_date == today:
            return state.current
        if state.last_date == today - timedelta(days=1):
            state.current += 1
        else:
            state.current = 1
        state.last_date = today
        return state.current

    def get_streak(self, user_id: int) -> int:
        state = self._states.get(user_id)
        return state.current if state else 0
