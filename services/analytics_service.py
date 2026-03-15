from __future__ import annotations


class AnalyticsService:
    def __init__(self) -> None:
        self._lessons_requested = 0
        self._topics: dict[str, int] = {}

    def record_lesson(self, topic: str) -> None:
        self._lessons_requested += 1
        key = topic.strip()
        if not key:
            return
        self._topics[key] = self._topics.get(key, 0) + 1

    def get_stats(self) -> dict[str, int]:
        return {"lessons_requested": self._lessons_requested, "topics": len(self._topics)}
