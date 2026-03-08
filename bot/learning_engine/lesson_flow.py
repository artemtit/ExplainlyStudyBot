from __future__ import annotations

from dataclasses import dataclass

from bot.core.models import LearningSession


@dataclass(frozen=True, slots=True)
class LessonFlow:
    stages: tuple[str, ...] = ("lesson", "flashcards", "test", "practice")

    def next_stage(self, current_stage: str) -> str | None:
        try:
            idx = self.stages.index(current_stage)
        except ValueError:
            return self.stages[0] if self.stages else None
        next_idx = idx + 1
        if next_idx >= len(self.stages):
            return None
        return self.stages[next_idx]

    def current_stage(self, session: LearningSession) -> str:
        return session.stage or (self.stages[0] if self.stages else "lesson")


__all__ = ["LessonFlow"]
