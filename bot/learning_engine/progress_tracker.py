from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from bot.core.models import LearningSession
from bot.core.ports import ProgressRepository, StatsRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProgressTracker:
    progress_repo: ProgressRepository
    stats_repo: StatsRepository

    async def load_session(self, user_id: int) -> LearningSession | None:
        return await self.progress_repo.load_session(user_id)

    async def start_session(self, user_id: int, topic: str) -> LearningSession:
        session = LearningSession(user_id=user_id, topic=topic, stage="lesson")
        await self.progress_repo.save_session(session)
        return session

    async def save_session(self, session: LearningSession) -> None:
        await self.progress_repo.save_session(session)

    async def mark_topic_completed(self, user_id: int, topic: str) -> bool:
        return await self.progress_repo.mark_topic_completed(user_id, topic)

    async def reset_progress(self, user_id: int) -> None:
        await self.progress_repo.reset_progress(user_id)
        await self.stats_repo.reset_user_stats(user_id)

    async def get_user_stats(self, user_id: int) -> dict[str, Any]:
        try:
            stats = await self.stats_repo.get_user_stats(user_id)
        except Exception:
            logger.exception("Failed to load user stats", extra={"user_id": user_id})
            return self._default_stats(user_id)
        return self._normalize_stats(stats, user_id=user_id)

    async def record_activity(
        self,
        user_id: int,
        *,
        topics_delta: int = 0,
        tests_passed_delta: int = 0,
        flashcards_delta: int = 0,
        last_topic: str | None = None,
        last_stage: str | None = None,
    ) -> dict[str, Any]:
        stats = await self.get_user_stats(user_id)
        today = date.today()
        streak = self._compute_streak(stats.get("last_active_date"), int(stats.get("daily_streak", 0)), today)

        updated = {
            "user_id": user_id,
            "topics_learned": max(0, int(stats.get("topics_learned", 0)) + topics_delta),
            "tests_passed": max(0, int(stats.get("tests_passed", 0)) + tests_passed_delta),
            "flashcards_reviewed": max(0, int(stats.get("flashcards_reviewed", 0)) + flashcards_delta),
            "daily_streak": streak,
            "last_active_date": today.isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
        }

        if last_topic is not None:
            updated["last_topic"] = last_topic
        if last_stage is not None:
            updated["last_stage"] = last_stage

        try:
            await self.stats_repo.upsert_user_stats(updated)
        except Exception:
            logger.exception("Failed to update user stats", extra={"user_id": user_id})
        else:
            logger.info(
                "Stats updated",
                extra={
                    "user_id": user_id,
                    "tests_delta": tests_passed_delta,
                    "topics_delta": topics_delta,
                },
            )
        return {**stats, **updated}

    @staticmethod
    def _default_stats(user_id: int) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "topics_learned": 0,
            "tests_passed": 0,
            "flashcards_reviewed": 0,
            "daily_streak": 0,
            "last_active_date": None,
            "last_topic": None,
            "last_stage": None,
            "card_index": 0,
            "test_index": 0,
            "test_score": 0,
        }

    @classmethod
    def _normalize_stats(cls, stats: dict[str, Any] | None, *, user_id: int) -> dict[str, Any]:
        if not isinstance(stats, dict):
            return cls._default_stats(user_id)
        normalized = cls._default_stats(user_id)
        normalized.update({key: value for key, value in stats.items() if value is not None})
        return normalized

    @staticmethod
    def _compute_streak(last_active: str | None, current: int, today: date) -> int:
        last_date = None
        if last_active:
            try:
                last_date = date.fromisoformat(last_active)
            except ValueError:
                last_date = None

        if last_date == today:
            return current
        if last_date == today - timedelta(days=1):
            return max(current + 1, 1)
        return 1


__all__ = ["ProgressTracker"]
