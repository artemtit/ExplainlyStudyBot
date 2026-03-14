from __future__ import annotations

from typing import Any

from bot.learning_engine.engine import LearningEngine


class MaterialService:
    """Thin compatibility wrapper around LearningEngine."""

    def __init__(self, engine: LearningEngine) -> None:
        self._engine = engine

    async def ensure_user(self, user_id: int, username: str | None) -> bool:
        return await self._engine.ensure_user(user_id, username)

    async def get_recent_topics(self, user_id: int, limit: int = 3) -> list[str]:
        return await self._engine.get_recent_topics(user_id, limit=limit)

    async def get_or_generate_material(
        self,
        *,
        user_id: int,
        username: str | None,
        topic: str,
        explanation_level: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        return await self._engine.get_or_generate_material(
            user_id=user_id,
            username=username,
            topic=topic,
            explanation_level=explanation_level,
        )

    async def generate_tests(self, topic: str, difficulty: str):
        return await self._engine.generate_tests(topic, difficulty)

    async def save_tests_history(
        self,
        user_id: int,
        topic: str,
        difficulty: str,
        tests,
        score: int,
        total: int,
    ) -> None:
        await self._engine.save_tests_history(user_id, topic, difficulty, tests, score, total)

    def update_cached_tests(self, topic: str, tests) -> None:
        self._engine.update_cached_tests(topic, tests)

    async def get_user_stats(self, user_id: int) -> dict[str, Any]:
        return await self._engine.get_user_stats(user_id)

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
        return await self._engine.record_activity(
            user_id,
            topics_delta=topics_delta,
            tests_passed_delta=tests_passed_delta,
            flashcards_delta=flashcards_delta,
            last_topic=last_topic,
            last_stage=last_stage,
        )

    async def mark_topic_completed(self, user_id: int, topic: str) -> bool:
        return await self._engine.mark_topic_completed(user_id, topic)

    async def load_resume_state(self, user_id: int) -> dict[str, Any] | None:
        return await self._engine.load_resume_state(user_id)

    async def save_resume_state(
        self,
        user_id: int,
        *,
        topic: str,
        stage: str,
        card_index: int = 0,
        test_index: int = 0,
        test_score: int = 0,
    ) -> None:
        await self._engine.save_resume_state(
            user_id,
            topic=topic,
            stage=stage,
            card_index=card_index,
            test_index=test_index,
            test_score=test_score,
        )

    async def reset_progress(self, user_id: int) -> bool:
        return await self._engine.reset_progress(user_id)


__all__ = ["MaterialService"]
