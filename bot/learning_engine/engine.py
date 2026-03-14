from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Mapping

from bot.core.errors import MaterialValidationError
from bot.core.models import LearningSession, QuizQuestion
from bot.core.ports import (
    ContentGenerator,
    MaterialCache,
    MaterialRepository,
    ProgressRepository,
    StatsRepository,
    UserRepository,
)
from bot.learning_engine.adaptive_learning import AdaptiveLearning
from bot.learning_engine.lesson_flow import LessonFlow
from bot.learning_engine.parser import JsonParseError, MaterialPayloadParser
from bot.learning_engine.progress_tracker import ProgressTracker
from bot.learning_engine.spaced_repetition import SpacedRepetition
from bot.utils.topic_utils import normalize_topic

logger = logging.getLogger(__name__)


def _log_repo_failure(message: str, **context: Any) -> None:
    extra = {key: value for key, value in context.items() if value is not None}
    if extra:
        logger.exception(message, extra=extra)
    else:
        logger.exception(message)


def _log_repo_warning(message: str, **context: Any) -> None:
    extra = {key: value for key, value in context.items() if value is not None}
    if extra:
        logger.warning(message, extra=extra)
    else:
        logger.warning(message)


@dataclass(frozen=True, slots=True)
class MaterialSource:
    material: dict[str, Any]
    source: str


class LearningEngine:
    def __init__(
        self,
        *,
        content_generator: ContentGenerator,
        material_repo: MaterialRepository,
        user_repo: UserRepository,
        progress_repo: ProgressRepository,
        stats_repo: StatsRepository,
        cache: MaterialCache,
        lesson_flow: LessonFlow | None = None,
        adaptive_learning: AdaptiveLearning | None = None,
        spaced_repetition: SpacedRepetition | None = None,
        progress_tracker: ProgressTracker | None = None,
    ) -> None:
        self._content = content_generator
        self._material_repo = material_repo
        self._user_repo = user_repo
        self._progress_repo = progress_repo
        self._stats_repo = stats_repo
        self._cache = cache
        self._lesson_flow = lesson_flow or LessonFlow()
        self._adaptive = adaptive_learning or AdaptiveLearning()
        self._spaced = spaced_repetition or SpacedRepetition()
        self._tracker = progress_tracker or ProgressTracker(progress_repo, stats_repo)

    async def start_lesson(
        self,
        user_id: int,
        topic: str,
        username: str | None = None,
        *,
        explanation_level: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        normalized_topic = normalize_topic(topic)
        await self.ensure_user(user_id, username)
        await self._safe_save_request(user_id, normalized_topic)

        material, source = await self._get_or_generate_material(normalized_topic, explanation_level=explanation_level)
        await self._tracker.start_session(user_id, normalized_topic)
        return material, source

    async def complete_lesson(self, user_id: int) -> bool:
        session = await self._progress_repo.load_session(user_id)
        if not session:
            return False
        completed = await self._tracker.mark_topic_completed(user_id, session.topic)
        if completed:
            await self._tracker.record_activity(user_id, topics_delta=1, last_topic=session.topic, last_stage="done")
        return completed

    async def get_next_task(self, user_id: int) -> dict[str, Any] | None:
        session = await self._progress_repo.load_session(user_id)
        if not session:
            return None
        stage = self._lesson_flow.current_stage(session)
        return {
            "topic": session.topic,
            "stage": stage,
            "card_index": session.card_index,
            "test_index": session.test_index,
            "test_score": session.test_score,
        }

    async def update_progress(self, user_id: int, result: Mapping[str, Any]) -> None:
        session = await self._progress_repo.load_session(user_id)
        if not session:
            return
        stage = str(result.get("stage") or session.stage)
        topic = str(result.get("topic") or session.topic)
        updated = LearningSession(
            user_id=user_id,
            topic=topic,
            stage=stage,
            card_index=int(result.get("card_index", session.card_index)),
            test_index=int(result.get("test_index", session.test_index)),
            test_score=int(result.get("test_score", session.test_score)),
        )
        await self._progress_repo.save_session(updated)

        await self._tracker.record_activity(
            user_id,
            topics_delta=int(result.get("topics_delta", 0)),
            tests_passed_delta=int(result.get("tests_passed_delta", 0)),
            flashcards_delta=int(result.get("flashcards_delta", 0)),
            last_topic=topic,
            last_stage=stage,
        )

    async def ensure_user(self, user_id: int, username: str | None) -> bool:
        try:
            await self._user_repo.ensure_user(user_id, username)
        except Exception:
            _log_repo_failure("Failed to upsert user")
            return False
        return True

    async def get_recent_topics(self, user_id: int, limit: int = 3) -> list[str]:
        try:
            return await self._user_repo.get_last_requests(user_id, limit=limit)
        except Exception:
            _log_repo_failure("Failed to load recent topics")
            return []

    async def get_or_generate_material(
        self,
        *,
        user_id: int,
        username: str | None,
        topic: str,
        explanation_level: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        normalized_topic = normalize_topic(topic)
        await self.ensure_user(user_id, username)
        await self._safe_save_request(user_id, normalized_topic)
        return await self._get_or_generate_material(normalized_topic, explanation_level=explanation_level)

    async def _get_or_generate_material(self, topic: str, *, explanation_level: str | None = None) -> tuple[dict[str, Any], str]:
        cached = await self._load_from_cache(topic)
        if cached is not None:
            return cached.to_dict(), "cache"

        stored = await self._load_from_repo(topic)
        if stored is not None:
            await self._cache.set(topic, stored.to_dict())
            return stored.to_dict(), "db"

        generated = await self._content.generate_material(topic, explanation_level=explanation_level)
        await self._cache.set(topic, generated.to_dict())
        await self._safe_save_material(topic, generated.to_dict())
        return generated.to_dict(), "llm"

    async def _safe_save_request(self, user_id: int, topic: str) -> None:
        try:
            await self._user_repo.save_request(user_id, topic)
        except Exception:
            _log_repo_failure("Failed to save user request", topic=topic)

    async def _load_from_cache(self, topic: str):
        cached_payload = await self._cache.get(topic)
        if cached_payload is None:
            return None
        try:
            return MaterialPayloadParser.parse(cached_payload, topic=topic)
        except (JsonParseError, MaterialValidationError) as exc:
            _log_repo_warning("Cached material is invalid", topic=topic, error=str(exc))
        return None

    async def _load_from_repo(self, topic: str):
        try:
            payload = await self._material_repo.get_material(topic)
        except Exception:
            _log_repo_failure("Failed to get material from db", topic=topic)
            return None

        if payload is None:
            return None

        try:
            return MaterialPayloadParser.parse(payload, topic=topic)
        except (JsonParseError, MaterialValidationError) as exc:
            _log_repo_warning("Stored material is invalid, will regenerate", topic=topic, error=str(exc))
            return None

    async def _safe_save_material(self, topic: str, payload: dict[str, Any]) -> None:
        try:
            await self._material_repo.save_material(topic, payload)
        except Exception:
            _log_repo_failure("Failed to save generated material", topic=topic)

    async def generate_tests(self, topic: str, difficulty: str) -> list[QuizQuestion]:
        return await self._content.generate_tests(topic, difficulty)

    def pick_adaptive_difficulty(self, score: int, total: int) -> str:
        return self._adaptive.pick_difficulty(score, total)

    def next_review_interval(self, current_interval_days: int, success: bool) -> int:
        return self._spaced.next_interval(current_interval_days, success)

    async def save_tests_history(
        self,
        user_id: int,
        topic: str,
        difficulty: str,
        tests: list[QuizQuestion] | list[dict[str, Any]],
        score: int,
        total: int,
    ) -> None:
        try:
            payload = self._tests_to_dicts(tests, context="save_tests_history", topic=topic)
        except MaterialValidationError as exc:
            _log_repo_warning("Failed to serialize tests history", topic=topic, error=str(exc))
            return
        try:
            await self._stats_repo.save_tests_history(user_id, topic, difficulty, payload, score, total)
        except Exception:
            _log_repo_failure("Failed to save tests history", topic=topic)

    def update_cached_tests(self, topic: str, tests: list[QuizQuestion] | list[dict[str, Any]]) -> None:
        normalized_topic = normalize_topic(topic)
        try:
            payload = self._tests_to_dicts(tests, context="update_cached_tests", topic=normalized_topic)
        except MaterialValidationError as exc:
            _log_repo_warning("Failed to update cached tests", topic=normalized_topic, error=str(exc))
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _log_repo_warning("No running loop for cache update", topic=normalized_topic)
            return
        loop.create_task(self._safe_cache_update(normalized_topic, payload))

    async def _safe_cache_update(self, topic: str, tests: list[dict[str, Any]]) -> None:
        try:
            await self._cache.update_tests(topic, tests)
        except Exception:
            _log_repo_failure("Failed to update cached tests", topic=topic)

    @staticmethod
    def _tests_to_dtos(
        tests: list[QuizQuestion] | list[dict[str, Any]],
        *,
        context: str,
        topic: str | None = None,
    ) -> list[QuizQuestion]:
        if not tests:
            raise MaterialValidationError(f"Tests payload is empty in {context}")
        if all(isinstance(item, QuizQuestion) for item in tests):
            return list(tests)
        if all(isinstance(item, Mapping) for item in tests):
            return QuizQuestion.list_from_raw(list(tests), topic=topic)
        raise MaterialValidationError(f"Tests payload has mixed types in {context}")

    def _tests_to_dicts(
        self,
        tests: list[QuizQuestion] | list[dict[str, Any]],
        *,
        context: str,
        topic: str | None = None,
    ) -> list[dict[str, Any]]:
        dtos = self._tests_to_dtos(tests, context=context, topic=topic)
        return [test.to_dict() for test in dtos]

    async def get_user_stats(self, user_id: int) -> dict[str, Any]:
        return await self._tracker.get_user_stats(user_id)

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
        return await self._tracker.record_activity(
            user_id,
            topics_delta=topics_delta,
            tests_passed_delta=tests_passed_delta,
            flashcards_delta=flashcards_delta,
            last_topic=last_topic,
            last_stage=last_stage,
        )

    async def mark_topic_completed(self, user_id: int, topic: str) -> bool:
        try:
            return await self._progress_repo.mark_topic_completed(user_id, topic)
        except Exception:
            _log_repo_failure("Failed to mark topic completed", topic=topic, user_id=user_id)
            return False

    async def load_resume_state(self, user_id: int) -> dict[str, Any] | None:
        try:
            session = await self._progress_repo.load_session(user_id)
        except Exception:
            _log_repo_failure("Failed to load resume state", user_id=user_id)
            return None
        if not session:
            return None
        return {
            "last_topic": session.topic,
            "last_stage": session.stage,
            "card_index": session.card_index,
            "test_index": session.test_index,
            "test_score": session.test_score,
        }

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
        session = LearningSession(
            user_id=user_id,
            topic=normalize_topic(topic),
            stage=stage,
            card_index=card_index,
            test_index=test_index,
            test_score=test_score,
        )
        try:
            await self._progress_repo.save_session(session)
        except Exception:
            _log_repo_failure("Failed to save resume state", topic=topic, user_id=user_id)

    async def reset_progress(self, user_id: int) -> bool:
        try:
            await self._tracker.reset_progress(user_id)
        except Exception:
            _log_repo_failure("Failed to reset user progress", user_id=user_id)
            return False
        return True


__all__ = ["LearningEngine", "MaterialSource"]
