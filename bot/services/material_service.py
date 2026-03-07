from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Mapping

from bot.services.openai_service import OpenAIService
from bot.services.redis_cache import MaterialCacheProtocol, RedisMaterialCache
from bot.services.supabase_service import SupabaseService
from bot.utils.json_parser import JsonParseError, safe_json_parse
from bot.utils.strings import PROMPT_SYSTEM, PROMPT_TESTS_SYSTEM, PROMPT_TESTS_USER, PROMPT_USER
from bot.utils.topic_utils import normalize_topic

logger = logging.getLogger(__name__)

TEST_LETTERS = ("A", "B", "C", "D")
MAX_TESTS = 5


def _log_validation_warning(message: str, **context: Any) -> None:
    extra = {key: value for key, value in context.items() if value is not None}
    if extra:
        logger.warning(message, extra=extra)
    else:
        logger.warning(message)


class MaterialValidationError(ValueError):
    """Raised when material payload cannot be validated."""


@dataclass(frozen=True, slots=True)
class TestQuestion:
    question: str
    options: list[str]
    correct: str
    explanation: str

    @classmethod
    def from_raw(
        cls,
        raw: Mapping[str, Any],
        *,
        topic: str | None = None,
        index: int | None = None,
    ) -> "TestQuestion":
        question = str(raw.get("question") or "").strip()
        if not question:
            raise MaterialValidationError("Test question is missing")

        options = cls._coerce_options(raw.get("options"))
        if len(options) != 4 or any(not option for option in options):
            raise MaterialValidationError("Malformed test options")

        correct_raw = str(raw.get("correct") or "").strip().upper()
        if correct_raw not in TEST_LETTERS:
            answer_raw = str(raw.get("answer") or "").strip()
            if answer_raw:
                for idx, option in enumerate(options):
                    if answer_raw.lower() == option.lower():
                        correct_raw = TEST_LETTERS[idx]
                        break
        if correct_raw not in TEST_LETTERS:
            raise MaterialValidationError("Invalid test correct answer")

        explanation = str(raw.get("explanation") or "").strip()
        if not explanation:
            explanation = "\u2014"
            _log_validation_warning(
                "Test explanation missing, using fallback",
                topic=topic,
                test_index=index,
            )

        return cls(question=question, options=options, correct=correct_raw, explanation=explanation)

    @classmethod
    def list_from_raw(cls, raw: Any, *, topic: str | None = None) -> list["TestQuestion"]:
        if not isinstance(raw, list):
            raise MaterialValidationError("Tests payload must be a list")
        if not raw:
            raise MaterialValidationError("Tests payload is empty")

        tests: list[TestQuestion] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise MaterialValidationError(f"Test item at index {idx} must be an object")
            tests.append(cls.from_raw(item, topic=topic, index=idx))

        return tests[:MAX_TESTS]

    @staticmethod
    def _coerce_options(raw: Any) -> list[str]:
        if isinstance(raw, Mapping):
            options = [str(raw.get(letter, "")).strip() for letter in TEST_LETTERS]
        elif isinstance(raw, list):
            options = [str(option).strip() for option in raw]
        else:
            raise MaterialValidationError("Test options must be a list or object")

        if len(options) < 4:
            return options
        return options[:4]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "options": list(self.options),
            "correct": self.correct,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class LessonSection:
    header: str
    text: str
    key_points: list[str]
    formula: str | None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "LessonSection":
        header = str(raw.get("header") or "\u0420\u0430\u0437\u0434\u0435\u043b").strip()
        text = str(raw.get("text") or "").strip()
        if not text:
            raise MaterialValidationError("Lesson section text is missing")

        key_points_raw = raw.get("key_points")
        if key_points_raw is None:
            key_points: list[str] = []
        elif isinstance(key_points_raw, list):
            key_points = [str(point).strip() for point in key_points_raw if str(point).strip()]
        else:
            raise MaterialValidationError("Lesson key_points must be a list")

        formula = raw.get("formula")
        if formula is not None:
            formula = str(formula)

        return cls(header=header, text=text, key_points=key_points, formula=formula)

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": self.header,
            "text": self.text,
            "key_points": list(self.key_points),
            "formula": self.formula,
        }


@dataclass(frozen=True, slots=True)
class Lesson:
    title: str
    sections: list[LessonSection]

    @classmethod
    def from_raw(cls, raw: Any) -> "Lesson":
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                raise MaterialValidationError("Lesson text is empty")
            sections = [
                LessonSection(
                    header="\u041e\u0431\u0437\u043e\u0440",
                    text=text,
                    key_points=[],
                    formula=None,
                )
            ]
            return cls(title="\u041c\u0438\u043d\u0438-\u0443\u0440\u043e\u043a", sections=sections)

        if not isinstance(raw, Mapping):
            raise MaterialValidationError("Lesson must be an object or string")

        title = str(raw.get("title") or "\u041c\u0438\u043d\u0438-\u0443\u0440\u043e\u043a").strip()
        sections_raw = raw.get("sections")
        sections: list[LessonSection] = []

        if isinstance(sections_raw, list):
            for idx, section in enumerate(sections_raw):
                if not isinstance(section, Mapping):
                    raise MaterialValidationError(f"Lesson section at index {idx} must be an object")
                sections.append(LessonSection.from_raw(section))

        if not sections:
            fallback_text = str(raw.get("text") or raw.get("content") or "").strip()
            if not fallback_text:
                raise MaterialValidationError("Lesson sections are missing")
            sections = [
                LessonSection(
                    header="\u041e\u0431\u0437\u043e\u0440",
                    text=fallback_text,
                    key_points=[],
                    formula=None,
                )
            ]

        return cls(title=title, sections=sections)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "sections": [section.to_dict() for section in self.sections]}


@dataclass(frozen=True, slots=True)
class Card:
    question: str
    answer: str

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "Card":
        question = str(raw.get("question") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        if not question or not answer:
            raise MaterialValidationError("Card question/answer is missing")
        return cls(question=question, answer=answer)

    @classmethod
    def list_from_raw(cls, raw: Any) -> list["Card"]:
        if not isinstance(raw, list):
            raise MaterialValidationError("Cards payload must be a list")
        if not raw:
            raise MaterialValidationError("Cards payload is empty")

        cards: list[Card] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise MaterialValidationError(f"Card item at index {idx} must be an object")
            cards.append(cls.from_raw(item))

        return cards[:5]

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer}


@dataclass(frozen=True, slots=True)
class Practice:
    problem: str
    solution: str

    @classmethod
    def from_raw(cls, raw: Any) -> "Practice":
        if not isinstance(raw, Mapping):
            raise MaterialValidationError("Practice must be an object")
        problem = str(raw.get("problem") or "").strip()
        solution = str(raw.get("solution") or "").strip()
        if not problem or not solution:
            raise MaterialValidationError("Practice problem/solution is missing")
        return cls(problem=problem, solution=solution)

    def to_dict(self) -> dict[str, Any]:
        return {"problem": self.problem, "solution": self.solution}


@dataclass(frozen=True, slots=True)
class Material:
    lesson: Lesson
    cards: list[Card]
    tests: list[TestQuestion]
    practice: Practice

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, topic: str | None = None) -> "Material":
        if not isinstance(payload, Mapping):
            raise MaterialValidationError("Material payload must be an object")

        normalized = dict(payload)
        lesson_raw = normalized.get("lesson")
        if isinstance(lesson_raw, Mapping):
            for key in ("cards", "tests", "practice"):
                if key in lesson_raw and key not in normalized:
                    normalized[key] = lesson_raw[key]

        lesson = Lesson.from_raw(normalized.get("lesson"))
        cards = Card.list_from_raw(normalized.get("cards"))
        tests = TestQuestion.list_from_raw(normalized.get("tests"), topic=topic)
        practice = Practice.from_raw(normalized.get("practice"))

        return cls(lesson=lesson, cards=cards, tests=tests, practice=practice)

    def with_tests(self, tests: list[TestQuestion]) -> "Material":
        return Material(lesson=self.lesson, cards=self.cards, tests=tests, practice=self.practice)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson": self.lesson.to_dict(),
            "cards": [card.to_dict() for card in self.cards],
            "tests": [test.to_dict() for test in self.tests],
            "practice": self.practice.to_dict(),
        }


class MaterialPayloadParser:
    @staticmethod
    def parse(payload: dict[str, Any] | str, *, topic: str | None = None) -> Material:
        data = payload
        if isinstance(payload, str):
            data = safe_json_parse(payload)
        if not isinstance(data, Mapping):
            raise MaterialValidationError("Material payload must be an object")
        return Material.from_payload(data, topic=topic)

    @staticmethod
    def parse_tests(payload: Any, *, topic: str | None = None) -> list[TestQuestion]:
        if isinstance(payload, Mapping):
            payload = payload.get("tests")
        return TestQuestion.list_from_raw(payload, topic=topic)


class MaterialGenerator:
    def __init__(self, llm_service: OpenAIService) -> None:
        self._llm = llm_service

    async def generate_material(self, topic: str) -> Material:
        system_prompt = PROMPT_SYSTEM
        user_prompt = PROMPT_USER.format(topic=topic)

        def parse(raw: str) -> Material:
            payload = safe_json_parse(raw)
            return Material.from_payload(payload, topic=topic)

        try:
            return await self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_response=parse,
            )
        except (JsonParseError, MaterialValidationError):
            logger.exception(
                "Failed to parse or validate material for topic",
                extra={"topic": topic},
            )
            raise

    async def generate_tests(self, topic: str, difficulty: str) -> list[TestQuestion]:
        system_prompt = PROMPT_TESTS_SYSTEM
        user_prompt = PROMPT_TESTS_USER.format(topic=topic, difficulty=difficulty)

        def parse(raw: str) -> list[TestQuestion]:
            payload = safe_json_parse(raw)
            return MaterialPayloadParser.parse_tests(payload, topic=topic)

        try:
            return await self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_response=parse,
            )
        except (JsonParseError, MaterialValidationError):
            logger.exception(
                "Failed to parse or validate tests for topic",
                extra={"topic": topic},
            )
            raise


class MaterialService:
    def __init__(
        self,
        *,
        llm_service: OpenAIService,
        supabase_service: SupabaseService,
        cache_ttl_seconds: int,
        cache: MaterialCacheProtocol | None = None,
    ) -> None:
        self._llm = MaterialGenerator(llm_service)
        self._db = supabase_service
        self._cache = cache or RedisMaterialCache.from_env(ttl_seconds=cache_ttl_seconds)

    async def ensure_user(self, user_id: int, username: str | None) -> bool:
        try:
            await self._db.ensure_user(user_id, username)
        except Exception:
            logger.exception("Failed to upsert user")
            return False
        return True

    async def get_recent_topics(self, user_id: int, limit: int = 3) -> list[str]:
        try:
            return await self._db.get_last_requests(user_id, limit=limit)
        except Exception:
            logger.exception("Failed to load recent topics")
            return []

    async def get_or_generate_material(
        self,
        *,
        user_id: int,
        username: str | None,
        topic: str,
    ) -> tuple[dict[str, Any], str]:
        """
        Returns tuple: (material, source) where source in {"cache", "db", "llm"}.
        """
        normalized_topic = normalize_topic(topic)

        await self.ensure_user(user_id, username)
        try:
            await self._db.save_request(user_id, normalized_topic)
        except Exception:
            logger.exception("Failed to save user request", extra={"topic": normalized_topic})

        cached_payload = await self._cache.get(normalized_topic)
        if cached_payload is not None:
            try:
                cached = MaterialPayloadParser.parse(cached_payload, topic=normalized_topic)
            except (JsonParseError, MaterialValidationError) as exc:
                logger.warning("Cached material is invalid: %s", exc, extra={"topic": normalized_topic})
            else:
                return cached.to_dict(), "cache"

        db_material = await self._load_from_db(normalized_topic)
        if db_material is not None:
            await self._cache.set(normalized_topic, db_material.to_dict())
            return db_material.to_dict(), "db"

        generated = await self._llm.generate_material(normalized_topic)
        await self._cache.set(normalized_topic, generated.to_dict())
        try:
            await self._db.save_material(normalized_topic, generated.to_dict())
        except Exception:
            logger.exception("Failed to save generated material", extra={"topic": normalized_topic})

        return generated.to_dict(), "llm"

    async def _load_from_db(self, topic: str) -> Material | None:
        try:
            payload = await self._db.get_material(topic)
        except Exception:
            logger.exception("Failed to get material from db", extra={"topic": topic})
            return None

        if payload is None:
            return None

        try:
            return MaterialPayloadParser.parse(payload, topic=topic)
        except (JsonParseError, MaterialValidationError) as exc:
            logger.warning(
                "Stored material is invalid, will regenerate: %s",
                exc,
                extra={"topic": topic},
            )
            return None

    async def generate_tests(self, topic: str, difficulty: str) -> list[TestQuestion]:
        return await self._llm.generate_tests(topic, difficulty)

    async def save_tests_history(
        self,
        user_id: int,
        topic: str,
        difficulty: str,
        tests: list[TestQuestion] | list[dict[str, Any]],
        score: int,
        total: int,
    ) -> None:
        try:
            payload = self._tests_to_dicts(tests, context="save_tests_history", topic=topic)
        except MaterialValidationError as exc:
            logger.warning("Failed to serialize tests history: %s", exc, extra={"topic": topic})
            return
        try:
            await self._db.save_tests_history(user_id, topic, difficulty, payload, score, total)
        except Exception:
            logger.exception("Failed to save tests history", extra={"topic": topic})

    def update_cached_tests(self, topic: str, tests: list[TestQuestion] | list[dict[str, Any]]) -> None:
        normalized_topic = normalize_topic(topic)
        try:
            payload = self._tests_to_dicts(tests, context="update_cached_tests", topic=normalized_topic)
        except MaterialValidationError as exc:
            logger.warning("Failed to update cached tests: %s", exc, extra={"topic": normalized_topic})
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No running loop for cache update", extra={"topic": normalized_topic})
            return
        loop.create_task(self._safe_cache_update(normalized_topic, payload))

    async def _safe_cache_update(self, topic: str, tests: list[dict[str, Any]]) -> None:
        try:
            await self._cache.update_tests(topic, tests)
        except Exception:
            logger.exception("Failed to update cached tests", extra={"topic": topic})

    @staticmethod
    def _tests_to_dtos(
        tests: list[TestQuestion] | list[dict[str, Any]],
        *,
        context: str,
        topic: str | None = None,
    ) -> list[TestQuestion]:
        if not tests:
            raise MaterialValidationError(f"Tests payload is empty in {context}")
        if all(isinstance(item, TestQuestion) for item in tests):
            return list(tests)
        if all(isinstance(item, Mapping) for item in tests):
            return TestQuestion.list_from_raw(list(tests), topic=topic)
        raise MaterialValidationError(f"Tests payload has mixed types in {context}")

    def _tests_to_dicts(
        self,
        tests: list[TestQuestion] | list[dict[str, Any]],
        *,
        context: str,
        topic: str | None = None,
    ) -> list[dict[str, Any]]:
        dtos = self._tests_to_dtos(tests, context=context, topic=topic)
        return [test.to_dict() for test in dtos]

    async def get_user_stats(self, user_id: int) -> dict[str, Any]:
        try:
            stats = await self._db.get_user_stats(user_id)
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
            await self._db.upsert_user_stats(updated)
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

    async def mark_topic_completed(self, user_id: int, topic: str) -> bool:
        try:
            return await self._db.mark_topic_completed(user_id, topic)
        except Exception:
            logger.exception("Failed to mark topic completed", extra={"topic": topic, "user_id": user_id})
            return False

    async def load_resume_state(self, user_id: int) -> dict[str, Any] | None:
        try:
            return await self._db.load_resume_state(user_id)
        except Exception:
            logger.exception("Failed to load resume state", extra={"user_id": user_id})
            return None

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
        try:
            await self._db.save_resume_state(
                user_id=user_id,
                topic=topic,
                stage=stage,
                card_index=card_index,
                test_index=test_index,
                test_score=test_score,
            )
        except Exception:
            logger.exception("Failed to save resume state", extra={"topic": topic, "user_id": user_id})

    async def reset_progress(self, user_id: int) -> bool:
        try:
            await self._db.reset_user_progress(user_id)
        except Exception:
            logger.exception("Failed to reset user progress", extra={"user_id": user_id})
            return False
        return True

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
        if last_active:
            try:
                last_date = date.fromisoformat(last_active)
            except ValueError:
                last_date = None
        else:
            last_date = None

        if last_date == today:
            return current
        if last_date == today - timedelta(days=1):
            return max(current + 1, 1)
        return 1
