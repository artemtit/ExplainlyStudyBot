from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any

from bot.services.openai_service import OpenAIService
from bot.services.supabase_service import SupabaseService
from bot.utils.json_parser import JsonParseError, safe_json_parse
from bot.utils.strings import PROMPT_SYSTEM, PROMPT_TESTS_SYSTEM, PROMPT_TESTS_USER, PROMPT_USER
from bot.utils.topic_utils import normalize_topic

logger = logging.getLogger(__name__)


class MaterialValidationError(ValueError):
    """Raised when material payload cannot be normalized."""


@dataclass(slots=True)
class CacheEntry:
    created_at: float
    material: dict[str, Any]


class MaterialService:
    def __init__(
        self,
        *,
        llm_service: OpenAIService,
        supabase_service: SupabaseService,
        cache_ttl_seconds: int,
    ) -> None:
        self._llm = llm_service
        self._db = supabase_service
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, CacheEntry] = {}

    async def ensure_user(self, user_id: int, username: str | None) -> None:
        try:
            await self._db.ensure_user(user_id, username)
        except Exception:
            logger.exception("Failed to upsert user")

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
            logger.exception("Failed to save user request")

        cached = self._get_cached(normalized_topic)
        if cached is not None:
            return cached, "cache"

        db_material = await self._load_from_db(normalized_topic)
        if db_material is not None:
            self._cache[normalized_topic] = CacheEntry(time.time(), db_material)
            return db_material, "db"

        generated = await self._generate_from_llm(normalized_topic)
        self._cache[normalized_topic] = CacheEntry(time.time(), generated)
        try:
            await self._db.save_material(normalized_topic, generated)
        except Exception:
            logger.exception("Failed to save generated material")

        return generated, "llm"

    async def _load_from_db(self, topic: str) -> dict[str, Any] | None:
        try:
            payload = await self._db.get_material(topic)
        except Exception:
            logger.exception("Failed to get material from db")
            return None

        if payload is None:
            return None

        try:
            if isinstance(payload, str):
                payload = safe_json_parse(payload)
            normalized = self.normalize_material(payload)
            return normalized
        except Exception:
            logger.exception("Stored material is invalid, will regenerate")
            return None

    async def _generate_from_llm(self, topic: str) -> dict[str, Any]:
        system_prompt = PROMPT_SYSTEM
        user_prompt = PROMPT_USER.format(topic=topic)

        def parse(raw: str) -> dict[str, Any]:
            payload = safe_json_parse(raw)
            return self.normalize_material(payload)

        try:
            return await self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_response=parse,
            )
        except JsonParseError:
            logger.exception("JSON parse failed for generated material")
            raise
        except MaterialValidationError:
            logger.exception("Material schema validation failed")
            raise

    async def generate_tests(self, topic: str, difficulty: str) -> list[dict[str, Any]]:
        system_prompt = PROMPT_TESTS_SYSTEM
        user_prompt = PROMPT_TESTS_USER.format(topic=topic, difficulty=difficulty)

        def parse(raw: str) -> list[dict[str, Any]]:
            payload = safe_json_parse(raw)
            tests_obj = payload.get("tests") if isinstance(payload, dict) else payload
            if not isinstance(tests_obj, list):
                raise MaterialValidationError("Tests payload missing")
            return self._normalize_tests(tests_obj)

        try:
            return await self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parse_response=parse,
            )
        except Exception:
            logger.exception("Failed to generate tests for topic=%s", topic)
            raise

    async def save_tests_history(
        self,
        user_id: int,
        topic: str,
        difficulty: str,
        tests: list[dict[str, Any]],
        score: int,
        total: int,
    ) -> None:
        try:
            await self._db.save_tests_history(user_id, topic, difficulty, tests, score, total)
        except Exception:
            logger.exception("Failed to save tests history")

    def update_cached_tests(self, topic: str, tests: list[dict[str, Any]]) -> None:
        normalized_topic = normalize_topic(topic)
        entry = self._cache.get(normalized_topic)
        if entry is None:
            return
        entry.material["tests"] = tests

    def _get_cached(self, topic: str) -> dict[str, Any] | None:
        entry = self._cache.get(topic)
        if entry is None:
            return None

        if time.time() - entry.created_at > self._cache_ttl:
            self._cache.pop(topic, None)
            return None

        return entry.material

    @staticmethod
    def normalize_material(material: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(material, dict):
            raise MaterialValidationError("Material must be a dict")

        lesson = material.get("lesson")
        if isinstance(lesson, dict):
            for key in ("cards", "tests", "practice"):
                if key in lesson and key not in material:
                    material[key] = lesson[key]

        normalized = {
            "lesson": MaterialService._normalize_lesson(material.get("lesson")),
            "cards": MaterialService._normalize_cards(material.get("cards")),
            "tests": MaterialService._normalize_tests(material.get("tests")),
            "practice": MaterialService._normalize_practice(material.get("practice")),
        }
        return normalized

    @staticmethod
    def _normalize_lesson(raw_lesson: Any) -> dict[str, Any]:
        if isinstance(raw_lesson, str):
            return {
                "title": "\u041c\u0438\u043d\u0438-\u0443\u0440\u043e\u043a",
                "sections": [{"header": "\u041e\u0431\u0437\u043e\u0440", "text": raw_lesson, "key_points": [], "formula": None}],
            }

        if not isinstance(raw_lesson, dict):
            raw_lesson = {}

        title = str(raw_lesson.get("title") or "\u041c\u0438\u043d\u0438-\u0443\u0440\u043e\u043a")
        sections_raw = raw_lesson.get("sections")
        if not isinstance(sections_raw, list):
            sections_raw = []

        sections: list[dict[str, Any]] = []
        for section in sections_raw:
            if not isinstance(section, dict):
                continue
            header = str(section.get("header") or "\u0420\u0430\u0437\u0434\u0435\u043b")
            text = str(section.get("text") or "")
            key_points_raw = section.get("key_points")
            key_points = [str(point) for point in key_points_raw] if isinstance(key_points_raw, list) else []
            formula = section.get("formula")
            if formula is not None:
                formula = str(formula)
            sections.append(
                {
                    "header": header,
                    "text": text,
                    "key_points": key_points,
                    "formula": formula,
                }
            )

        if not sections:
            fallback_text = str(raw_lesson.get("text") or raw_lesson.get("content") or "\u041c\u0438\u043d\u0438-\u0443\u0440\u043e\u043a \u043d\u0435 \u0441\u0433\u0435\u043d\u0435\u0440\u0438\u0440\u043e\u0432\u0430\u043b\u0441\u044f.")
            sections = [{"header": "\u041e\u0431\u0437\u043e\u0440", "text": fallback_text, "key_points": [], "formula": None}]

        return {"title": title, "sections": sections}

    @staticmethod
    def _normalize_cards(raw_cards: Any) -> list[dict[str, str]]:
        cards: list[dict[str, str]] = []
        if isinstance(raw_cards, list):
            for item in raw_cards:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question") or "\u0412\u043e\u043f\u0440\u043e\u0441")
                answer = str(item.get("answer") or "\u041e\u0442\u0432\u0435\u0442")
                cards.append({"question": question, "answer": answer})

        while len(cards) < 5:
            idx = len(cards) + 1
            cards.append({"question": f"\u041a\u0430\u0440\u0442\u043e\u0447\u043a\u0430 {idx}: \u043d\u0435\u0442 \u0432\u043e\u043f\u0440\u043e\u0441\u0430", "answer": "\u041d\u0435\u0442 \u043e\u0442\u0432\u0435\u0442\u0430"})

        return cards[:5]

    @staticmethod
    def _normalize_tests(raw_tests: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        letters = ["A", "B", "C", "D"]

        if isinstance(raw_tests, list):
            for item in raw_tests:
                if not isinstance(item, dict):
                    continue

                question = str(item.get("question") or "\u0422\u0435\u0441\u0442\u043e\u0432\u044b\u0439 \u0432\u043e\u043f\u0440\u043e\u0441")
                options = item.get("options")

                if isinstance(options, dict):
                    options_list = [str(options.get(letter, "")) for letter in letters]
                elif isinstance(options, list):
                    options_list = [str(opt) for opt in options[:4]]
                else:
                    options_list = []

                while len(options_list) < 4:
                    options_list.append(f"\u0412\u0430\u0440\u0438\u0430\u043d\u0442 {len(options_list) + 1}")

                correct_raw = str(item.get("correct") or "").upper().strip()
                if correct_raw not in letters:
                    answer_raw = str(item.get("answer") or "").strip()
                    if answer_raw:
                        found_index = None
                        for idx, option in enumerate(options_list):
                            if answer_raw.lower() == option.lower():
                                found_index = idx
                                break
                        if found_index is not None:
                            correct_raw = letters[found_index]
                    if correct_raw not in letters:
                        correct_raw = random.choice(letters)

                normalized.append(
                    {
                        "question": question,
                        "options": options_list[:4],
                        "correct": correct_raw,
                    }
                )

        while len(normalized) < 5:
            idx = len(normalized) + 1
            normalized.append(
                {
                    "question": f"\u0412\u043e\u043f\u0440\u043e\u0441 {idx}: \u043d\u0435\u0442 \u0432\u043e\u043f\u0440\u043e\u0441\u0430",
                    "options": ["A", "B", "C", "D"],
                    "correct": "A",
                }
            )

        return normalized[:5]

    @staticmethod
    def _normalize_practice(raw_practice: Any) -> dict[str, str]:
        if not isinstance(raw_practice, dict):
            raw_practice = {}

        problem = str(raw_practice.get("problem") or "\u041d\u0435\u0442 \u0437\u0430\u0434\u0430\u0447\u0438.")
        solution = str(raw_practice.get("solution") or "\u041d\u0435\u0442 \u0440\u0435\u0448\u0435\u043d\u0438\u044f.")
        return {"problem": problem, "solution": solution}
