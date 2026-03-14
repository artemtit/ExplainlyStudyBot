from __future__ import annotations

import logging

from bot.ai.llm_router import LLMRouter
from bot.ai.prompts import PROMPT_REGISTRY, PromptBuilder, PromptRegistry
from bot.core.models import Flashcard, Lesson, LessonSection, Material, PracticeProblem, QuizQuestion
from bot.core.ports import ContentGenerator, LlmClient
from bot.learning_engine.parser import MaterialPayloadParser
from bot.utils.formula_parser import normalize_lesson_payload
from bot.utils.json_parser import safe_json_parse

logger = logging.getLogger(__name__)


class AiContentGenerator(ContentGenerator):
    def __init__(
        self,
        llm_client: LlmClient,
        *,
        prompt_registry: PromptRegistry | None = None,
        router: LLMRouter | None = None,
    ) -> None:
        self._llm = llm_client
        self._registry = prompt_registry or PROMPT_REGISTRY
        self._router = router or LLMRouter(client=llm_client)

    async def generate_material(self, topic: str, explanation_level: str | None = None) -> Material:
        difficulty = explanation_level if explanation_level in {"simple", "normal", "hard"} else "normal"
        prompt = PromptBuilder(self._registry, "lesson_generation").build(topic=topic, difficulty=difficulty)

        def parse(raw: str) -> Material:
            payload = safe_json_parse(raw)
            normalize_lesson_payload(payload)
            return Material.from_payload(payload, topic=topic)

        result: Material | None = None
        try:
            result = await self._router.generate_json(prompt=prompt, parse_response=parse)
        except Exception:
            logger.exception("Unexpected LLM failure for topic", extra={"topic": topic})
            result = None

        if result is not None:
            return result

        logger.error(
            "LLM generation failed completely. Using fallback lesson for topic: %s",
            topic,
        )
        return self._fallback_material(topic)

    async def generate_tests(self, topic: str, difficulty: str) -> list[QuizQuestion]:
        prompt = PromptBuilder(self._registry, "tests_generation").build(topic=topic, difficulty=difficulty)

        def parse(raw: str) -> list[QuizQuestion]:
            payload = safe_json_parse(raw)
            return MaterialPayloadParser.parse_tests(payload, topic=topic)

        result: list[QuizQuestion] | None = None
        try:
            result = await self._router.generate_json(prompt=prompt, parse_response=parse)
        except Exception:
            logger.exception(
                "Unexpected LLM failure for tests",
                extra={"topic": topic, "difficulty": difficulty},
            )
            result = None

        if result is not None:
            return result

        logger.error(
            "LLM tests generation failed completely. Using fallback tests for topic: %s",
            topic,
        )
        return self._fallback_tests(topic)

    @staticmethod
    def _fallback_material(topic: str) -> Material:
        fallback_text = (
            f"Тема '{topic}' относится к школьной программе. "
            "Попробуйте сформулировать вопрос иначе или запросить более конкретное объяснение."
        )
        lesson = Lesson(
            title=topic,
            sections=[
                LessonSection(
                    header="Краткое объяснение",
                    text=fallback_text,
                    key_points=[],
                    formula=None,
                )
            ],
        )
        cards = [
            Flashcard(
                question=f"Что такое {topic}?",
                answer="Краткое определение недоступно из-за ошибки.",
            )
            for _ in range(5)
        ]
        tests = AiContentGenerator._fallback_tests(topic)
        practice = PracticeProblem(
            problem=f"Сформулируйте один уточняющий вопрос по теме '{topic}'.",
            solution="Например: уточните определение, формулу или пример применения.",
        )
        return Material(lesson=lesson, cards=cards, tests=tests, practice=practice)

    @staticmethod
    def _fallback_tests(topic: str) -> list[QuizQuestion]:
        return [
            QuizQuestion(
                question=f"Вопрос по теме '{topic}' недоступен. Выберите вариант A.",
                options=["A", "B", "C", "D"],
                correct="A",
                explanation="Генерация временно недоступна.",
            )
            for _ in range(5)
        ]


__all__ = ["AiContentGenerator"]
