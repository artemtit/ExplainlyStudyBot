from __future__ import annotations

import logging

from bot.core.errors import MaterialValidationError
from bot.ai.llm_router import LLMRouter
from bot.ai.prompts import PROMPT_REGISTRY, PromptBuilder, PromptRegistry
from bot.core.models import Material, QuizQuestion
from bot.core.ports import ContentGenerator, LlmClient
from bot.learning_engine.parser import MaterialPayloadParser
from bot.utils.json_parser import JsonParseError, safe_json_parse

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
        fallback: LlmClient | None = None
        supports_provider = getattr(llm_client, "supports_provider", None)
        if callable(supports_provider) and supports_provider("groq"):
            fallback = llm_client
        self._router = router or LLMRouter(primary=llm_client, fallback=fallback)

    async def generate_material(self, topic: str) -> Material:
        prompt = PromptBuilder(self._registry, "lesson_generation").build(topic=topic)
        system_prompt = prompt.system
        user_prompt = prompt.user

        def parse(raw: str) -> Material:
            payload = safe_json_parse(raw)
            return Material.from_payload(payload, topic=topic)

        try:
            return await self._router.generate_json(prompt=prompt, parse_response=parse)
        except (JsonParseError, MaterialValidationError):
            logger.exception(
                "Failed to parse or validate material for topic",
                extra={"topic": topic},
            )
            raise

    async def generate_tests(self, topic: str, difficulty: str) -> list[QuizQuestion]:
        prompt = PromptBuilder(self._registry, "tests_generation").build(topic=topic, difficulty=difficulty)
        system_prompt = prompt.system
        user_prompt = prompt.user

        def parse(raw: str) -> list[QuizQuestion]:
            payload = safe_json_parse(raw)
            return MaterialPayloadParser.parse_tests(payload, topic=topic)

        try:
            return await self._router.generate_json(prompt=prompt, parse_response=parse)
        except (JsonParseError, MaterialValidationError):
            logger.exception(
                "Failed to parse or validate tests for topic",
                extra={"topic": topic},
            )
            raise


__all__ = ["AiContentGenerator"]
