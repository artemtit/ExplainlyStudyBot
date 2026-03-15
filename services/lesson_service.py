from __future__ import annotations

import logging
from dataclasses import dataclass

from ai.llm_client import LlmClient
from ai.prompts.loader import PromptLoader
from database.repository import Repository


@dataclass
class LessonResult:
    topic: str
    explanation: str
    examples: list[str]


class LessonService:
    def __init__(self, llm_client: LlmClient, *, repository: Repository | None = None) -> None:
        self._llm_client = llm_client
        self._repository = repository
        self._prompts = PromptLoader()
        self._logger = logging.getLogger(__name__)

    async def explain(self, topic: str, *, user_id: int | None = None) -> LessonResult:
        explain_prompt = self._prompts.load("explain_topic", topic=topic)
        explanation = await self._llm_client.complete(explain_prompt)
        examples_prompt = self._prompts.load("examples", topic=topic)
        examples_raw = await self._llm_client.complete(examples_prompt)
        examples = self._parse_examples(examples_raw)
        if self._repository and user_id is not None:
            await self._repository.store_lesson(user_id=user_id, topic=topic, content=explanation)
            self._logger.info("Lesson stored: user_id=%s topic=%s", user_id, topic)
        return LessonResult(topic=topic, explanation=explanation, examples=examples)

    @staticmethod
    def _parse_examples(raw: str) -> list[str]:
        items: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(("-", "*")):
                line = line.lstrip("-* ").strip()
            items.append(line)
        return items[:3]
