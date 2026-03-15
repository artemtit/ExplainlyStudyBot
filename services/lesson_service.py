from __future__ import annotations

from dataclasses import dataclass

from ai.llm_client import LlmClient
from database.repository import Repository


@dataclass
class LessonResult:
    topic: str
    explanation: str


class LessonService:
    def __init__(self, llm_client: LlmClient, *, repository: Repository | None = None) -> None:
        self._llm_client = llm_client
        self._repository = repository

    async def explain(self, topic: str, *, user_id: int | None = None) -> LessonResult:
        prompt = f"Объясни тему простыми словами: {topic}"
        explanation = await self._llm_client.complete(prompt)
        if self._repository and user_id is not None:
            await self._repository.store_lesson(user_id=user_id, topic=topic, content=explanation)
        return LessonResult(topic=topic, explanation=explanation)
