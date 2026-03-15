from __future__ import annotations

from dataclasses import dataclass

from ai.llm_client import LlmClient


@dataclass
class LessonResult:
    topic: str
    explanation: str


class LessonService:
    def __init__(self, llm_client: LlmClient) -> None:
        self._llm_client = llm_client

    async def explain(self, topic: str) -> LessonResult:
        prompt = f"Объясни тему простыми словами: {topic}"
        explanation = await self._llm_client.complete(prompt)
        return LessonResult(topic=topic, explanation=explanation)
