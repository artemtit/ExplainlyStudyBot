from __future__ import annotations

import logging
from typing import Callable, TypeVar

from bot.ai.prompts.base import PromptInstance
from bot.core.ports import LlmClient

logger = logging.getLogger(__name__)
T = TypeVar("T")


class LLMRouter:
    """Single-provider LLM router for Groq."""

    def __init__(self, *, client: LlmClient) -> None:
        self._client = client

    async def generate_json(
        self,
        *,
        prompt: PromptInstance,
        parse_response: Callable[[str], T],
    ) -> T | None:
        raw = await self._client.generate_json(
            system_prompt=prompt.system,
            user_prompt=prompt.user,
        )
        if not raw:
            return None

        try:
            return parse_response(raw)
        except Exception as exc:
            logger.warning(
                "LLM JSON parse failed: prompt=%s error=%s",
                prompt.name,
                repr(exc),
            )
            return None


__all__ = ["LLMRouter"]
