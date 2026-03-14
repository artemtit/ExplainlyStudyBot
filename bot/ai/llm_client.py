from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from openai import OpenAI

from bot.analytics.metrics import ensure_metrics_server_started, record_llm_request
from bot.utils.llm_limiter import get_global_limiter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmProvider:
    name: str
    model: str
    client: OpenAI


class OpenAIService:
    def __init__(
        self,
        *,
        groq_api_key: str | None,
        groq_model: str,
        timeout_seconds: int = 180,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._limiter = get_global_limiter()
        ensure_metrics_server_started()

        self._provider: LlmProvider | None = None
        if groq_api_key:
            self._provider = LlmProvider(
                name="groq",
                model=groq_model,
                client=OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1"),
            )
        else:
            logger.warning("Groq disabled: GROQ_API_KEY is not configured")

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        if self._provider is None:
            return None

        started_at = time.monotonic()
        try:
            raw = await self._call_provider(
                self._provider,
                system_prompt,
                user_prompt,
            )
            record_llm_request(duration_seconds=time.monotonic() - started_at, success=raw is not None)
            if raw is None:
                return None
            clean = raw.strip()
            return clean if clean else None
        except Exception as exc:
            record_llm_request(duration_seconds=time.monotonic() - started_at, success=False)
            logger.warning("LLM request failed: %s", repr(exc))
            return None

    async def _call_provider(
        self,
        provider: LlmProvider,
        system_prompt: str,
        user_prompt: str,
    ) -> str | None:
        async def _invoke() -> str | None:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._sync_chat_completion,
                    provider,
                    system_prompt,
                    user_prompt,
                ),
                timeout=self._timeout_seconds,
            )

        return await self._limiter.run(_invoke)

    @staticmethod
    def _sync_chat_completion(
        provider: LlmProvider,
        system_prompt: str,
        user_prompt: str,
    ) -> str | None:
        response = provider.client.chat.completions.create(
            model=provider.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        content = response.choices[0].message.content
        return content if content else None


__all__ = ["OpenAIService", "LlmProvider"]
