from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LlmResponse:
    text: str


class LlmClient:
    def __init__(self, *, api_key: str | None = None, cache_size: int = 128) -> None:
        self._api_key = api_key or os.getenv("LLM_API_KEY", "")
        self._cache_size = cache_size
        self._cache: dict[str, str] = {}
        self._logger = logging.getLogger(__name__)

    async def complete(self, prompt: str) -> str:
        if prompt in self._cache:
            return self._cache[prompt]
        response = await self._call_with_retry(prompt)
        self._store_cache(prompt, response.text)
        return response.text

    async def _call_with_retry(self, prompt: str, *, retries: int = 2) -> LlmResponse:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await self._call_api(prompt)
            except Exception as exc:  # pragma: no cover - stubbed client
                last_exc = exc
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError("LLM call failed") from last_exc

    async def _call_api(self, prompt: str) -> LlmResponse:
        # Stub implementation for now. Replace with actual API call.
        if not self._api_key:
            raise RuntimeError("LLM_API_KEY is not set")
        return LlmResponse(text=f"[LLM stub] {prompt}")

    def _store_cache(self, prompt: str, text: str) -> None:
        if self._cache_size <= 0:
            return
        if len(self._cache) >= self._cache_size:
            self._logger.debug("LLM cache cleared")
            self._cache.clear()
        self._cache[prompt] = text
