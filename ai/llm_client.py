from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LlmResponse:
    text: str


class LlmClient:
    def __init__(self, *, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("LLM_API_KEY", "")

    async def complete(self, prompt: str) -> str:
        response = await self._call_with_retry(prompt)
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
