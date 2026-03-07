from __future__ import annotations

import asyncio
import logging
import os
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

DEFAULT_MAX_CONCURRENT = 4

logger = logging.getLogger(__name__)


class LLMLimiter:
    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiters = 0

    async def run(self, func: Callable[[], Awaitable[T]]) -> T:
        self._waiters += 1
        logger.debug("LLM queue waiters=%d", self._waiters)
        try:
            async with self._semaphore:
                return await func()
        finally:
            self._waiters = max(0, self._waiters - 1)


def _load_max_concurrent() -> int:
    raw = os.getenv("LLM_MAX_CONCURRENT", "").strip()
    if not raw:
        return DEFAULT_MAX_CONCURRENT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CONCURRENT
    return value if value > 0 else DEFAULT_MAX_CONCURRENT


_GLOBAL_LIMITER = LLMLimiter(_load_max_concurrent())


def get_global_limiter() -> LLMLimiter:
    return _GLOBAL_LIMITER
