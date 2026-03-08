from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

try:
    import redis.asyncio as redis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import RedisError
    from redis.exceptions import TimeoutError as RedisTimeoutError
except Exception:  # pragma: no cover - optional dependency
    redis = None
    RedisConnectionError = None
    RedisError = None
    RedisTimeoutError = None

from bot.ai.prompts.base import PromptInstance

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 60 * 60 * 24
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_KEY_PREFIX = "llm:"


@dataclass(slots=True)
class _MemoryEntry:
    value: str
    expires_at: float


class LLMResponseCache:
    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        redis_url: str | None = None,
        key_prefix: str = DEFAULT_KEY_PREFIX,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._memory: dict[str, _MemoryEntry] = {}
        self._redis = None
        self._use_redis = False

        if redis is None:
            logger.info("Redis not available, using in-memory LLM cache")
            return

        url = (redis_url or os.getenv("REDIS_URL", "").strip() or DEFAULT_REDIS_URL).strip()
        try:
            self._redis = redis.Redis.from_url(url, encoding="utf-8", decode_responses=True)
            self._use_redis = True
        except Exception:
            logger.exception("Failed to initialize Redis LLM cache, using in-memory")
            self._redis = None
            self._use_redis = False

    def make_key(self, prompt: PromptInstance, *, model: str | None) -> str:
        payload = "|".join(
            [
                prompt.name,
                prompt.version,
                prompt.variant,
                prompt.user,
                model or "",
            ]
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{self._key_prefix}{digest}"

    async def get(self, prompt: PromptInstance, *, model: str | None) -> str | None:
        key = self.make_key(prompt, model=model)

        if self._use_redis and self._redis is not None:
            try:
                cached = await self._redis.get(key)
                if cached is not None:
                    logger.info("LLM cache hit", extra={"key": key, "prompt": prompt.name})
                    return cached
                logger.info("LLM cache miss", extra={"key": key, "prompt": prompt.name})
                return None
            except (RedisTimeoutError, RedisConnectionError, RedisError, OSError):
                logger.exception("Redis cache failure, falling back to in-memory")
                self._use_redis = False

        entry = self._memory.get(key)
        now = time.time()
        if entry and entry.expires_at > now:
            logger.info("LLM cache hit", extra={"key": key, "prompt": prompt.name})
            return entry.value
        if entry:
            self._memory.pop(key, None)
        logger.info("LLM cache miss", extra={"key": key, "prompt": prompt.name})
        return None

    async def set(self, prompt: PromptInstance, *, model: str | None, response: str) -> None:
        key = self.make_key(prompt, model=model)

        if self._use_redis and self._redis is not None:
            try:
                await self._redis.set(key, response, ex=self._ttl_seconds)
                logger.info("LLM cache store", extra={"key": key, "prompt": prompt.name})
                return
            except (RedisTimeoutError, RedisConnectionError, RedisError, OSError):
                logger.exception("Redis cache failure, falling back to in-memory")
                self._use_redis = False

        self._memory[key] = _MemoryEntry(value=response, expires_at=time.time() + self._ttl_seconds)
        logger.info("LLM cache store", extra={"key": key, "prompt": prompt.name})


__all__ = ["LLMResponseCache", "DEFAULT_TTL_SECONDS"]
