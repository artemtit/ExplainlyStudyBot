from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Any, Awaitable, Callable, Protocol, TypeVar

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from bot.utils.metrics import (
    ensure_metrics_server_started,
    inc_cache_hit,
    inc_cache_miss,
    inc_redis_error,
    log_json,
)

logger = logging.getLogger(__name__)

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
REDIS_TIMEOUT_SECONDS = 3
MAX_CONNECTIONS = 20

T = TypeVar("T")


class MaterialCacheProtocol(Protocol):
    async def get(self, topic: str) -> dict[str, Any] | None:
        ...

    async def set(self, topic: str, material: dict[str, Any]) -> None:
        ...

    async def update_tests(self, topic: str, tests: list[dict[str, Any]]) -> None:
        ...


class RedisMaterialCache:
    def __init__(self, *, url: str, ttl_seconds: int, key_prefix: str = "material:") -> None:
        self._url = url
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._redis = self._create_client()
        self._reconnect_lock = asyncio.Lock()
        ensure_metrics_server_started()

    @classmethod
    def from_env(cls, *, ttl_seconds: int, key_prefix: str = "material:") -> "RedisMaterialCache":
        url = os.getenv("REDIS_URL", "").strip() or DEFAULT_REDIS_URL
        return cls(url=url, ttl_seconds=ttl_seconds, key_prefix=key_prefix)

    async def close(self) -> None:
        try:
            await self._redis.close()
        except Exception:
            logger.exception("Failed to close Redis connection")

    def _create_client(self) -> redis.Redis:
        pool = redis.ConnectionPool.from_url(
            self._url,
            max_connections=MAX_CONNECTIONS,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=REDIS_TIMEOUT_SECONDS,
            socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
        )
        return redis.Redis(connection_pool=pool)

    async def _reconnect(self) -> None:
        try:
            await self._redis.close()
        except Exception:
            logger.exception("Failed to close Redis connection during reconnect")
        self._redis = self._create_client()

    async def _reconnect_once(self) -> None:
        async with self._reconnect_lock:
            try:
                await self._redis.ping()
                return
            except Exception:
                await self._reconnect()

    def _key(self, topic: str) -> str:
        digest = hashlib.sha256(topic.encode("utf-8")).hexdigest()
        return f"{self._key_prefix}{digest}"

    def _make_runner(self, *, topic: str) -> Callable[[str, Callable[[], Awaitable[T]], str | None], Awaitable[T | None]]:
        reconnected = False

        async def run(op: str, func: Callable[[], Awaitable[T]], stage: str | None = None) -> T | None:
            nonlocal reconnected
            try:
                return await func()
            except RedisTimeoutError as exc:
                inc_redis_error()
                log_json(logger, logging.WARNING, "redis_timeout", op=op, topic=topic, stage=stage, error=str(exc))
                return None
            except RedisConnectionError as exc:
                inc_redis_error()
                log_json(
                    logger,
                    logging.ERROR,
                    "redis_connection_error",
                    op=op,
                    topic=topic,
                    stage=stage,
                    error=str(exc),
                )
                if reconnected:
                    return None
                reconnected = True
                await self._reconnect_once()
                try:
                    return await func()
                except RedisTimeoutError as retry_exc:
                    inc_redis_error()
                    log_json(
                        logger,
                        logging.WARNING,
                        "redis_timeout",
                        op=op,
                        topic=topic,
                        stage=stage,
                        error=str(retry_exc),
                    )
                    return None
                except RedisConnectionError as retry_exc:
                    inc_redis_error()
                    log_json(
                        logger,
                        logging.ERROR,
                        "redis_connection_error",
                        op=op,
                        topic=topic,
                        stage=stage,
                        error=str(retry_exc),
                    )
                    return None
                except (RedisError, OSError) as retry_exc:
                    inc_redis_error()
                    log_json(
                        logger,
                        logging.ERROR,
                        "redis_error",
                        op=op,
                        topic=topic,
                        stage=stage,
                        error=str(retry_exc),
                    )
                    return None
            except (RedisError, OSError) as exc:
                inc_redis_error()
                log_json(
                    logger,
                    logging.ERROR,
                    "redis_error",
                    op=op,
                    topic=topic,
                    stage=stage,
                    error=str(exc),
                )
                return None

        return run

    async def get(self, topic: str) -> dict[str, Any] | None:
        key = self._key(topic)
        run = self._make_runner(topic=topic)

        raw = await run("get", lambda: self._redis.get(key))
        if raw is None:
            inc_cache_miss()
            return None
        try:
            data = json.loads(raw)
        except Exception as exc:
            inc_redis_error()
            inc_cache_miss()
            log_json(logger, logging.ERROR, "redis_decode_error", topic=topic, error=str(exc))
            return None
        if not isinstance(data, dict):
            inc_redis_error()
            inc_cache_miss()
            log_json(logger, logging.ERROR, "redis_payload_invalid", topic=topic)
            return None
        inc_cache_hit()
        return data

    async def set(self, topic: str, material: dict[str, Any]) -> None:
        key = self._key(topic)
        run = self._make_runner(topic=topic)

        try:
            payload = json.dumps(material, ensure_ascii=False)
        except Exception as exc:
            inc_redis_error()
            log_json(logger, logging.ERROR, "redis_encode_error", topic=topic, error=str(exc))
            return

        await run("set", lambda: self._redis.set(key, payload, ex=self._ttl_seconds))

    async def update_tests(self, topic: str, tests: list[dict[str, Any]]) -> None:
        key = self._key(topic)
        run = self._make_runner(topic=topic)

        raw = await run("get", lambda: self._redis.get(key), stage="update_tests")
        if raw is None:
            inc_cache_miss()
            return
        try:
            data = json.loads(raw)
        except Exception as exc:
            inc_redis_error()
            inc_cache_miss()
            log_json(logger, logging.ERROR, "redis_decode_error", topic=topic, stage="update_tests", error=str(exc))
            return
        if not isinstance(data, dict):
            inc_redis_error()
            inc_cache_miss()
            log_json(logger, logging.ERROR, "redis_payload_invalid", topic=topic, stage="update_tests")
            return

        data["tests"] = tests
        try:
            payload = json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            inc_redis_error()
            log_json(logger, logging.ERROR, "redis_encode_error", topic=topic, stage="update_tests", error=str(exc))
            return

        ttl = await run("ttl", lambda: self._redis.ttl(key), stage="update_tests")
        if ttl is None or ttl < 0:
            ttl = self._ttl_seconds

        await run("set", lambda: self._redis.set(key, payload, ex=ttl), stage="update_tests")
