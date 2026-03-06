from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from supabase import Client, create_client

from bot.utils.topic_utils import normalize_topic, topic_hash

logger = logging.getLogger(__name__)


class SupabaseService:
    def __init__(self, url: str, key: str) -> None:
        self._client: Client = create_client(url, key)

    async def _to_thread(self, label: str, fn, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except httpx.HTTPError as exc:
                last_exc = exc
                delay = 0.5 * (2 ** attempt)
                logger.warning("Supabase transient failure [%s], attempt %d/3: %s", label, attempt + 1, exc)
                await asyncio.sleep(delay)
            except Exception:
                logger.exception("Supabase call failed [%s]", label)
                raise

        logger.error("Supabase failed after retries [%s]: %s", label, last_exc)
        raise RuntimeError(f"Supabase request failed: {label}") from last_exc

    def _upsert_user_sync(self, user_id: int, username: str | None) -> Any:
        return self._client.table("users").upsert({"id": user_id, "username": username}).execute()

    async def ensure_user(self, user_id: int, username: str | None) -> None:
        await self._to_thread("ensure_user", self._upsert_user_sync, user_id, username)

    def _save_request_sync(self, user_id: int, topic: str) -> Any:
        return self._client.table("requests").insert({"user_id": user_id, "topic": topic}).execute()

    async def save_request(self, user_id: int, topic: str) -> None:
        await self._to_thread("save_request", self._save_request_sync, user_id, topic)

    def _get_last_requests_sync(self, user_id: int, limit: int) -> list[str]:
        response = (
            self._client.table("requests")
            .select("topic")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return [str(item.get("topic", "")).strip() for item in (response.data or []) if item.get("topic")]

    async def get_last_requests(self, user_id: int, limit: int = 3) -> list[str]:
        topics = await self._to_thread("get_last_requests", self._get_last_requests_sync, user_id, limit)
        return topics or []

    def _get_material_sync(self, normalized_hash: str) -> dict[str, Any] | str | None:
        response = (
            self._client.table("materials")
            .select("content")
            .eq("topic_hash", normalized_hash)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return response.data[0].get("content")

    async def get_material(self, topic: str) -> dict[str, Any] | str | None:
        return await self._to_thread("get_material", self._get_material_sync, topic_hash(topic))

    def _save_material_sync(self, topic: str, content: dict[str, Any]) -> Any:
        payload = {
            "topic": normalize_topic(topic),
            "topic_hash": topic_hash(topic),
            "content": content,
        }
        return self._client.table("materials").upsert(payload, on_conflict="topic_hash").execute()

    async def save_material(self, topic: str, content: dict[str, Any]) -> None:
        await self._to_thread("save_material", self._save_material_sync, topic, content)

    def _save_tests_history_sync(
        self,
        user_id: int,
        topic: str,
        difficulty: str,
        tests: list[dict[str, Any]],
        score: int,
        total: int,
    ) -> Any:
        payload = {
            "user_id": user_id,
            "topic": normalize_topic(topic),
            "difficulty": difficulty,
            "tests": tests,
            "score": score,
            "total": total,
        }
        return self._client.table("tests_history").insert(payload).execute()

    async def save_tests_history(
        self,
        user_id: int,
        topic: str,
        difficulty: str,
        tests: list[dict[str, Any]],
        score: int,
        total: int,
    ) -> None:
        await self._to_thread(
            "save_tests_history",
            self._save_tests_history_sync,
            user_id,
            topic,
            difficulty,
            tests,
            score,
            total,
        )
