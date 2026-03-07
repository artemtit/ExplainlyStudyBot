from __future__ import annotations

import asyncio
import logging
from datetime import datetime
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

    def _get_user_stats_sync(self, user_id: int) -> dict[str, Any] | None:
        response = (
            self._client.table("user_stats")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return response.data[0]

    async def get_user_stats(self, user_id: int) -> dict[str, Any] | None:
        return await self._to_thread("get_user_stats", self._get_user_stats_sync, user_id)

    def _upsert_user_stats_sync(self, payload: dict[str, Any]) -> Any:
        user_id = payload.get("user_id")
        if user_id is None:
            raise ValueError("user_id is required for user_stats upsert")

        existing = self._get_user_stats_sync(int(user_id))
        if existing:
            updated = dict(existing)
            updated.update(payload)
            for field in ("tests_passed", "topics_learned", "flashcards_reviewed"):
                current = int(existing.get(field, 0) or 0)
                desired = int(payload.get(field, current) or 0)
                delta = desired - current
                updated[field] = current + delta
            return (
                self._client.table("user_stats")
                .update(updated)
                .eq("user_id", int(user_id))
                .execute()
            )

        return self._client.table("user_stats").insert(payload).execute()

    async def upsert_user_stats(self, payload: dict[str, Any]) -> None:
        await self._to_thread("upsert_user_stats", self._upsert_user_stats_sync, payload)

    def _topic_exists_sync(self, user_id: int, topic: str) -> bool:
        response = (
            self._client.table("user_topics")
            .select("topic")
            .eq("user_id", user_id)
            .eq("topic", normalize_topic(topic))
            .limit(1)
            .execute()
        )
        return bool(response.data)

    def _mark_topic_completed_sync(self, user_id: int, topic: str) -> bool:
        normalized = normalize_topic(topic)
        payload = {
            "user_id": user_id,
            "topic": normalized,
            "completed_at": datetime.utcnow().isoformat(),
        }
        response = (
            self._client.table("user_topics")
            .upsert(payload, on_conflict="user_id,topic", ignore_duplicates=True)
            .execute()
        )
        return bool(response.data)

    async def mark_topic_completed(self, user_id: int, topic: str) -> bool:
        return await self._to_thread("mark_topic_completed", self._mark_topic_completed_sync, user_id, topic)

    def _load_resume_state_sync(self, user_id: int) -> dict[str, Any] | None:
        response = (
            self._client.table("user_stats")
            .select("last_topic,last_stage,card_index,test_index,test_score")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        payload = response.data[0]
        if not payload.get("last_topic"):
            return None
        return payload

    async def load_resume_state(self, user_id: int) -> dict[str, Any] | None:
        return await self._to_thread("load_resume_state", self._load_resume_state_sync, user_id)

    def _save_resume_state_sync(
        self,
        *,
        user_id: int,
        topic: str,
        stage: str,
        card_index: int,
        test_index: int,
        test_score: int,
    ) -> Any:
        payload = {
            "user_id": user_id,
            "last_topic": normalize_topic(topic),
            "last_stage": stage,
            "card_index": card_index,
            "test_index": test_index,
            "test_score": test_score,
            "last_updated": datetime.utcnow().isoformat(),
        }
        return self._client.table("user_stats").upsert(payload, on_conflict="user_id").execute()

    async def save_resume_state(
        self,
        *,
        user_id: int,
        topic: str,
        stage: str,
        card_index: int,
        test_index: int,
        test_score: int,
    ) -> None:
        await self._to_thread(
            "save_resume_state",
            self._save_resume_state_sync,
            user_id=user_id,
            topic=topic,
            stage=stage,
            card_index=card_index,
            test_index=test_index,
            test_score=test_score,
        )

    def _reset_user_progress_sync(self, user_id: int) -> Any:
        self._client.table("user_topics").delete().eq("user_id", user_id).execute()
        payload = {
            "user_id": user_id,
            "topics_learned": 0,
            "tests_passed": 0,
            "flashcards_reviewed": 0,
            "daily_streak": 0,
            "last_active_date": None,
            "last_topic": None,
            "last_stage": None,
            "card_index": 0,
            "test_index": 0,
            "test_score": 0,
            "last_updated": datetime.utcnow().isoformat(),
        }
        return self._client.table("user_stats").upsert(payload, on_conflict="user_id").execute()

    async def reset_user_progress(self, user_id: int) -> None:
        await self._to_thread("reset_user_progress", self._reset_user_progress_sync, user_id)
