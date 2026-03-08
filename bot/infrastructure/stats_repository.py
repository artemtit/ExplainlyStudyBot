from __future__ import annotations

from datetime import datetime
from typing import Any

from supabase import Client

from bot.infrastructure.supabase_base import SupabaseRepository
from bot.utils.topic_utils import normalize_topic


class SupabaseStatsRepository(SupabaseRepository):
    def __init__(self, client: Client) -> None:
        super().__init__(client)

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

    def _reset_user_stats_sync(self, user_id: int) -> Any:
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

    async def reset_user_stats(self, user_id: int) -> None:
        await self._to_thread("reset_user_stats", self._reset_user_stats_sync, user_id)


__all__ = ["SupabaseStatsRepository"]
