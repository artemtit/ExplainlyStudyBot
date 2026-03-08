from __future__ import annotations

from datetime import datetime
from typing import Any

from supabase import Client

from bot.core.models import LearningSession
from bot.infrastructure.supabase_base import SupabaseRepository
from bot.utils.topic_utils import normalize_topic


class SupabaseProgressRepository(SupabaseRepository):
    def __init__(self, client: Client) -> None:
        super().__init__(client)

    def _load_session_sync(self, user_id: int) -> dict[str, Any] | None:
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

    async def load_session(self, user_id: int) -> LearningSession | None:
        payload = await self._to_thread("load_session", self._load_session_sync, user_id)
        if not payload:
            return None
        return LearningSession(
            user_id=user_id,
            topic=str(payload.get("last_topic") or ""),
            stage=str(payload.get("last_stage") or "lesson"),
            card_index=int(payload.get("card_index", 0)),
            test_index=int(payload.get("test_index", 0)),
            test_score=int(payload.get("test_score", 0)),
        )

    def _save_session_sync(self, session: LearningSession) -> Any:
        payload = {
            "user_id": session.user_id,
            "last_topic": normalize_topic(session.topic),
            "last_stage": session.stage,
            "card_index": session.card_index,
            "test_index": session.test_index,
            "test_score": session.test_score,
            "last_updated": datetime.utcnow().isoformat(),
        }
        return self._client.table("user_stats").upsert(payload, on_conflict="user_id").execute()

    async def save_session(self, session: LearningSession) -> None:
        await self._to_thread("save_session", self._save_session_sync, session)

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

    def _reset_progress_sync(self, user_id: int) -> Any:
        return self._client.table("user_topics").delete().eq("user_id", user_id).execute()

    async def reset_progress(self, user_id: int) -> None:
        await self._to_thread("reset_progress", self._reset_progress_sync, user_id)


__all__ = ["SupabaseProgressRepository"]
