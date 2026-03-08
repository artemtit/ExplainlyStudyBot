from __future__ import annotations

from typing import Any

from supabase import Client

from bot.infrastructure.supabase_base import SupabaseRepository
from bot.utils.topic_utils import normalize_topic, topic_hash


class SupabaseMaterialRepository(SupabaseRepository):
    def __init__(self, client: Client) -> None:
        super().__init__(client)

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


__all__ = ["SupabaseMaterialRepository"]
