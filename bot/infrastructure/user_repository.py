from __future__ import annotations

from typing import Any

from supabase import Client

from bot.infrastructure.supabase_base import SupabaseRepository


class SupabaseUserRepository(SupabaseRepository):
    def __init__(self, client: Client) -> None:
        super().__init__(client)

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


__all__ = ["SupabaseUserRepository"]
