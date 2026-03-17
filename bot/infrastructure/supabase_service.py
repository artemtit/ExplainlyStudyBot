from __future__ import annotations

from supabase import Client

from bot.infrastructure.material_repository import SupabaseMaterialRepository
from bot.infrastructure.progress_repository import SupabaseProgressRepository
from bot.infrastructure.stats_repository import SupabaseStatsRepository
from bot.infrastructure.supabase_client import create_supabase_client
from bot.infrastructure.user_repository import SupabaseUserRepository


class SupabaseService:
    """Compatibility facade that exposes all repository methods on a single object."""

    def __init__(self, url: str, key: str) -> None:
        client: Client = create_supabase_client(url, key)
        self._users = SupabaseUserRepository(client)
        self._materials = SupabaseMaterialRepository(client)
        self._progress = SupabaseProgressRepository(client)
        self._stats = SupabaseStatsRepository(client)

    async def ensure_user(self, user_id: int, username: str | None) -> None:
        await self._users.ensure_user(user_id, username)

    async def save_request(self, user_id: int, topic: str) -> None:
        await self._users.save_request(user_id, topic)

    async def get_last_requests(self, user_id: int, limit: int = 3) -> list[str]:
        return await self._users.get_last_requests(user_id, limit=limit)

    async def get_material(self, topic: str, *, difficulty: str | None = None):
        return await self._materials.get_material(topic, difficulty=difficulty)

    async def save_material(self, topic: str, content, *, difficulty: str | None = None):
        await self._materials.save_material(topic, content, difficulty=difficulty)

    async def save_tests_history(
        self,
        user_id: int,
        topic: str,
        difficulty: str,
        tests: list[dict[str, object]],
        score: int,
        total: int,
    ) -> None:
        await self._stats.save_tests_history(user_id, topic, difficulty, tests, score, total)

    async def get_user_stats(self, user_id: int):
        return await self._stats.get_user_stats(user_id)

    async def upsert_user_stats(self, payload):
        await self._stats.upsert_user_stats(payload)

    async def reset_user_stats(self, user_id: int) -> None:
        await self._stats.reset_user_stats(user_id)

    async def load_session(self, user_id: int):
        return await self._progress.load_session(user_id)

    async def save_session(self, session) -> None:
        await self._progress.save_session(session)

    async def mark_topic_completed(self, user_id: int, topic: str) -> bool:
        return await self._progress.mark_topic_completed(user_id, topic)

    async def reset_progress(self, user_id: int) -> None:
        await self._progress.reset_progress(user_id)

    async def load_resume_state(self, user_id: int) -> dict[str, object] | None:
        session = await self._progress.load_session(user_id)
        if not session:
            return None
        return {
            "last_topic": session.topic,
            "last_stage": session.stage,
            "card_index": session.card_index,
            "test_index": session.test_index,
            "test_score": session.test_score,
        }

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
        from bot.core.models import LearningSession

        await self._progress.save_session(
            LearningSession(
                user_id=user_id,
                topic=topic,
                stage=stage,
                card_index=card_index,
                test_index=test_index,
                test_score=test_score,
            )
        )

    async def reset_user_progress(self, user_id: int) -> None:
        await self.reset_progress(user_id)
        await self.reset_user_stats(user_id)


__all__ = ["SupabaseService"]
