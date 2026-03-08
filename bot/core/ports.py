from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, TypeVar

from bot.core.models import LearningSession

T = TypeVar("T")


class LlmClient(Protocol):
    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        parse_response: Callable[[str], T],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        usage_collector: Callable[[dict[str, int] | None], None] | None = None,
        provider: str | None = None,
    ) -> T:
        ...


class ContentGenerator(Protocol):
    async def generate_material(self, topic: str):
        ...

    async def generate_tests(self, topic: str, difficulty: str):
        ...


class MaterialCache(Protocol):
    async def get(self, topic: str) -> dict[str, Any] | None:
        ...

    async def set(self, topic: str, material: dict[str, Any]) -> None:
        ...

    async def update_tests(self, topic: str, tests: list[dict[str, Any]]) -> None:
        ...


class MaterialRepository(Protocol):
    async def get_material(self, topic: str) -> dict[str, Any] | str | None:
        ...

    async def save_material(self, topic: str, content: dict[str, Any]) -> None:
        ...


class UserRepository(Protocol):
    async def ensure_user(self, user_id: int, username: str | None) -> None:
        ...

    async def save_request(self, user_id: int, topic: str) -> None:
        ...

    async def get_last_requests(self, user_id: int, limit: int = 3) -> list[str]:
        ...


class ProgressRepository(Protocol):
    async def load_session(self, user_id: int) -> LearningSession | None:
        ...

    async def save_session(self, session: LearningSession) -> None:
        ...

    async def mark_topic_completed(self, user_id: int, topic: str) -> bool:
        ...

    async def reset_progress(self, user_id: int) -> None:
        ...


class StatsRepository(Protocol):
    async def get_user_stats(self, user_id: int) -> dict[str, Any] | None:
        ...

    async def upsert_user_stats(self, payload: dict[str, Any]) -> None:
        ...

    async def save_tests_history(
        self,
        user_id: int,
        topic: str,
        difficulty: str,
        tests: list[dict[str, Any]],
        score: int,
        total: int,
    ) -> None:
        ...

    async def reset_user_stats(self, user_id: int) -> None:
        ...
