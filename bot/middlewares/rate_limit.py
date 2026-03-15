from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, Update


class RateLimitMiddleware(BaseMiddleware):
    def __init__(self, *, min_interval_seconds: float = 1.0) -> None:
        self._min_interval = min_interval_seconds
        self._last_seen: dict[int, float] = {}

    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        user_id = _extract_user_id(event)
        if user_id is None:
            return await handler(event, data)
        now = time.monotonic()
        last = self._last_seen.get(user_id, 0.0)
        if now - last < self._min_interval:
            if event.message:
                await event.message.answer("Слишком часто. Подождите пару секунд.")
            return None
        self._last_seen[user_id] = now
        return await handler(event, data)


def _extract_user_id(event: Update) -> int | None:
    if isinstance(event, Message):
        return event.from_user.id if event.from_user else None
    if event.message and event.message.from_user:
        return event.message.from_user.id
    return None
