from __future__ import annotations

from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, Update

from services.storage import session_service


class SessionTrackingMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        user_id = _extract_user_id(event)
        if user_id is not None:
            session_service.touch(user_id)
        return await handler(event, data)


def _extract_user_id(event: Update) -> int | None:
    if isinstance(event, Message):
        return event.from_user.id if event.from_user else None
    if event.message and event.message.from_user:
        return event.message.from_user.id
    return None
