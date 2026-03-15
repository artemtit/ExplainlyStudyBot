from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, Update

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[Update, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        try:
            return await handler(event, data)
        except ValueError as exc:
            logger.warning("Invalid input: %s", exc)
            await _safe_reply(event, "Некорректный ввод. Попробуй еще раз.")
        except asyncio.TimeoutError:
            logger.warning("Operation timed out")
            await _safe_reply(event, "Время ожидания истекло. Попробуй еще раз.")
        except RuntimeError as exc:
            logger.exception("API error: %s", exc)
            await _safe_reply(event, "Сервис временно недоступен. Попробуй позже.")
        except Exception:
            logger.exception("Unhandled error")
            await _safe_reply(event, "Произошла ошибка. Попробуй еще раз позже.")
        return None


async def _safe_reply(event: Update, text: str) -> None:
    message: Message | None = None
    if isinstance(event, Message):
        message = event
    elif event.message:
        message = event.message
    if message:
        await message.answer(text)
