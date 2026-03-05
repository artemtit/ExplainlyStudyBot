from __future__ import annotations

from contextlib import suppress

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, InlineKeyboardMarkup


async def edit_or_send(
    call: CallbackQuery,
    text: str,
    *,
    reply_markup: InlineKeyboardMarkup | None = None,
    parse_mode: str | None = None,
) -> None:
    with suppress(TelegramBadRequest):
        await call.message.edit_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
        return
    await call.message.answer(text, reply_markup=reply_markup, parse_mode=parse_mode)
