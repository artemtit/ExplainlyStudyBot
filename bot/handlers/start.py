from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.learning_engine.engine import LearningEngine
from bot.ui.formatting import SEPARATOR
from bot.ui.keyboards import create_main_menu
from bot.utils.locks import UserLockManager

logger = logging.getLogger(__name__)

WELCOME_TEXT = (
    f"{SEPARATOR}\n"
    "\U0001F44B \u041F\u0440\u0438\u0432\u0435\u0442! \u042F ExplainlyStudy.\n"
    f"{SEPARATOR}\n\n"
    "\u0412\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u0440\u0435\u0436\u0438\u043C \u043E\u0431\u0443\u0447\u0435\u043D\u0438\u044F."
)


async def show_main_menu(target: Message | CallbackQuery, *, text: str | None = None) -> None:
    message = target.message if isinstance(target, CallbackQuery) else target
    await message.answer(text or WELCOME_TEXT, reply_markup=create_main_menu())
    if isinstance(target, CallbackQuery):
        await target.answer()


def build_router(material_service: LearningEngine, lock_manager: UserLockManager) -> Router:
    router = Router(name="start")

    @router.message(CommandStart())
    async def start_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await material_service.ensure_user(message.from_user.id, message.from_user.username)
            await state.clear()
            await show_main_menu(message)

    return router
