from __future__ import annotations

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers.start import show_main_menu
from bot.learning_engine.engine import LearningEngine
from bot.ui.formatting import SEPARATOR
from bot.ui.keyboards import BTN_PROFILE, create_profile_keyboard
from bot.utils.locks import UserLockManager
from bot.utils.telegram_utils import edit_or_send

PROFILE_TEXT = (
    f"{SEPARATOR}\n\U0001F464 \u041F\u0440\u043E\u0444\u0438\u043B\u044C\n{SEPARATOR}\n\n"
    "ID: {user_id}\n"
    "Username: {username}\n\n"
    "\u041F\u043E\u0434\u043F\u0438\u0441\u043A\u0430: Free"
)


def _format_profile(user_id: int, username: str | None) -> str:
    return PROFILE_TEXT.format(user_id=user_id, username=username or "\u2014")


async def _open_profile(target: Message | CallbackQuery) -> None:
    user = target.from_user
    text = _format_profile(user.id, user.username)
    if isinstance(target, CallbackQuery):
        await edit_or_send(target, text, reply_markup=create_profile_keyboard())
    else:
        await target.answer(text, reply_markup=create_profile_keyboard())


def build_router(material_service: LearningEngine, lock_manager: UserLockManager) -> Router:
    router = Router(name="profile")

    @router.message(F.text == BTN_PROFILE)
    async def profile_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await _open_profile(message)

    @router.callback_query(F.data == "profile:back")
    async def profile_back_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await show_main_menu(call)

    @router.callback_query(F.data == "profile:subscription")
    async def profile_subscription_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await _open_profile(call)

    return router
