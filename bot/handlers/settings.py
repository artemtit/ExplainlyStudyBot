from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers.start import show_main_menu
from bot.learning_engine.engine import LearningEngine
from bot.ui.formatting import SEPARATOR, format_settings
from bot.ui.keyboards import BTN_SETTINGS, create_reset_confirm_keyboard, create_settings_keyboard
from bot.utils.locks import UserLockManager

NOTIFICATIONS_TEXT = (
    f"{SEPARATOR}\n\U0001F514 \u0423\u0432\u0435\u0434\u043E\u043C\u043B\u0435\u043D\u0438\u044F\n{SEPARATOR}\n\n"
    "\u042D\u0442\u0430 \u0444\u0443\u043D\u043A\u0446\u0438\u044F \u043F\u043E\u043A\u0430 \u0432 \u0440\u0430\u0437\u0440\u0430\u0431\u043E\u0442\u043A\u0435."
)
RESET_DONE_TEXT = (
    f"{SEPARATOR}\n\u26A0 \u0421\u0431\u0440\u043E\u0441 \u043F\u0440\u043E\u0433\u0440\u0435\u0441\u0441\u0430\n{SEPARATOR}\n\n"
    "\u041F\u0440\u043E\u0433\u0440\u0435\u0441\u0441 \u043E\u0447\u0438\u0449\u0435\u043D."
)
RESET_FAILED_TEXT = (
    f"{SEPARATOR}\n\u26A0 \u0421\u0431\u0440\u043E\u0441 \u043F\u0440\u043E\u0433\u0440\u0435\u0441\u0441\u0430\n{SEPARATOR}\n\n"
    "\u041D\u0435 \u0443\u0434\u0430\u043B\u043E\u0441\u044C \u0441\u0431\u0440\u043E\u0441\u0438\u0442\u044C \u043F\u0440\u043E\u0433\u0440\u0435\u0441\u0441."
)
RESET_CONFIRM_TEXT = (
    f"{SEPARATOR}\n\u26A0 \u0421\u0431\u0440\u043E\u0441 \u043F\u0440\u043E\u0433\u0440\u0435\u0441\u0441\u0430\n{SEPARATOR}\n\n"
    "\u0422\u043E\u0447\u043D\u043E \u0441\u0431\u0440\u043E\u0441\u0438\u0442\u044C \u043F\u0440\u043E\u0433\u0440\u0435\u0441\u0441? \u042D\u0442\u043E \u043D\u0435\u043B\u044C\u0437\u044F \u0431\u0443\u0434\u0435\u0442 \u043E\u0442\u043C\u0435\u043D\u0438\u0442\u044C."
)


async def _send_settings(target: Message | CallbackQuery, *, support_url: str | None) -> None:
    message = target.message if isinstance(target, CallbackQuery) else target
    await message.answer(format_settings(), reply_markup=create_settings_keyboard(support_url=support_url))


def build_router(material_service: LearningEngine, lock_manager: UserLockManager, support_url: str | None) -> Router:
    router = Router(name="settings")

    @router.message(F.text == BTN_SETTINGS)
    async def settings_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await _send_settings(message, support_url=support_url)

    @router.message(Command("settings"))
    async def settings_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await _send_settings(message, support_url=support_url)

    @router.message(Command("reset"))
    async def reset_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await message.answer(RESET_CONFIRM_TEXT, reply_markup=create_reset_confirm_keyboard())

    @router.callback_query(F.data == "profile:settings")
    async def profile_settings_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await _send_settings(call, support_url=support_url)

    @router.callback_query(F.data == "settings:notifications")
    async def notifications_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await call.message.answer(NOTIFICATIONS_TEXT)

    @router.callback_query(F.data == "settings:reset")
    async def reset_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await call.message.answer(RESET_CONFIRM_TEXT, reply_markup=create_reset_confirm_keyboard())

    @router.callback_query(F.data == "settings:reset:confirm")
    async def reset_confirm_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            ok = await material_service.reset_progress(call.from_user.id)
            await call.message.answer(
                RESET_DONE_TEXT if ok else RESET_FAILED_TEXT,
                reply_markup=create_settings_keyboard(support_url=support_url),
            )

    @router.callback_query(F.data == "settings:reset:cancel")
    async def reset_cancel_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await _send_settings(call, support_url=support_url)

    @router.callback_query(F.data == "settings:back")
    async def settings_back_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await show_main_menu(call)

    return router
