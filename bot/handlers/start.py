from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers.common import show_start_screen
from bot.keyboards.main_menu import main_menu_kb
from bot.keyboards.study_menu import profile_kb
from bot.services.material_service import MaterialService
from bot.states.study_state import StudyState
from bot.utils.locks import UserLockManager
from bot.utils.strings import PROFILE_TEXT, START_TEXT
from bot.utils.telegram_utils import edit_or_send

logger = logging.getLogger(__name__)


def build_router(material_service: MaterialService, lock_manager: UserLockManager) -> Router:
    router = Router(name="start")

    @router.message(CommandStart())
    async def start_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await material_service.ensure_user(message.from_user.id, message.from_user.username)
            await state.set_state(StudyState.awaiting_topic)
            await state.update_data(material=None, topic=None)
            await show_start_screen(message)

    @router.callback_query(F.data == "back_to_start")
    async def back_to_start_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await state.set_state(StudyState.awaiting_topic)
            await state.update_data(current_stage=None)
            await edit_or_send(call, START_TEXT, reply_markup=main_menu_kb())

    @router.callback_query(F.data == "profile")
    async def profile_handler(call: CallbackQuery) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            text = PROFILE_TEXT.format(user_id=call.from_user.id)
            await edit_or_send(call, text, reply_markup=profile_kb())

    return router
