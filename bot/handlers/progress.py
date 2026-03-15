from __future__ import annotations

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers.start import show_main_menu
from bot.handlers.study import resume_flow
from bot.learning_engine.engine import LearningEngine
from bot.ui.formatting import format_progress
from bot.ui.keyboards import BTN_PROGRESS, create_progress_keyboard
from bot.utils.locks import UserLockManager


def build_router(material_service: LearningEngine, lock_manager: UserLockManager) -> Router:
    router = Router(name="progress")

    @router.message(F.text == BTN_PROGRESS)
    async def progress_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            stats = await material_service.get_user_stats(message.from_user.id)
            resume = await material_service.load_resume_state(message.from_user.id)
            await message.answer(
                format_progress(stats),
                reply_markup=create_progress_keyboard(can_continue=bool(resume)),
            )

    @router.callback_query(F.data == "progress:continue")
    async def progress_continue_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            if call.message is None:
                return
            await resume_flow(message=call.message, state=state, service=material_service)

    @router.callback_query(F.data == "progress:back")
    async def progress_back_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await show_main_menu(call)

    return router
