from __future__ import annotations

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from bot.handlers.common import ensure_material_data
from bot.keyboards.study_menu import stage_nav_kb
from bot.utils.locks import UserLockManager
from bot.utils.telegram_utils import edit_or_send
from bot.utils.text_format import format_lesson, format_stage_header, split_text_by_limit


async def open_lesson(call: CallbackQuery, state: FSMContext) -> None:
    await state.update_data(current_stage="lesson")
    payload = await ensure_material_data(call, state)
    if payload is None:
        return

    material, topic = payload
    header = format_stage_header(topic, "lesson")
    lesson = material.get("lesson", {})
    text = format_lesson(lesson if isinstance(lesson, dict) else {})

    chunks = split_text_by_limit(text, limit=3600)
    if chunks:
        await edit_or_send(
            call,
            f"{header}\n\n{chunks[0]}",
            parse_mode="HTML",
            reply_markup=stage_nav_kb("lesson"),
        )
    for chunk in chunks[1:]:
        await call.message.answer(chunk, parse_mode="HTML")


def build_router(lock_manager: UserLockManager) -> Router:
    router = Router(name="lesson")

    @router.callback_query(F.data == "lesson")
    async def lesson_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await open_lesson(call, state)

    return router
