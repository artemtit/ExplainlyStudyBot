from __future__ import annotations

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from bot.handlers.common import ensure_material_data
from bot.keyboards.study_menu import practice_kb, stage_nav_kb
from bot.utils.locks import UserLockManager
from bot.utils.strings import PRACTICE_MISSING, PRACTICE_TEXT, SOLUTION_TEXT
from bot.utils.telegram_utils import edit_or_send
from bot.utils.text_format import format_stage_header


async def open_practice(call: CallbackQuery, state: FSMContext) -> None:
    await state.update_data(current_stage="practice")
    payload = await ensure_material_data(call, state)
    if payload is None:
        return

    material, topic = payload
    practice = material.get("practice", {})
    if not isinstance(practice, dict):
        practice = {}

    problem = str(practice.get("problem") or "")
    header = format_stage_header(topic, "practice")

    if not problem:
        await edit_or_send(call, f"{header}\n\n{PRACTICE_MISSING}", reply_markup=stage_nav_kb("practice"))
        return

    await edit_or_send(call, f"{header}\n\n{PRACTICE_TEXT.format(problem=problem)}", reply_markup=practice_kb())


def build_router(lock_manager: UserLockManager) -> Router:
    router = Router(name="practice")

    @router.callback_query(F.data == "practice")
    async def practice_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await open_practice(call, state)

    @router.callback_query(F.data == "solution")
    async def solution_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            payload = await ensure_material_data(call, state)
            if payload is None:
                return

            material, topic = payload
            practice = material.get("practice", {})
            if not isinstance(practice, dict):
                practice = {}

            solution = str(practice.get("solution") or "—")
            header = format_stage_header(topic, "practice")
            await edit_or_send(call, f"{header}\n\n{SOLUTION_TEXT.format(solution=solution)}", reply_markup=stage_nav_kb("practice"))

    return router
