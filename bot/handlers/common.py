from __future__ import annotations

from typing import Any

from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.keyboards.main_menu import main_menu_kb
from bot.keyboards.study_menu import study_menu_kb
from bot.states.study_state import StudyState
from bot.utils.strings import (
    GENERATION_TEXT,
    SESSION_EXPIRED_TEXT,
    START_TEXT,
    TOPIC_FORMATS_TEXT,
    TOPIC_HINT_TEXT,
)
from bot.utils.telegram_utils import edit_or_send


async def show_start_screen(message: Message) -> None:
    await message.answer(START_TEXT, reply_markup=main_menu_kb())


async def show_formats_screen(call: CallbackQuery | Message, topic: str) -> None:
    text = TOPIC_FORMATS_TEXT.format(topic=topic)
    if isinstance(call, CallbackQuery):
        await edit_or_send(call, text, reply_markup=study_menu_kb())
        return
    await call.answer(text, reply_markup=study_menu_kb())


async def ensure_material_data(call: CallbackQuery, state: FSMContext) -> tuple[dict[str, Any], str] | None:
    data = await state.get_data()
    material = data.get("material")
    topic = data.get("topic")
    if not isinstance(material, dict) or not isinstance(topic, str) or not topic:
        await call.answer()
        await call.message.answer(SESSION_EXPIRED_TEXT)
        await show_start_screen(call.message)
        await state.set_state(StudyState.awaiting_topic)
        return None
    return material, topic
