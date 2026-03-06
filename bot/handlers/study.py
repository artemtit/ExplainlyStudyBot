from __future__ import annotations

import logging
from contextlib import suppress

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.handlers.cards import open_cards
from bot.handlers.common import (
    SESSION_EXPIRED_TEXT,
    show_formats_screen,
    show_start_screen,
)
from bot.handlers.lesson import open_lesson
from bot.handlers.practice import open_practice
from bot.handlers.tests import open_test
from bot.keyboards.study_menu import study_topics_kb
from bot.services.material_service import MaterialService
from bot.states.study_state import StudyState
from bot.utils.locks import UserLockManager
from bot.utils.strings import (
    FREE_NOTICE_TEXT,
    GENERATION_TEXT,
    LOADED_CACHE,
    LOADED_DB,
    LOADED_NEW,
    NO_RECENT_TOPICS,
    RECENT_TOPICS,
    TEMP_UNAVAILABLE_TEXT,
    TOPIC_HINT_TEXT,
    TOPIC_INDEX_INVALID,
    UNKNOWN_STAGE,
)
from bot.utils.telegram_utils import edit_or_send
from bot.utils.topic_utils import validate_topic

logger = logging.getLogger(__name__)


async def _load_topic(
    *,
    topic: str,
    sender: Message | CallbackQuery,
    state: FSMContext,
    service: MaterialService,
    free_tier_notice: bool,
) -> None:
    user = sender.from_user
    user_id = user.id
    username = user.username

    status_target = sender if isinstance(sender, Message) else sender.message

    gen_msg = await status_target.answer(GENERATION_TEXT)
    free_msg = None
    if free_tier_notice:
        free_msg = await status_target.answer(FREE_NOTICE_TEXT)

    try:
        material, source = await service.get_or_generate_material(user_id=user_id, username=username, topic=topic)
    except Exception:
        logger.exception("Failed to get material for topic: %s", topic)
        with suppress(Exception):
            if gen_msg:
                await gen_msg.delete()
            if free_msg:
                await free_msg.delete()
        await status_target.answer(TEMP_UNAVAILABLE_TEXT)
        return

    await state.set_state(StudyState.material_ready)
    await state.update_data(
        topic=topic,
        material=material,
        current_stage=None,
        card_index=0,
        test_index=0,
        test_score=0,
        accepting_answer=False,
    )

    with suppress(Exception):
        if gen_msg:
            await gen_msg.delete()
        if free_msg:
            await free_msg.delete()

    if source == "db":
        await status_target.answer(LOADED_DB)
    elif source == "cache":
        await status_target.answer(LOADED_CACHE)
    else:
        await status_target.answer(LOADED_NEW)

    await show_formats_screen(status_target, topic)


def build_router(material_service: MaterialService, lock_manager: UserLockManager, free_tier_notice: bool) -> Router:
    router = Router(name="study")

    @router.callback_query(F.data == "study")
    async def study_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            topics = await material_service.get_recent_topics(call.from_user.id, limit=5)
            await state.update_data(last_topics=topics)

            if not topics:
                keyboard = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text="\u2795 \u041D\u043E\u0432\u0430\u044F \u0442\u0435\u043C\u0430", callback_data="new_topic")],
                        [InlineKeyboardButton(text="\u2B05\uFE0F \u041D\u0430\u0437\u0430\u0434", callback_data="back_to_start")],
                    ]
                )
                await edit_or_send(call, NO_RECENT_TOPICS, reply_markup=keyboard)
                return

            await edit_or_send(call, RECENT_TOPICS, reply_markup=study_topics_kb(topics))

    @router.callback_query(F.data == "new_topic")
    async def new_topic_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await state.set_state(StudyState.awaiting_topic)
            await edit_or_send(call, TOPIC_HINT_TEXT)

    @router.callback_query(F.data.startswith("topic_idx:"))
    async def select_topic_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            try:
                idx = int((call.data or "").split(":", 1)[1])
            except Exception:
                await call.message.answer(TOPIC_INDEX_INVALID)
                return

            data = await state.get_data()
            topics = data.get("last_topics")
            if not isinstance(topics, list) or idx < 0 or idx >= len(topics):
                await call.message.answer(TOPIC_INDEX_INVALID)
                return

            topic = str(topics[idx])
            await _load_topic(
                topic=topic,
                sender=call,
                state=state,
                service=material_service,
                free_tier_notice=free_tier_notice,
            )

    @router.message(StateFilter(StudyState.awaiting_topic), F.text)
    async def topic_message_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            text = message.text or ""
            if text.startswith("/"):
                return

            topic = validate_topic(text)
            if not topic:
                await message.answer(TOPIC_HINT_TEXT)
                return

            await _load_topic(
                topic=topic,
                sender=message,
                state=state,
                service=material_service,
                free_tier_notice=free_tier_notice,
            )

    @router.callback_query(F.data == "back_to_formats")
    async def back_to_formats_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            data = await state.get_data()
            topic = data.get("topic")
            if not isinstance(topic, str) or not topic:
                await call.message.answer(SESSION_EXPIRED_TEXT)
                await show_start_screen(call.message)
                await state.set_state(StudyState.awaiting_topic)
                return
            await show_formats_screen(call, topic)

    @router.callback_query(F.data.startswith("next_stage:"))
    async def next_stage_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            stage = (call.data or "").split(":", 1)[1]
            if stage == "lesson":
                await open_lesson(call, state)
                return
            if stage == "cards":
                await open_cards(call, state, reset_index=True)
                return
            if stage == "test":
                await open_test(call, state, reset_progress=True)
                return
            if stage == "practice":
                await open_practice(call, state)
                return
            await call.message.answer(UNKNOWN_STAGE)

    @router.callback_query(F.data == "finish_lesson")
    async def finish_lesson_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            data = await state.get_data()
            topic = str(data.get("topic") or "")
            await state.set_state(StudyState.awaiting_topic)
            await state.update_data(topic=None, material=None, current_stage=None)
            if topic:
                await edit_or_send(call, "\u2705 \u0422\u044b \u0437\u0430\u0432\u0435\u0440\u0448\u0438\u043b \u0442\u0435\u043c\u0443: {topic}\n\u0427\u0442\u043e \u0434\u0430\u043b\u044c\u0448\u0435?".format(topic=topic))
            await show_start_screen(call.message)

    return router
