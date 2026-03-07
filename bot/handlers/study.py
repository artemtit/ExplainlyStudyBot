from __future__ import annotations

import logging
import time
from contextlib import suppress

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers.start import show_main_menu
from bot.services.material_service import MaterialService
from bot.states.study_state import StudyState
from bot.ui.formatting import SEPARATOR, format_lesson, format_no_resume, format_topic_prompt
from bot.ui.keyboards import (
    BTN_CONTINUE,
    BTN_FLASHCARDS,
    BTN_PROFILE,
    BTN_PROGRESS,
    BTN_START_LEARNING,
    BTN_TEST,
    create_lesson_keyboard,
)
from bot.utils.locks import UserLockManager
from bot.utils.strings import FREE_NOTICE_TEXT, GENERATION_TEXT, TEMP_UNAVAILABLE_TEXT
from bot.utils.telegram_utils import edit_or_send
from bot.utils.text_format import split_text_by_limit
from bot.utils.topic_utils import validate_topic

logger = logging.getLogger(__name__)

GEN_RATE_LIMIT_SECONDS = 5.0
_LAST_GEN_AT: dict[int, float] = {}
RATE_LIMIT_TEXT = (
    f"{SEPARATOR}\n\u23F3 \u041D\u0435\u043C\u043D\u043E\u0433\u043E \u043F\u043E\u0434\u043E\u0436\u0434\u0438\u0442\u0435\n{SEPARATOR}\n\n"
    "\u041F\u043E\u043F\u0440\u043E\u0431\u0443\u0439\u0442\u0435 \u0435\u0449\u0451 \u0440\u0430\u0437 \u0447\u0435\u0440\u0435\u0437 \u043F\u0430\u0440\u0443 \u0441\u0435\u043A\u0443\u043D\u0434."
)


async def render_lesson(target: Message | CallbackQuery, state: FSMContext, service: MaterialService) -> None:
    data = await state.get_data()
    material = data.get("material")
    topic = data.get("topic")
    if not isinstance(material, dict) or not isinstance(topic, str) or not topic:
        await show_main_menu(target)
        await state.clear()
        return

    lesson = material.get("lesson", {}) if isinstance(material.get("lesson"), dict) else {}
    text = format_lesson(topic, lesson)
    chunks = split_text_by_limit(text, limit=3600)
    if not chunks:
        chunks = [text]

    if isinstance(target, CallbackQuery):
        await edit_or_send(target, chunks[0], reply_markup=create_lesson_keyboard())
        message = target.message
        user_id = target.from_user.id
    else:
        await target.answer(chunks[0], reply_markup=create_lesson_keyboard())
        message = target
        user_id = target.from_user.id

    for chunk in chunks[1:]:
        await message.answer(chunk)

    await state.set_state(StudyState.in_lesson)
    await state.update_data(flash_show_answer=False, practice_show_solution=False)

    card_index = int(data.get("card_index", 0))
    test_index = int(data.get("test_index", 0))
    test_score = int(data.get("test_score", 0))
    await maybe_save_resume_state(
        state,
        service,
        user_id=user_id,
        topic=topic,
        stage="lesson",
        card_index=card_index,
        test_index=test_index,
        test_score=test_score,
    )
    await service.record_activity(user_id, last_topic=topic, last_stage="lesson")


async def maybe_save_resume_state(
    state: FSMContext,
    service: MaterialService,
    *,
    user_id: int,
    topic: str,
    stage: str,
    card_index: int,
    test_index: int,
    test_score: int,
) -> None:
    data = await state.get_data()
    last_stage = data.get("resume_stage")
    last_card_index = data.get("resume_card_index")
    last_test_index = data.get("resume_test_index")

    if not isinstance(last_card_index, int):
        last_card_index = None
    if not isinstance(last_test_index, int):
        last_test_index = None

    should_save = stage != last_stage
    if stage == "flashcards" and card_index != last_card_index:
        should_save = True
    if stage == "test" and test_index != last_test_index:
        should_save = True

    if not should_save:
        return

    await service.save_resume_state(
        user_id,
        topic=topic,
        stage=stage,
        card_index=card_index,
        test_index=test_index,
        test_score=test_score,
    )
    await state.update_data(
        resume_stage=stage,
        resume_card_index=card_index,
        resume_test_index=test_index,
    )


async def _load_topic(
    *,
    topic: str,
    sender: Message | CallbackQuery,
    state: FSMContext,
    service: MaterialService,
    free_tier_notice: bool,
) -> bool:
    user = sender.from_user
    user_id = user.id
    username = user.username

    status_target = sender if isinstance(sender, Message) else sender.message

    gen_msg = await status_target.answer(GENERATION_TEXT)
    free_msg = None
    if free_tier_notice:
        free_msg = await status_target.answer(FREE_NOTICE_TEXT)

    try:
        material, _ = await service.get_or_generate_material(user_id=user_id, username=username, topic=topic)
    except Exception:
        logger.exception("Failed to get material for topic: %s", topic)
        with suppress(Exception):
            if gen_msg:
                await gen_msg.delete()
            if free_msg:
                await free_msg.delete()
        await status_target.answer(TEMP_UNAVAILABLE_TEXT)
        return False

    await state.set_state(StudyState.in_lesson)
    await state.update_data(
        topic=topic,
        material=material,
        card_index=0,
        test_index=0,
        test_score=0,
        flash_show_answer=False,
        practice_show_solution=False,
        resume_stage=None,
        resume_card_index=None,
        resume_test_index=None,
    )

    await service.record_activity(user_id, last_topic=topic, last_stage="lesson")

    with suppress(Exception):
        if gen_msg:
            await gen_msg.delete()
        if free_msg:
            await free_msg.delete()

    await render_lesson(sender, state, service)
    return True


async def _resume_flow(
    *,
    message: Message,
    state: FSMContext,
    service: MaterialService,
) -> None:
    resume = await service.load_resume_state(message.from_user.id)
    if not resume:
        await show_main_menu(message, text=format_no_resume())
        return

    topic = str(resume.get("last_topic") or resume.get("topic") or "")
    if not topic:
        await show_main_menu(message, text=format_no_resume())
        return

    try:
        material, _ = await service.get_or_generate_material(
            user_id=message.from_user.id,
            username=message.from_user.username,
            topic=topic,
        )
    except Exception:
        logger.exception("Failed to resume material for topic: %s", topic)
        await message.answer(TEMP_UNAVAILABLE_TEXT)
        return

    await state.update_data(
        topic=topic,
        material=material,
        card_index=int(resume.get("card_index", 0)),
        test_index=int(resume.get("test_index", 0)),
        test_score=int(resume.get("test_score", 0)),
        flash_show_answer=False,
        practice_show_solution=False,
        resume_stage=resume.get("last_stage"),
        resume_card_index=int(resume.get("card_index", 0)),
        resume_test_index=int(resume.get("test_index", 0)),
    )

    stage = str(resume.get("last_stage") or "lesson")
    if stage == "flashcards":
        from bot.handlers.flashcards import open_flashcards

        await open_flashcards(message, state, service, reset_index=False)
        return

    if stage == "test":
        from bot.handlers.tests import open_test

        await open_test(message, state, service, reset_progress=False)
        return

    if stage == "practice":
        from bot.handlers.tests import open_practice

        await open_practice(message, state, service, show_solution=False)
        return

    await render_lesson(message, state, service)


def build_router(material_service: MaterialService, lock_manager: UserLockManager, free_tier_notice: bool) -> Router:
    router = Router(name="study")

    @router.message(F.text == BTN_START_LEARNING)
    async def start_learning_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await state.set_state(StudyState.awaiting_topic)
            await message.answer(format_topic_prompt())

    @router.message(F.text == BTN_CONTINUE)
    async def continue_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await _resume_flow(message=message, state=state, service=material_service)

    @router.message(StateFilter(StudyState.awaiting_topic), F.text)
    async def topic_message_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            text = message.text or ""
            if text.startswith("/"):
                return
            if text in {BTN_START_LEARNING, BTN_CONTINUE, BTN_FLASHCARDS, BTN_TEST, BTN_PROGRESS, BTN_PROFILE}:
                return

            topic = validate_topic(text)
            if not topic:
                await message.answer(format_topic_prompt())
                return
            now = time.monotonic()
            last = _LAST_GEN_AT.get(message.from_user.id, 0.0)
            if now - last < GEN_RATE_LIMIT_SECONDS:
                await message.answer(RATE_LIMIT_TEXT)
                return
            _LAST_GEN_AT[message.from_user.id] = now

            await _load_topic(
                topic=topic,
                sender=message,
                state=state,
                service=material_service,
                free_tier_notice=free_tier_notice,
            )

    @router.callback_query(F.data == "lesson:back")
    async def lesson_back_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await show_main_menu(call)
            await state.set_state(StudyState.in_lesson)

    return router
