from __future__ import annotations

import logging
import time
from contextlib import suppress

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from bot.handlers.start import show_main_menu
from bot.learning_engine.engine import LearningEngine
from bot.states.study_state import StudyState
from bot.ui.formatting import (
    SEPARATOR,
    format_explanation_prompt,
    format_lesson,
    format_no_resume,
    format_recent_topics,
    format_topic_prompt,
    format_topic_too_long,
    format_topic_too_short,
)
from bot.ui.keyboards import (
    BTN_CONTINUE,
    BTN_FLASHCARDS,
    BTN_PROFILE,
    BTN_PROGRESS,
    BTN_START_LEARNING,
    BTN_TEST,
    create_explanation_level_keyboard,
    create_lesson_keyboard,
    create_recent_topics_keyboard,
)
from bot.utils.locks import UserLockManager
from bot.utils.formula_detector import contains_formula
from bot.utils.formula_renderer import render_formula_image
from bot.utils.strings import (
    FREE_NOTICE_TEXT,
    GENERATION_TEXT,
    LOADED_CACHE,
    LOADED_DB,
    LOADED_NEW,
    TEMP_UNAVAILABLE_TEXT,
)
from bot.utils.telegram_utils import edit_or_send
from bot.utils.text_format import split_text_by_limit
from bot.utils.topic_utils import validate_topic

logger = logging.getLogger(__name__)

GEN_RATE_LIMIT_SECONDS = 5.0
_LAST_GEN_AT: dict[int, float] = {}
MIN_TOPIC_LEN = 2
MAX_TOPIC_LEN = 200
RATE_LIMIT_TEXT = (
    f"{SEPARATOR}\n\u23F3 \u041D\u0435\u043C\u043D\u043E\u0433\u043E \u043F\u043E\u0434\u043E\u0436\u0434\u0438\u0442\u0435\n{SEPARATOR}\n\n"
    "\u041F\u043E\u043F\u0440\u043E\u0431\u0443\u0439\u0442\u0435 \u0435\u0449\u0451 \u0440\u0430\u0437 \u0447\u0435\u0440\u0435\u0437 \u043F\u0430\u0440\u0443 \u0441\u0435\u043A\u0443\u043D\u0434."
)


async def render_lesson(target: Message | CallbackQuery, state: FSMContext, service: LearningEngine) -> None:
    data = await state.get_data()
    material = data.get("material")
    topic = data.get("topic")
    if not isinstance(material, dict) or not isinstance(topic, str) or not topic:
        await show_main_menu(target)
        await state.clear()
        return

    lesson = material.get("lesson", {}) if isinstance(material.get("lesson"), dict) else {}
    text = format_lesson(topic, lesson)
    if isinstance(target, CallbackQuery):
        message = target.message
        user_id = target.from_user.id
    else:
        message = target
        user_id = target.from_user.id

    sender = message
    delete_original = isinstance(target, CallbackQuery) and target.message

    async def delete_original_message() -> None:
        nonlocal delete_original
        if delete_original:
            with suppress(Exception):
                await target.message.delete()
            delete_original = False

    if message is None:
        await show_main_menu(target)
        await state.clear()
        return

    if contains_formula(text):
        images = render_formula_image(text)
        if isinstance(images, list):
            for idx, image in enumerate(images):
                if idx == 0:
                    await delete_original_message()
                markup = create_lesson_keyboard() if idx == len(images) - 1 else None
                await sender.answer_photo(
                    photo=BufferedInputFile(image, filename=f"lesson_{idx + 1}.png"),
                    reply_markup=markup,
                )
        else:
            await delete_original_message()
            await sender.answer_photo(
                photo=BufferedInputFile(images, filename="lesson.png"),
                reply_markup=create_lesson_keyboard(),
            )
    else:
        chunks = split_text_by_limit(text, limit=3600)
        if not chunks:
            chunks = [text]

        if isinstance(target, CallbackQuery):
            await delete_original_message()
            await edit_or_send(target, chunks[0], reply_markup=create_lesson_keyboard())
        else:
            await delete_original_message()
            await sender.answer(chunks[0], reply_markup=create_lesson_keyboard())

        for chunk in chunks[1:]:
            await sender.answer(chunk)

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
    service: LearningEngine,
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
    explanation_level: str | None,
    sender: Message | CallbackQuery,
    state: FSMContext,
    service: LearningEngine,
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
        material, source = await service.start_lesson(
            user_id,
            topic,
            username=username,
            explanation_level=explanation_level,
        )
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
        explanation_level=explanation_level,
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

    status_map = {"cache": LOADED_CACHE, "db": LOADED_DB, "llm": LOADED_NEW}
    status_text = status_map.get(str(source))
    if status_text:
        await status_target.answer(status_text)

    await render_lesson(sender, state, service)
    return True


async def resume_flow(
    *,
    message: Message,
    state: FSMContext,
    service: LearningEngine,
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
        material, source = await service.get_or_generate_material(
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

    status_map = {"cache": LOADED_CACHE, "db": LOADED_DB, "llm": LOADED_NEW}
    status_text = status_map.get(str(source))
    if status_text:
        await message.answer(status_text)

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


async def _ensure_lesson_material(message: Message, state: FSMContext, service: LearningEngine) -> bool:
    data = await state.get_data()
    material = data.get("material")
    topic = data.get("topic")
    if isinstance(material, dict) and isinstance(topic, str) and topic:
        return True

    resume = await service.load_resume_state(message.from_user.id)
    if not resume:
        await show_main_menu(message, text=format_no_resume())
        return False

    topic = str(resume.get("last_topic") or resume.get("topic") or "")
    if not topic:
        await show_main_menu(message, text=format_no_resume())
        return False

    try:
        material, source = await service.get_or_generate_material(
            user_id=message.from_user.id,
            username=message.from_user.username,
            topic=topic,
        )
    except Exception:
        logger.exception("Failed to load material for lesson: %s", topic)
        await message.answer(TEMP_UNAVAILABLE_TEXT)
        return False

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

    status_map = {"cache": LOADED_CACHE, "db": LOADED_DB, "llm": LOADED_NEW}
    status_text = status_map.get(str(source))
    if status_text:
        await message.answer(status_text)

    return True


async def show_topic_entry(message: Message, state: FSMContext, service: LearningEngine, *, user_id: int) -> None:
    await state.set_state(StudyState.awaiting_topic)
    recent_topics = await service.get_recent_topics(user_id, limit=3)
    if recent_topics:
        await state.update_data(recent_topics=recent_topics)
        await message.answer(
            format_recent_topics(recent_topics),
            reply_markup=create_recent_topics_keyboard(recent_topics),
        )
    else:
        await message.answer(format_topic_prompt())


def build_router(material_service: LearningEngine, lock_manager: UserLockManager, free_tier_notice: bool) -> Router:
    router = Router(name="study")

    @router.message(Command("lesson"))
    async def lesson_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            ready = await _ensure_lesson_material(message, state, material_service)
            if ready:
                await render_lesson(message, state, material_service)

    @router.message(Command("continue"))
    async def continue_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await resume_flow(message=message, state=state, service=material_service)

    @router.message(Command("topic"))
    async def topic_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await show_topic_entry(message, state, material_service, user_id=message.from_user.id)

    @router.message(Command("study"))
    async def study_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await show_topic_entry(message, state, material_service, user_id=message.from_user.id)

    @router.message(Command("learn"))
    async def learn_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await show_topic_entry(message, state, material_service, user_id=message.from_user.id)

    @router.message(F.text == BTN_START_LEARNING)
    async def start_learning_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await show_topic_entry(message, state, material_service, user_id=message.from_user.id)

    @router.callback_query(StateFilter(StudyState.awaiting_topic), F.data.startswith("recent:pick:"))
    async def recent_topic_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            data = await state.get_data()
            topics = data.get("recent_topics")
            if not isinstance(topics, list) or not topics:
                await call.answer("\u0422\u0435\u043C\u0430 \u043D\u0435\u0434\u043E\u0441\u0442\u0443\u043F\u043D\u0430")
                return

            raw = (call.data or "").split(":")
            if len(raw) < 3:
                await call.answer("\u0422\u0435\u043C\u0430 \u043D\u0435\u0434\u043E\u0441\u0442\u0443\u043F\u043D\u0430")
                return
            try:
                idx = int(raw[2])
            except ValueError:
                await call.answer("\u0422\u0435\u043C\u0430 \u043D\u0435\u0434\u043E\u0441\u0442\u0443\u043F\u043D\u0430")
                return
            if idx < 0 or idx >= len(topics):
                await call.answer("\u0422\u0435\u043C\u0430 \u043D\u0435\u0434\u043E\u0441\u0442\u0443\u043F\u043D\u0430")
                return

            topic = str(topics[idx]).strip()
            if not topic:
                await call.answer("\u0422\u0435\u043C\u0430 \u043D\u0435\u0434\u043E\u0441\u0442\u0443\u043F\u043D\u0430")
                return

            await state.set_state(StudyState.awaiting_explanation_level)
            await state.update_data(topic=topic)
            await call.answer()
            await call.message.answer(
                format_explanation_prompt(topic),
                reply_markup=create_explanation_level_keyboard(),
            )

    @router.callback_query(StateFilter(StudyState.awaiting_topic), F.data == "recent:new")
    async def recent_new_topic_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await call.message.answer(format_topic_prompt())

    @router.callback_query(StateFilter(StudyState.awaiting_topic), F.data == "recent:menu")
    async def recent_menu_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await show_main_menu(call)
            await state.set_state(StudyState.in_lesson)

    @router.message(F.text == BTN_CONTINUE)
    async def continue_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await resume_flow(message=message, state=state, service=material_service)

    @router.message(StateFilter(StudyState.awaiting_topic), F.text)
    async def topic_message_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            text = message.text or ""
            if text.startswith("/"):
                return
            if text in {BTN_START_LEARNING, BTN_CONTINUE, BTN_FLASHCARDS, BTN_TEST, BTN_PROGRESS, BTN_PROFILE}:
                return
            stripped = text.strip()
            if len(stripped) < MIN_TOPIC_LEN:
                await message.answer(format_topic_too_short(MIN_TOPIC_LEN))
                await message.answer(format_topic_prompt())
                return
            if len(stripped) > MAX_TOPIC_LEN:
                await message.answer(format_topic_too_long(MAX_TOPIC_LEN))
                await message.answer(format_topic_prompt())
                return

            topic = validate_topic(stripped, min_len=MIN_TOPIC_LEN, max_len=MAX_TOPIC_LEN)
            if not topic:
                await message.answer(format_topic_prompt())
                return
            await state.set_state(StudyState.awaiting_explanation_level)
            await state.update_data(topic=topic)
            await message.answer(format_explanation_prompt(topic), reply_markup=create_explanation_level_keyboard())

    @router.callback_query(StateFilter(StudyState.awaiting_explanation_level), F.data.startswith("explain_level:"))
    async def explanation_level_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            data = await state.get_data()
            topic = str(data.get("topic") or "")
            if not topic:
                await state.set_state(StudyState.awaiting_topic)
                await call.message.answer(format_topic_prompt())
                return

            level_raw = (call.data or "").split(":", 1)[-1]
            if level_raw == "back":
                await call.answer()
                if call.message:
                    await show_topic_entry(call.message, state, material_service, user_id=call.from_user.id)
                return
            if level_raw == "menu":
                await call.answer()
                if call.message:
                    await show_main_menu(call)
                    await state.set_state(StudyState.in_lesson)
                return
            explanation_level = level_raw if level_raw in {"simple", "normal", "hard"} else "normal"

            now = time.monotonic()
            last = _LAST_GEN_AT.get(call.from_user.id, 0.0)
            if now - last < GEN_RATE_LIMIT_SECONDS:
                await call.message.answer(RATE_LIMIT_TEXT)
                return
            if len(_LAST_GEN_AT) > 10000:
                _LAST_GEN_AT.clear()
            _LAST_GEN_AT[call.from_user.id] = now

            await state.update_data(explanation_level=explanation_level)
            await _load_topic(
                topic=topic,
                explanation_level=explanation_level,
                sender=call,
                state=state,
                service=material_service,
                free_tier_notice=free_tier_notice,
            )

    @router.callback_query(F.data == "lesson:back")
    async def lesson_back_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await show_main_menu(call)
            await state.set_state(StudyState.in_lesson)

    @router.callback_query(F.data == "lesson:menu")
    async def lesson_menu_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await show_main_menu(call)
            await state.set_state(StudyState.in_lesson)

    return router
