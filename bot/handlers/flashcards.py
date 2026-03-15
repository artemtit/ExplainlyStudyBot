from __future__ import annotations

import logging
import random

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers.start import show_main_menu
from bot.handlers.study import maybe_save_resume_state, render_lesson
from bot.learning_engine.engine import LearningEngine
from bot.states.study_state import StudyState
from bot.ui.formatting import SEPARATOR, format_flashcard, format_no_resume
from bot.ui.keyboards import BTN_FLASHCARDS, create_flashcards_keyboard, create_lesson_keyboard
from bot.utils.locks import UserLockManager
from bot.utils.telegram_utils import edit_or_send

logger = logging.getLogger(__name__)


async def _send(target: Message | CallbackQuery, text: str, *, reply_markup) -> None:
    if isinstance(target, CallbackQuery):
        await edit_or_send(target, text, reply_markup=reply_markup)
    else:
        await target.answer(text, reply_markup=reply_markup)


async def _ensure_material(
    source: Message | CallbackQuery,
    state: FSMContext,
    service: LearningEngine,
) -> tuple[dict, str] | None:
    data = await state.get_data()
    material = data.get("material")
    topic = data.get("topic")
    if isinstance(material, dict) and isinstance(topic, str) and topic:
        return material, topic

    resume = await service.load_resume_state(source.from_user.id)
    if not resume:
        await show_main_menu(source, text=format_no_resume())
        return None

    topic = str(resume.get("last_topic") or resume.get("topic") or "")
    if not topic:
        await show_main_menu(source, text=format_no_resume())
        return None

    try:
        material, _ = await service.get_or_generate_material(
            user_id=source.from_user.id,
            username=source.from_user.username,
            topic=topic,
        )
    except Exception:
        logger.exception("Failed to load material for flashcards: %s", topic)
        await show_main_menu(source, text=format_no_resume())
        return None

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
    return material, topic


async def _render_card(target: Message | CallbackQuery, state: FSMContext, service: LearningEngine) -> None:
    payload = await _ensure_material(target, state, service)
    if payload is None:
        return

    material, topic = payload
    cards = material.get("cards", []) if isinstance(material.get("cards"), list) else []

    data = await state.get_data()
    index = int(data.get("card_index", 0))
    order = data.get("flash_order")
    if isinstance(order, list) and order:
        order = [idx for idx in order if isinstance(idx, int) and 0 <= idx < len(cards)]
        if order:
            cards = [cards[idx] for idx in order]

    show_answer = bool(data.get("flash_show_answer", False))

    if not cards:
        text = f"{SEPARATOR}\n\U0001F9E0 \u0424\u043B\u044D\u0448\u043A\u0430\u0440\u0442\u044B\n{SEPARATOR}\n\n\u041A\u0430\u0440\u0442\u043E\u0447\u0435\u043A \u043F\u043E \u0442\u0435\u043C\u0435 \u043D\u0435\u0442."
        await _send(target, text, reply_markup=create_lesson_keyboard())
        return

    if index >= len(cards):
        text = (
            f"{SEPARATOR}\n\U0001F9E0 \u0424\u043B\u044D\u0448\u043A\u0430\u0440\u0442\u044B\n{SEPARATOR}\n\n"
            "\u0412\u0441\u0435 \u043A\u0430\u0440\u0442\u043E\u0447\u043A\u0438 \u043F\u0440\u043E\u0441\u043C\u043E\u0442\u0440\u0435\u043D\u044B."
        )
        await _send(target, text, reply_markup=create_lesson_keyboard())
        return

    item = cards[index] if isinstance(cards[index], dict) else {}
    question = str(item.get("question") or "\u2014")
    answer = str(item.get("answer") or "\u2014")

    text = format_flashcard(
        question=question,
        answer=answer,
        index=index + 1,
        total=len(cards),
        show_answer=show_answer,
    )

    await _send(target, text, reply_markup=create_flashcards_keyboard(show_answer=show_answer))


async def open_flashcards(
    target: Message | CallbackQuery,
    state: FSMContext,
    service: LearningEngine,
    *,
    reset_index: bool,
) -> None:
    payload = await _ensure_material(target, state, service)
    if payload is None:
        return

    await state.set_state(StudyState.in_flashcards)
    if reset_index:
        await state.update_data(card_index=0)
    await state.update_data(flash_show_answer=False, flash_order=None)

    data = await state.get_data()
    topic = str(data.get("topic") or "")
    user_id = target.from_user.id
    await maybe_save_resume_state(
        state,
        service,
        user_id=user_id,
        topic=topic,
        stage="flashcards",
        card_index=int(data.get("card_index", 0)),
        test_index=int(data.get("test_index", 0)),
        test_score=int(data.get("test_score", 0)),
    )
    await service.record_activity(user_id, last_topic=topic, last_stage="flashcards")

    await _render_card(target, state, service)


def build_router(material_service: LearningEngine, lock_manager: UserLockManager) -> Router:
    router = Router(name="flashcards")

    @router.message(F.text == BTN_FLASHCARDS)
    async def flashcards_menu_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await open_flashcards(message, state, material_service, reset_index=True)

    @router.callback_query(F.data == "lesson:flashcards")
    async def flashcards_lesson_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await open_flashcards(call, state, material_service, reset_index=True)

    @router.callback_query(F.data == "flash:show")
    async def flash_show_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await state.update_data(flash_show_answer=True)
            data = await state.get_data()
            topic = str(data.get("topic") or "")
            await material_service.record_activity(
                call.from_user.id,
                flashcards_delta=1,
                last_topic=topic,
                last_stage="flashcards",
            )
            await _render_card(call, state, material_service)

    @router.callback_query(F.data == "flash:next")
    async def flash_next_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            data = await state.get_data()
            next_index = int(data.get("card_index", 0)) + 1
            await state.update_data(card_index=next_index, flash_show_answer=False)
            topic = str(data.get("topic") or "")
            await maybe_save_resume_state(
                state,
                material_service,
                user_id=call.from_user.id,
                topic=topic,
                stage="flashcards",
                card_index=next_index,
                test_index=int(data.get("test_index", 0)),
                test_score=int(data.get("test_score", 0)),
            )
            await material_service.record_activity(call.from_user.id, last_topic=topic, last_stage="flashcards")
            await _render_card(call, state, material_service)

    @router.callback_query(F.data == "flash:restart")
    async def flash_restart_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await state.update_data(card_index=0, flash_show_answer=False)
            data = await state.get_data()
            topic = str(data.get("topic") or "")
            await maybe_save_resume_state(
                state,
                material_service,
                user_id=call.from_user.id,
                topic=topic,
                stage="flashcards",
                card_index=0,
                test_index=int(data.get("test_index", 0)),
                test_score=int(data.get("test_score", 0)),
            )
            await material_service.record_activity(call.from_user.id, last_topic=topic, last_stage="flashcards")
            await _render_card(call, state, material_service)

    @router.callback_query(F.data == "flash:shuffle")
    async def flash_shuffle_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            data = await state.get_data()
            material = data.get("material") if isinstance(data.get("material"), dict) else {}
            cards = material.get("cards", []) if isinstance(material.get("cards"), list) else []
            if not cards:
                await _render_card(call, state, material_service)
                return

            order = list(range(len(cards)))
            random.shuffle(order)
            await state.update_data(card_index=0, flash_show_answer=False, flash_order=order)
            topic = str(data.get("topic") or "")
            await maybe_save_resume_state(
                state,
                material_service,
                user_id=call.from_user.id,
                topic=topic,
                stage="flashcards",
                card_index=0,
                test_index=int(data.get("test_index", 0)),
                test_score=int(data.get("test_score", 0)),
            )
            await material_service.record_activity(call.from_user.id, last_topic=topic, last_stage="flashcards")
            await _render_card(call, state, material_service)

    @router.callback_query(F.data == "flash:back")
    async def flash_back_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await render_lesson(call, state, material_service)

    return router
