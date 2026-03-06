from __future__ import annotations

from contextlib import suppress

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from bot.handlers.common import ensure_material_data
from bot.keyboards.study_menu import new_test_kb, stage_nav_kb, test_options_kb
from bot.services.material_service import MaterialService
from bot.states.study_state import StudyState
from bot.utils.locks import UserLockManager
from bot.utils.strings import (
    ANSWER_CORRECT,
    ANSWER_INCORRECT,
    ANSWER_INVALID,
    NEW_TEST_CREATING,
    NEW_TEST_FAILED,
    QUESTION_CLOSED,
    QUESTION_NOT_FOUND,
    TEST_DONE_TEMPLATE,
    TEST_QUESTION_TEMPLATE,
    TESTS_NONE_TEXT,
)
from bot.utils.telegram_utils import edit_or_send
from bot.utils.text_format import format_stage_header


def _extract_correct_letter(test: dict) -> str:
    value = str(test.get("correct") or test.get("answer") or "A").upper().strip()
    return value if value in {"A", "B", "C", "D"} else "A"


def _difficulty_from_score(score: int, total: int) -> str:
    if total <= 0:
        return "medium"
    if score >= 4:
        return "hard"
    if score >= 2:
        return "medium"
    return "easy"


async def _render_question(call: CallbackQuery, state: FSMContext, material_service: MaterialService | None = None) -> None:
    payload = await ensure_material_data(call, state)
    if payload is None:
        return

    material, topic = payload
    tests = material.get("tests", [])
    if not isinstance(tests, list):
        tests = []

    data = await state.get_data()
    index = int(data.get("test_index", 0))
    score = int(data.get("test_score", 0))
    header = format_stage_header(topic, "test")

    if not tests:
        await state.set_state(StudyState.material_ready)
        await state.update_data(accepting_answer=False)
        await edit_or_send(call, f"{header}\n\n{TESTS_NONE_TEXT}", reply_markup=stage_nav_kb("test"))
        return

    if index >= len(tests):
        await state.set_state(StudyState.material_ready)
        await state.update_data(accepting_answer=False)
        body = TEST_DONE_TEMPLATE.format(score=score, total=len(tests))
        await edit_or_send(call, f"{header}\n\n{body}", reply_markup=new_test_kb())
        return

    item = tests[index] if isinstance(tests[index], dict) else {}
    question = str(item.get("question") or "—")
    options = item.get("options")
    if not isinstance(options, list):
        options = []

    while len(options) < 4:
        options.append(f"\u0412\u0430\u0440\u0438\u0430\u043d\u0442 {len(options) + 1}")

    await state.set_state(StudyState.passing_test)
    await state.update_data(accepting_answer=True)

    body = TEST_QUESTION_TEMPLATE.format(pos=index + 1, total=len(tests), question=question)
    await edit_or_send(call, f"{header}\n\n{body}", reply_markup=test_options_kb([str(x) for x in options[:4]], index))


async def open_test(call: CallbackQuery, state: FSMContext, *, reset_progress: bool) -> None:
    await state.update_data(current_stage="test")
    if reset_progress:
        await state.update_data(test_index=0, test_score=0, accepting_answer=False)
    await _render_question(call, state)


def build_router(material_service: MaterialService, lock_manager: UserLockManager) -> Router:
    router = Router(name="tests")

    @router.callback_query(F.data == "test")
    async def test_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await open_test(call, state, reset_progress=True)

    @router.callback_query(F.data.startswith("answer:"))
    async def answer_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            payload = await ensure_material_data(call, state)
            if payload is None:
                await call.answer()
                return

            material, _ = payload
            tests = material.get("tests", [])
            if not isinstance(tests, list):
                tests = []

            try:
                _, idx_raw, selected = (call.data or "").split(":")
                idx = int(idx_raw)
            except Exception:
                await call.answer(ANSWER_INVALID)
                return

            data = await state.get_data()
            active_idx = int(data.get("test_index", 0))
            if not data.get("accepting_answer", False) or idx != active_idx:
                await call.answer(QUESTION_CLOSED)
                return

            if idx < 0 or idx >= len(tests):
                await call.answer(QUESTION_NOT_FOUND)
                return

            current_test = tests[idx] if isinstance(tests[idx], dict) else {}
            correct = _extract_correct_letter(current_test)
            score = int(data.get("test_score", 0))

            if selected == correct:
                score += 1
                await call.answer(ANSWER_CORRECT)
            else:
                await call.answer(ANSWER_INCORRECT.format(correct=correct))

            await state.update_data(test_index=idx + 1, test_score=score, accepting_answer=False)
            await _render_question(call, state, material_service)

    @router.callback_query(F.data == "new_test")
    async def new_test_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            payload = await ensure_material_data(call, state)
            if payload is None:
                return

            material, topic = payload
            data = await state.get_data()
            score = int(data.get("test_score", 0))
            tests = material.get("tests", [])
            total = len(tests) if isinstance(tests, list) else 0

            create_msg = await call.message.answer(NEW_TEST_CREATING)
            difficulty = _difficulty_from_score(score, total)
            try:
                new_tests = await material_service.generate_tests(topic, difficulty)
            except Exception:
                with suppress(Exception):
                    await create_msg.delete()
                await call.message.answer(NEW_TEST_FAILED)
                return

            if not new_tests:
                with suppress(Exception):
                    await create_msg.delete()
                await call.message.answer(NEW_TEST_FAILED)
                return

            material["tests"] = new_tests
            material_service.update_cached_tests(topic, new_tests)
            await material_service.save_tests_history(
                user_id=call.from_user.id,
                topic=topic,
                difficulty=difficulty,
                tests=new_tests,
                score=score,
                total=total,
            )

            await state.update_data(material=material, test_index=0, test_score=0, accepting_answer=False)
            with suppress(Exception):
                await create_msg.delete()
            await _render_question(call, state, material_service)

    return router
