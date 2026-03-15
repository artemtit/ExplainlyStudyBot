from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers.start import show_main_menu
from bot.handlers.study import maybe_save_resume_state, render_lesson
from bot.learning_engine.engine import LearningEngine
from bot.states.study_state import StudyState
from bot.ui.formatting import SEPARATOR, format_no_resume, format_practice, format_test_question, format_test_result
from bot.ui.keyboards import (
    BTN_TEST,
    create_practice_keyboard,
    create_test_done_keyboard,
    create_test_keyboard,
    create_test_result_keyboard,
    create_test_review_done_keyboard,
)
from bot.utils.locks import UserLockManager
from bot.utils.strings import NEW_TEST_CREATING, NEW_TEST_FAILED
from bot.utils.telegram_utils import edit_or_send

logger = logging.getLogger(__name__)

TEST_LETTERS = ("A", "B", "C", "D")
TEST_DEFAULT_DIFFICULTY = "\u0441\u0440\u0435\u0434\u043d\u044f\u044f"
REVIEW_COMPLETE_TEXT = (
    f"{SEPARATOR}\n\U0001F4D8 \u041F\u043E\u0432\u0442\u043E\u0440 \u043E\u0448\u0438\u0431\u043E\u043A \u0437\u0430\u0432\u0435\u0440\u0448\u0451\u043D\n{SEPARATOR}"
)


async def _send(target: Message | CallbackQuery, text: str, *, reply_markup=None) -> None:
    if isinstance(target, CallbackQuery):
        await edit_or_send(target, text, reply_markup=reply_markup)
    else:
        await target.answer(text, reply_markup=reply_markup)


def _parse_answer_payload(data: str | None) -> tuple[int, str] | None:
    if not data:
        return None
    parts = data.split(":")
    if len(parts) != 4:
        return None
    _, _, idx_raw, selected = parts
    try:
        idx = int(idx_raw)
    except ValueError:
        return None
    selected = selected.strip().upper()
    if selected not in TEST_LETTERS:
        return None
    return idx, selected


def _extract_correct_letter(test: dict) -> str | None:
    value = str(test.get("correct") or test.get("answer") or "").strip().upper()
    if value in TEST_LETTERS:
        return value
    options = test.get("options")
    if isinstance(options, list):
        answer_raw = str(test.get("answer") or "").strip().lower()
        for idx, option in enumerate(options):
            if answer_raw and answer_raw == str(option).strip().lower():
                return TEST_LETTERS[idx]
    return None


def _normalize_options(raw_options: dict | list | None) -> list[str]:
    options: list[str]
    if isinstance(raw_options, dict):
        options = [str(raw_options.get(letter, "")) for letter in TEST_LETTERS]
    elif isinstance(raw_options, list):
        options = [str(option) for option in raw_options]
    else:
        options = []

    while len(options) < 4:
        options.append(f"\u0412\u0430\u0440\u0438\u0430\u043D\u0442 {len(options) + 1}")

    return options[:4]


def _get_review_tests(data: dict) -> list[dict]:
    wrong_tests = data.get("wrong_tests")
    if not isinstance(wrong_tests, list):
        return []
    return [item for item in wrong_tests if isinstance(item, dict)]


def _get_active_tests(data: dict, material: dict) -> tuple[list[dict], int, int, bool]:
    review_mode = bool(data.get("review_mode", False))
    if review_mode:
        tests = _get_review_tests(data)
        index = int(data.get("review_index", 0))
        score = int(data.get("review_score", 0))
        return tests, index, score, True

    tests = material.get("tests", []) if isinstance(material.get("tests"), list) else []
    index = int(data.get("test_index", 0))
    score = int(data.get("test_score", 0))
    return tests, index, score, False


def _append_wrong_test(
    *,
    wrong_tests: list[dict],
    question: str,
    options: list[str],
    correct: str,
    explanation: str,
    max_len: int,
) -> list[dict]:
    if len(wrong_tests) >= max_len:
        return wrong_tests
    key = (question, correct)
    for item in wrong_tests:
        if (str(item.get("question") or ""), str(item.get("correct") or "")) == key:
            return wrong_tests
    wrong_tests.append(
        {
            "question": question,
            "options": options,
            "correct": correct,
            "explanation": explanation,
        }
    )
    return wrong_tests


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
        logger.exception("Failed to load material for tests: %s", topic)
        await show_main_menu(source, text=format_no_resume())
        return None

    await state.update_data(
        topic=topic,
        material=material,
        card_index=int(resume.get("card_index", 0)),
        test_index=int(resume.get("test_index", 0)),
        test_score=int(resume.get("test_score", 0)),
        test_completed=bool(resume.get("test_completed", False)),
        accepting_answer=False,
        flash_show_answer=False,
        practice_show_solution=False,
        wrong_tests=[],
        review_mode=False,
        review_index=0,
        review_score=0,
        resume_stage=resume.get("last_stage"),
        resume_card_index=int(resume.get("card_index", 0)),
        resume_test_index=int(resume.get("test_index", 0)),
    )
    return material, topic


async def _render_question(target: Message | CallbackQuery, state: FSMContext, service: LearningEngine) -> None:
    payload = await _ensure_material(target, state, service)
    if payload is None:
        return

    material, topic = payload
    data = await state.get_data()
    tests, index, score, review_mode = _get_active_tests(data, material)
    completed = bool(data.get("test_completed", False))

    if not tests:
        text = f"{SEPARATOR}\n\U0001F9EA \u0422\u0435\u0441\u0442\n{SEPARATOR}\n\n\u0412 \u043C\u0430\u0442\u0435\u0440\u0438\u0430\u043B\u0435 \u043D\u0435\u0442 \u0442\u0435\u0441\u0442\u043E\u0432."
        await _send(target, text)
        return

    if index >= len(tests):
        if review_mode:
            text = f"{REVIEW_COMPLETE_TEXT}\n\n\u0420\u0435\u0437\u0443\u043B\u044C\u0442\u0430\u0442: {score} / {len(tests)}"
            await _send(target, text, reply_markup=create_test_review_done_keyboard())
            await state.update_data(
                wrong_tests=[],
                review_mode=False,
                review_index=0,
                review_score=0,
            )
        else:
            text = (
                f"{SEPARATOR}\n\U0001F9EA \u0422\u0435\u0441\u0442 \u0437\u0430\u0432\u0435\u0440\u0448\u0451\u043D\n{SEPARATOR}\n\n"
                f"\u0420\u0435\u0437\u0443\u043B\u044C\u0442\u0430\u0442: {score} / {len(tests)}"
            )
            await _send(target, text, reply_markup=create_test_done_keyboard())

            if not completed:
                await state.update_data(test_completed=True)
                topics_delta = 1 if await service.mark_topic_completed(target.from_user.id, topic) else 0
                await service.record_activity(
                    target.from_user.id,
                    tests_passed_delta=1,
                    topics_delta=topics_delta,
                    last_topic=topic,
                    last_stage="lesson",
                )
            await maybe_save_resume_state(
                state,
                service,
                user_id=target.from_user.id,
                topic=topic,
                stage="lesson",
                card_index=int(data.get("card_index", 0)),
                test_index=index,
                test_score=score,
            )
        return

    item = tests[index] if isinstance(tests[index], dict) else {}
    question = str(item.get("question") or "\u2014")
    options = _normalize_options(item.get("options"))

    await state.set_state(StudyState.in_test)
    await state.update_data(accepting_answer=True, test_completed=False)

    text = format_test_question(topic, question, index + 1, len(tests))
    await _send(target, text, reply_markup=create_test_keyboard(options, index))


async def open_test(
    target: Message | CallbackQuery,
    state: FSMContext,
    service: LearningEngine,
    *,
    reset_progress: bool,
) -> None:
    payload = await _ensure_material(target, state, service)
    if payload is None:
        return

    await state.set_state(StudyState.in_test)
    if reset_progress:
        await state.update_data(
            test_index=0,
            test_score=0,
            test_completed=False,
            wrong_tests=[],
        )
    await state.update_data(
        accepting_answer=False,
        review_mode=False,
        review_index=0,
        review_score=0,
    )

    data = await state.get_data()
    topic = str(data.get("topic") or "")
    await maybe_save_resume_state(
        state,
        service,
        user_id=target.from_user.id,
        topic=topic,
        stage="test",
        card_index=int(data.get("card_index", 0)),
        test_index=int(data.get("test_index", 0)),
        test_score=int(data.get("test_score", 0)),
    )
    await service.record_activity(target.from_user.id, last_topic=topic, last_stage="test")

    await _render_question(target, state, service)


async def open_practice(
    target: Message | CallbackQuery,
    state: FSMContext,
    service: LearningEngine,
    *,
    show_solution: bool,
) -> None:
    payload = await _ensure_material(target, state, service)
    if payload is None:
        return

    material, topic = payload
    practice = material.get("practice", {}) if isinstance(material.get("practice"), dict) else {}
    problem = str(practice.get("problem") or "\u2014")
    solution = str(practice.get("solution") or "\u2014")

    await state.set_state(StudyState.in_practice)
    await state.update_data(practice_show_solution=show_solution)
    data = await state.get_data()
    await maybe_save_resume_state(
        state,
        service,
        user_id=target.from_user.id,
        topic=topic,
        stage="practice",
        card_index=int(data.get("card_index", 0)),
        test_index=int(data.get("test_index", 0)),
        test_score=int(data.get("test_score", 0)),
    )
    await service.record_activity(target.from_user.id, last_topic=topic, last_stage="practice")

    text = format_practice(problem, solution, show_solution=show_solution)
    await _send(target, text, reply_markup=create_practice_keyboard(show_solution=show_solution))


def build_router(material_service: LearningEngine, lock_manager: UserLockManager) -> Router:
    router = Router(name="tests")

    @router.message(Command("tests"))
    async def tests_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await open_test(message, state, material_service, reset_progress=True)

    @router.message(Command("test"))
    async def test_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await open_test(message, state, material_service, reset_progress=True)

    @router.message(Command("practice"))
    async def practice_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await open_practice(message, state, material_service, show_solution=False)

    @router.message(F.text == BTN_TEST)
    async def test_menu_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await open_test(message, state, material_service, reset_progress=True)

    @router.callback_query(F.data == "lesson:test")
    async def test_lesson_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await open_test(call, state, material_service, reset_progress=True)

    @router.callback_query(F.data == "lesson:practice")
    async def practice_lesson_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await open_practice(call, state, material_service, show_solution=False)

    @router.callback_query(F.data == "test:next")
    async def test_next_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await _render_question(call, state, material_service)

    @router.callback_query(F.data == "test:review")
    async def test_review_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await render_lesson(call, state, material_service)

    @router.callback_query(F.data == "test:review_wrong")
    async def test_review_wrong_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            data = await state.get_data()
            wrong_tests = _get_review_tests(data)
            if not wrong_tests:
                await call.answer()
                await edit_or_send(call, "\u041e\u0448\u0438\u0431\u043e\u043a \u043d\u0435\u0442. \u041e\u0442\u043b\u0438\u0447\u043d\u0430\u044f \u0440\u0430\u0431\u043e\u0442\u0430.")
                await render_lesson(call, state, material_service)
                return

            await state.update_data(
                review_mode=True,
                review_index=0,
                review_score=0,
                accepting_answer=False,
            )
            await call.answer()
            await _render_question(call, state, material_service)

    @router.callback_query(F.data == "test:new")
    async def test_new_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            payload = await _ensure_material(call, state, material_service)
            if payload is None:
                await call.answer()
                return

            material, topic = payload
            await call.answer()
            await edit_or_send(call, NEW_TEST_CREATING)

            try:
                tests = await material_service.generate_tests(topic, TEST_DEFAULT_DIFFICULTY)
            except Exception:
                logger.exception("Failed to generate new tests", extra={"topic": topic})
                await edit_or_send(call, NEW_TEST_FAILED, reply_markup=create_test_done_keyboard())
                return

            tests_payload = [test.to_dict() for test in tests]
            updated_material = dict(material)
            updated_material["tests"] = tests_payload
            await state.update_data(
                material=updated_material,
                test_index=0,
                test_score=0,
                test_completed=False,
                accepting_answer=False,
                wrong_tests=[],
                review_mode=False,
                review_index=0,
                review_score=0,
            )
            material_service.update_cached_tests(topic, tests)

            data = await state.get_data()
            await maybe_save_resume_state(
                state,
                material_service,
                user_id=call.from_user.id,
                topic=topic,
                stage="test",
                card_index=int(data.get("card_index", 0)),
                test_index=0,
                test_score=0,
            )
            await material_service.record_activity(call.from_user.id, last_topic=topic, last_stage="test")

            await _render_question(call, state, material_service)

    @router.callback_query(F.data == "test:retry")
    async def test_retry_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await open_test(call, state, material_service, reset_progress=True)

    @router.callback_query(F.data == "test:menu")
    async def test_menu_back_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await show_main_menu(call)
            await state.set_state(StudyState.in_lesson)

    @router.callback_query(F.data == "test:practice")
    async def test_practice_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await open_practice(call, state, material_service, show_solution=False)

    @router.callback_query(F.data == "practice:show")
    async def practice_show_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await open_practice(call, state, material_service, show_solution=True)

    @router.callback_query(F.data == "practice:back")
    async def practice_back_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await render_lesson(call, state, material_service)

    @router.callback_query(F.data == "practice:menu")
    async def practice_menu_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await show_main_menu(call)
            await state.set_state(StudyState.in_lesson)

    @router.callback_query(F.data == "test:back")
    async def test_back_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await render_lesson(call, state, material_service)

    @router.callback_query(F.data.startswith("test:answer:"))
    async def answer_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            payload = await _ensure_material(call, state, material_service)
            if payload is None:
                await call.answer()
                return

            material, topic = payload
            parsed = _parse_answer_payload(call.data)
            if parsed is None:
                await call.answer()
                return

            idx, selected = parsed
            data = await state.get_data()
            tests, current_index, score, review_mode = _get_active_tests(data, material)
            accepting = bool(data.get("accepting_answer", False))
            if not accepting or idx != current_index:
                await call.answer()
                return

            if idx < 0 or idx >= len(tests):
                await call.answer()
                return

            current_test = tests[idx] if isinstance(tests[idx], dict) else {}
            correct = _extract_correct_letter(current_test)
            if correct is None:
                logger.warning("Test item has invalid correct answer", extra={"test_index": idx, "topic": topic})
                if review_mode:
                    await state.update_data(review_index=idx + 1, accepting_answer=False)
                else:
                    await state.update_data(test_index=idx + 1, accepting_answer=False)
                    await maybe_save_resume_state(
                        state,
                        material_service,
                        user_id=call.from_user.id,
                        topic=topic,
                        stage="test",
                        card_index=int(data.get("card_index", 0)),
                        test_index=idx + 1,
                        test_score=int(data.get("test_score", 0)),
                    )
                await call.answer()
                await _render_question(call, state, material_service)
                return

            is_correct = selected == correct
            if is_correct:
                score += 1

            explanation = str(current_test.get("explanation") or "\u2014")
            text = format_test_result(is_correct, explanation, correct_letter=None if is_correct else correct)

            if review_mode:
                await state.update_data(review_index=idx + 1, review_score=score, accepting_answer=False)
            else:
                if not is_correct:
                    wrong_tests = _get_review_tests(data)
                    question = str(current_test.get("question") or "\u2014")
                    options = _normalize_options(current_test.get("options"))
                    wrong_tests = _append_wrong_test(
                        wrong_tests=wrong_tests,
                        question=question,
                        options=options,
                        correct=correct,
                        explanation=explanation,
                        max_len=len(tests),
                    )
                    await state.update_data(wrong_tests=wrong_tests)

                await state.update_data(test_index=idx + 1, test_score=score, accepting_answer=False)
                await maybe_save_resume_state(
                    state,
                    material_service,
                    user_id=call.from_user.id,
                    topic=topic,
                    stage="test",
                    card_index=int(data.get("card_index", 0)),
                    test_index=idx + 1,
                    test_score=score,
                )
                await material_service.record_activity(call.from_user.id, last_topic=topic, last_stage="test")
            await call.answer()
            await edit_or_send(call, text, reply_markup=create_test_result_keyboard())

    return router
