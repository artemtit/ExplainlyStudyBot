from __future__ import annotations

import logging
from contextlib import suppress

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from bot.core.errors import LessonResponseInvalidError
from bot.handlers.start import show_main_menu
from bot.lesson_engine import LessonEngine
from bot.learning_engine.engine import LearningEngine
from bot.states.tutor_state import TutorState
from bot.ui.keyboards import (
    BTN_TUTOR,
    create_similar_problem_keyboard,
    create_tutor_mode_keyboard,
)
from bot.utils.locks import UserLockManager
from bot.utils.strings import GENERATION_TEXT, INVALID_TUTOR_TEXT, TEMP_UNAVAILABLE_TEXT
from bot.utils.topic_utils import validate_topic
from bot.utils.lesson_renderer import render_solution_card
logger = logging.getLogger(__name__)


async def _ask_for_problem(target: Message | CallbackQuery) -> None:
    message = target.message if isinstance(target, CallbackQuery) else target
    await message.answer("Отправьте задачу одним сообщением.")


async def _send_learning_question(target: Message, step: dict) -> None:
    question = str(step.get("question") or "\u2014")
    step_no = step.get("step") or ""
    text = f"\u0428\u0430\u0433 {step_no}: {question}" if step_no else question
    await target.answer(text)


def _extract_steps(data: dict) -> list[dict]:
    lesson_state = data.get("lesson_state") if isinstance(data.get("lesson_state"), dict) else {}
    steps = lesson_state.get("steps") if isinstance(lesson_state.get("steps"), list) else []
    return [step for step in steps if isinstance(step, dict)]


def build_router(
    learning_engine: LearningEngine,
    lesson_engine: LessonEngine,
    lock_manager: UserLockManager,
) -> Router:
    router = Router(name="tutor")

    @router.message(F.text == BTN_TUTOR)
    async def tutor_entry_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await state.set_state(TutorState.awaiting_problem)
            await _ask_for_problem(message)

    @router.message(StateFilter(TutorState.awaiting_problem), F.text)
    async def tutor_problem_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            text = message.text or ""
            if text.startswith("/"):
                return
            problem = validate_topic(text, min_len=4, max_len=800)
            if not problem:
                await message.answer("Пожалуйста, отправьте задачу одной фразой.")
                return

            await state.update_data(
                mode=None,
                current_problem=problem,
                current_step=0,
                lesson_state=None,
                current_attempts=0,
            )
            await state.set_state(TutorState.awaiting_mode)
            await message.answer("\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0440\u0435\u0436\u0438\u043c:", reply_markup=create_tutor_mode_keyboard())

    @router.callback_query(StateFilter(TutorState.awaiting_mode), F.data.startswith("tutor_mode:"))
    async def tutor_mode_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            data = await state.get_data()
            problem = str(data.get("current_problem") or "").strip()
            if not problem:
                await call.answer()
                await state.set_state(TutorState.awaiting_problem)
                await _ask_for_problem(call)
                return

            mode_raw = (call.data or "").split(":", 1)[-1]
            mode = "learning" if mode_raw == "learning" else "solution"
            await call.answer()

            status_target = call.message if call.message else call
            gen_msg = await status_target.answer(GENERATION_TEXT)

            try:
                if mode == "learning":
                    plan = await lesson_engine.generate_steps(problem)
                else:
                    plan = await lesson_engine.generate_solution(problem)
            except LessonResponseInvalidError:
                with suppress(Exception):
                    await gen_msg.delete()
                await status_target.answer(INVALID_TUTOR_TEXT)
                return
            except Exception:
                logger.exception("Tutor generation failed", extra={"mode": mode})
                with suppress(Exception):
                    await gen_msg.delete()
                await status_target.answer(TEMP_UNAVAILABLE_TEXT)
                return
            finally:
                with suppress(Exception):
                    await gen_msg.delete()

            if mode == "learning":
                steps_payload = [step.to_dict() for step in plan.steps]
                await state.update_data(
                    mode="learning",
                    lesson_state={"topic": plan.topic, "steps": steps_payload},
                    current_step=0,
                    current_attempts=0,
                )
                await state.set_state(TutorState.learning_in_progress)
                await learning_engine.save_lesson_progress(
                    call.from_user.id,
                    problem=problem,
                    current_step=0,
                    completed=False,
                )
                if call.message:
                    await _send_learning_question(call.message, steps_payload[0])
                return

            steps_payload = [step.to_dict() for step in plan.steps]
            await state.update_data(
                mode="solution",
                lesson_state={"topic": plan.topic, "steps": steps_payload, "final_answer": plan.final_answer},
                current_step=len(steps_payload),
            )
            await state.set_state(TutorState.solution_in_progress)

            try:
                image_bytes = render_solution_card(topic=plan.topic, problem=problem, steps=steps_payload)
            except Exception:
                logger.exception("Failed to render solution card", extra={"topic": plan.topic})
                await status_target.answer(TEMP_UNAVAILABLE_TEXT)
                return

            await status_target.answer_photo(
                photo=BufferedInputFile(image_bytes, filename="solution.png"),
            )
            await status_target.answer(f"\u0424\u0438\u043d\u0430\u043b\u044c\u043d\u044b\u0439 \u043e\u0442\u0432\u0435\u0442: {plan.final_answer}")
            await status_target.answer(
                "\u0425\u043e\u0447\u0435\u0448\u044c \u043f\u043e\u043f\u0440\u043e\u0431\u043e\u0432\u0430\u0442\u044c \u0440\u0435\u0448\u0438\u0442\u044c \u043f\u043e\u0445\u043e\u0436\u0443\u044e \u0437\u0430\u0434\u0430\u0447\u0443?",
                reply_markup=create_similar_problem_keyboard(),
            )

            await learning_engine.save_lesson_progress(
                call.from_user.id,
                problem=problem,
                current_step=len(steps_payload),
                completed=True,
            )
            await learning_engine.save_request(
                call.from_user.id,
                plan.topic,
                mode="solution",
                steps_count=len(steps_payload),
                success=True,
            )

    @router.message(StateFilter(TutorState.learning_in_progress), F.text)
    async def tutor_learning_answer_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            data = await state.get_data()
            problem = str(data.get("current_problem") or "").strip()
            steps = _extract_steps(data)
            if not steps:
                await message.answer(INVALID_TUTOR_TEXT)
                await state.set_state(TutorState.awaiting_problem)
                return

            current_step = int(data.get("current_step", 0))
            attempts = int(data.get("current_attempts", 0))

            if current_step >= len(steps):
                await message.answer("\u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u043e.")
                await state.set_state(TutorState.awaiting_problem)
                return

            step = steps[current_step]
            user_answer = message.text or ""
            correct = str(step.get("correct_answer") or "")

            if lesson_engine.check_answer(user_answer, correct):
                await message.answer("\u0412\u0435\u0440\u043d\u043e! \u0414\u0432\u0438\u0433\u0430\u0435\u043c\u0441\u044f \u0434\u0430\u043b\u044c\u0448\u0435.")
                current_step += 1
                attempts = 0
            else:
                attempts += 1
                if attempts == 1:
                    await message.answer("\u041f\u043e\u043f\u0440\u043e\u0431\u0443\u0439 \u0435\u0449\u0451 \u0440\u0430\u0437.")
                    await state.update_data(current_attempts=attempts)
                    return
                if attempts == 2:
                    hint = str(step.get("hint") or "")
                    await message.answer(f"\u041f\u043e\u0434\u0441\u043a\u0430\u0437\u043a\u0430: {hint}")
                    await state.update_data(current_attempts=attempts)
                    return

                explanation = str(step.get("explanation") or "")
                await message.answer(f"\u041e\u0431\u044a\u044f\u0441\u043d\u0435\u043d\u0438\u0435: {explanation}")
                current_step += 1
                attempts = 0

            if current_step >= len(steps):
                await message.answer("\u041e\u0442\u043b\u0438\u0447\u043d\u043e, \u0432\u0441\u0435 \u0448\u0430\u0433\u0438 \u043f\u0440\u043e\u0439\u0434\u0435\u043d\u044b!")
                topic = str((data.get("lesson_state") or {}).get("topic") or problem)
                await learning_engine.save_lesson_progress(
                    message.from_user.id,
                    problem=problem,
                    current_step=len(steps),
                    completed=True,
                )
                await learning_engine.save_request(
                    message.from_user.id,
                    topic,
                    mode="learning",
                    steps_count=len(steps),
                    success=True,
                )
                await state.set_state(TutorState.awaiting_problem)
                await _ask_for_problem(message)
                return

            await state.update_data(current_step=current_step, current_attempts=attempts)
            await learning_engine.save_lesson_progress(
                message.from_user.id,
                problem=problem,
                current_step=current_step,
                completed=False,
            )
            await _send_learning_question(message, steps[current_step])

    @router.callback_query(StateFilter(TutorState.solution_in_progress), F.data.startswith("tutor_similar:"))
    async def tutor_similar_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            choice = (call.data or "").split(":", 1)[-1]
            if choice == "yes":
                await state.set_state(TutorState.awaiting_problem)
                await _ask_for_problem(call)
                return
            await show_main_menu(call)
            await state.clear()

    return router
