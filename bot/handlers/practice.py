from __future__ import annotations

from dataclasses import dataclass

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import Message

from services.question_service import PracticeQuestion
from services.storage import history_service, question_service, session_service
from utils.formatting import format_question_text
from utils.validation import validate_topic

router = Router(name="practice")


@dataclass
class PracticeSession:
    questions: list[PracticeQuestion]
    index: int = 0


_sessions: dict[int, PracticeSession] = {}


async def _send_question(message: Message, session: PracticeSession) -> None:
    question = session.questions[session.index]
    await message.answer(
        format_question_text(
            question.question,
            index=session.index + 1,
            total=len(session.questions),
        ),
        parse_mode="Markdown",
    )


@router.message(Command("practice"))
async def practice_start_handler(message: Message) -> None:
    topic = message.get_args().strip()
    if not topic:
        await message.answer("Укажи тему после команды, например: /practice Дроби")
        return
    validated = validate_topic(topic)
    if not validated:
        await message.answer("Тема слишком короткая или длинная. Попробуй еще раз.")
        return
    questions = await question_service.generate_questions(validated)
    session = PracticeSession(questions=questions)
    _sessions[message.from_user.id] = session
    session_service.set_last_topic(user_id=message.from_user.id, topic=validated, mode="practice")
    await _send_question(message, session)


@router.message(Command("hint"))
async def hint_handler(message: Message) -> None:
    session = _sessions.get(message.from_user.id)
    if not session:
        await message.answer("Сначала начни практику: /practice <тема>")
        return
    current = session.questions[session.index]
    hint = await question_service.generate_hint(question=current)
    await message.answer(f"Подсказка: {hint}")


@router.message(Command("exam"))
async def exam_start_handler(message: Message) -> None:
    topic = message.get_args().strip()
    if not topic:
        await message.answer("Укажи тему после команды, например: /exam Фотосинтез")
        return
    validated = validate_topic(topic)
    if not validated:
        await message.answer("Тема слишком короткая или длинная. Попробуй еще раз.")
        return
    questions = await question_service.generate_exam_questions(validated)
    session = PracticeSession(questions=questions)
    _sessions[message.from_user.id] = session
    session_service.set_last_topic(user_id=message.from_user.id, topic=validated, mode="exam")
    await _send_question(message, session)


@router.message(F.text)
async def practice_answer_handler(message: Message) -> None:
    if message.text.startswith("/"):
        return
    session = _sessions.get(message.from_user.id)
    if not session:
        return
    current = session.questions[session.index]
    is_correct = await question_service.validate_answer(
        question=current,
        user_answer=message.text,
    )
    if is_correct:
        await message.answer("Верно! Отличная работа.")
    else:
        answer = current.answer or "Нет эталонного ответа."
        feedback = await question_service.get_feedback(question=current, user_answer=message.text)
        await message.answer(f"Похоже, есть неточность.\n{feedback}\nПример ответа: {answer}")
    await history_service.store_answer(
        user_id=message.from_user.id,
        question=current.question,
        answer=message.text,
        is_correct=is_correct,
    )

    session.index += 1
    if session.index >= len(session.questions):
        _sessions.pop(message.from_user.id, None)
        await message.answer("Практика завершена. Можешь начать новую тему.")
        return
    await _send_question(message, session)
