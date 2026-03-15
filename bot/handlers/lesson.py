from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from services.storage import lesson_service, question_service, session_service, topic_suggestion_service
from utils.formatting import format_lesson_text, format_questions_block, format_suggestions_text
from services.storage import streak_service
from utils.validation import extract_difficulty_and_topic, validate_topic

router = Router(name="lesson")


@router.message(Command("lesson"))
async def lesson_handler(message: Message) -> None:
    args = message.get_args().strip()
    if not args:
        await message.answer("Укажи тему после команды, например: /lesson Пифагорова теорема")
        return
    difficulty, topic = extract_difficulty_and_topic(args)
    validated = validate_topic(topic)
    if not validated:
        await message.answer("Тема слишком короткая или длинная. Попробуй еще раз.")
        return
    difficulty_label = _difficulty_to_label(difficulty)
    result = await lesson_service.explain(
        validated,
        user_id=message.from_user.id,
        difficulty=difficulty_label,
    )
    lesson_text = format_lesson_text(
        result.topic,
        result.explanation,
        examples=result.examples,
        summary=result.summary,
    )
    questions = await question_service.generate_questions(
        result.topic,
        count=3,
        difficulty=difficulty_label,
    )
    questions_block = format_questions_block([q.question for q in questions])
    text = lesson_text if not questions_block else f"{lesson_text}\n\n{questions_block}"
    await message.answer(text, parse_mode="Markdown")


@router.message(Command("retry"))
async def retry_handler(message: Message) -> None:
    last_topic = session_service.get_last_topic(message.from_user.id)
    if not last_topic:
        await message.answer("Нет последней темы. Используй /lesson <тема>.")
        return
    result = await lesson_service.explain(last_topic, user_id=message.from_user.id)
    lesson_text = format_lesson_text(
        result.topic,
        result.explanation,
        examples=result.examples,
        summary=result.summary,
    )
    await message.answer(lesson_text, parse_mode="Markdown")


@router.message(Command("suggest"))
async def suggest_handler(message: Message) -> None:
    suggestions = topic_suggestion_service.suggest(limit=5)
    await message.answer(format_suggestions_text(suggestions), parse_mode="Markdown")


@router.message(Command("streak"))
async def streak_handler(message: Message) -> None:
    current = streak_service.get_streak(message.from_user.id)
    await message.answer(f"\U0001F525 Текущая серия: {current} дн.")


def _difficulty_to_label(level: str) -> str:
    mapping = {
        "easy": "\u043f\u0440\u043e\u0441\u0442\u0430\u044f",
        "normal": "\u0441\u0440\u0435\u0434\u043d\u044f\u044f",
        "hard": "\u0441\u043b\u043e\u0436\u043d\u0430\u044f",
    }
    return mapping.get(level, "\u0441\u0440\u0435\u0434\u043d\u044f\u044f")
