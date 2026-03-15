from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from services.storage import lesson_service, question_service, session_service, topic_suggestion_service
from utils.formatting import format_lesson_text, format_questions_block, format_suggestions_text
from utils.validation import validate_topic

router = Router(name="lesson")


@router.message(Command("lesson"))
async def lesson_handler(message: Message) -> None:
    topic = message.get_args().strip()
    if not topic:
        await message.answer("Укажи тему после команды, например: /lesson Пифагорова теорема")
        return
    validated = validate_topic(topic)
    if not validated:
        await message.answer("Тема слишком короткая или длинная. Попробуй еще раз.")
        return
    result = await lesson_service.explain(validated, user_id=message.from_user.id)
    lesson_text = format_lesson_text(result.topic, result.explanation, examples=result.examples)
    questions = await question_service.generate_questions(result.topic, count=3)
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
    lesson_text = format_lesson_text(result.topic, result.explanation, examples=result.examples)
    await message.answer(lesson_text, parse_mode="Markdown")


@router.message(Command("suggest"))
async def suggest_handler(message: Message) -> None:
    suggestions = topic_suggestion_service.suggest(limit=5)
    await message.answer(format_suggestions_text(suggestions), parse_mode="Markdown")
