from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from ai.llm_client import LlmClient
from database.repository import Repository
from services.lesson_service import LessonService
from services.question_service import QuestionService
from utils.formatting import format_lesson_text, format_questions_block
from utils.validation import validate_topic

router = Router(name="lesson")

_llm_client = LlmClient()
_repository = Repository()
_lesson_service = LessonService(_llm_client, repository=_repository)
_question_service = QuestionService(_llm_client)


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
    result = await _lesson_service.explain(validated, user_id=message.from_user.id)
    lesson_text = format_lesson_text(result.topic, result.explanation, examples=result.examples)
    questions = await _question_service.generate_questions(result.topic, count=3)
    questions_block = format_questions_block([q.question for q in questions])
    text = lesson_text if not questions_block else f"{lesson_text}\n\n{questions_block}"
    await message.answer(text, parse_mode="Markdown")
