from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from ai.llm_client import LlmClient
from database.repository import Repository
from services.lesson_service import LessonService
from utils.formatting import format_lesson_text
from utils.validation import validate_topic

router = Router(name="lesson")

_llm_client = LlmClient()
_repository = Repository()
_lesson_service = LessonService(_llm_client, repository=_repository)


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
    await message.answer(format_lesson_text(result.topic, result.explanation))
