from __future__ import annotations

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

router = Router(name="start")


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    text = (
        "Привет! Я Explainly — AI tutor бот.\n\n"
        "Как это работает:\n"
        "1) Ты пишешь тему.\n"
        "2) Я объясняю простым языком.\n"
        "3) Потом даю вопросы для практики.\n\n"
        "Примеры тем:\n"
        "• Квадратные уравнения\n"
        "• Пифагорова теорема\n"
        "• Великая французская революция\n"
        "• Фотосинтез\n"
        "• Дроби\n\n"
        "Напиши тему, и начнем."
    )
    await message.answer(text)
