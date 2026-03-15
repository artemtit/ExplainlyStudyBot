from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher

from bot.middlewares.error_handling import ErrorHandlingMiddleware
from config import load_settings
from utils.logging import setup_logging

def _register_handlers(dispatcher: Dispatcher) -> None:
    try:
        from bot.handlers.start import router as start_router
        from bot.handlers.lesson import router as lesson_router
        from bot.handlers.practice import router as practice_router
    except Exception:  # pragma: no cover - safe fallback for early bootstrap
        logging.getLogger(__name__).exception("Failed to import handlers.start")
        return
    dispatcher.include_router(start_router)
    dispatcher.include_router(lesson_router)
    dispatcher.include_router(practice_router)


async def main() -> None:
    settings = load_settings()
    bot = Bot(token=settings.bot_token)
    dispatcher = Dispatcher()
    dispatcher.update.middleware(ErrorHandlingMiddleware())
    _register_handlers(dispatcher)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
