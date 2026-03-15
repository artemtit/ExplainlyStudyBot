from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher


def _register_handlers(dispatcher: Dispatcher) -> None:
    try:
        from handlers.start import router as start_router
    except Exception:  # pragma: no cover - safe fallback for early bootstrap
        logging.getLogger(__name__).exception("Failed to import handlers.start")
        return
    dispatcher.include_router(start_router)


async def main() -> None:
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN is not set")

    bot = Bot(token=token)
    dispatcher = Dispatcher()
    _register_handlers(dispatcher)
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
