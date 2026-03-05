from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress

from aiohttp import web
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types.error_event import ErrorEvent

from bot.config import configure_logging, load_settings
from bot.handlers.cards import build_router as build_cards_router
from bot.handlers.lesson import build_router as build_lesson_router
from bot.handlers.practice import build_router as build_practice_router
from bot.handlers.start import build_router as build_start_router
from bot.handlers.study import build_router as build_study_router
from bot.handlers.tests import build_router as build_tests_router
from bot.services.material_service import MaterialService
from bot.services.openai_service import OpenAIService
from bot.services.supabase_service import SupabaseService
from bot.utils.locks import UserLockManager

logger = logging.getLogger(__name__)


async def _run_health_server() -> None:
    app = web.Application()

    async def health(_: web.Request) -> web.Response:
        return web.Response(text="ok")

    app.add_routes([web.get("/", health), web.get("/health", health)])
    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.getenv("PORT", os.getenv("RENDER_PORT", "8000")))
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    logger.info("Health server started on port %s", port)

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        with suppress(Exception):
            await runner.cleanup()


def _build_dispatcher(material_service: MaterialService, lock_manager: UserLockManager, free_tier_notice: bool) -> Dispatcher:
    dp = Dispatcher(storage=MemoryStorage())

    dp.include_router(build_start_router(material_service, lock_manager))
    dp.include_router(build_study_router(material_service, lock_manager, free_tier_notice))
    dp.include_router(build_lesson_router(lock_manager))
    dp.include_router(build_cards_router(lock_manager))
    dp.include_router(build_tests_router(lock_manager))
    dp.include_router(build_practice_router(lock_manager))

    @dp.error()
    async def on_error(event: ErrorEvent) -> None:
        logger.exception("Unhandled update error", exc_info=event.exception)
        update = event.update
        message = update.message if update else None
        callback = update.callback_query if update else None
        try:
            if callback and callback.message:
                await callback.message.answer("Unexpected error occurred. Please try again.")
            elif message:
                await message.answer("Unexpected error occurred. Please try again.")
        except Exception:
            logger.exception("Failed to send error message to user")

    return dp


async def run_async() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)

    openai_service = OpenAIService(
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_model=settings.openrouter_model,
        groq_api_key=settings.groq_api_key,
        groq_model=settings.groq_model,
        timeout_seconds=settings.generation_timeout_seconds,
    )
    supabase_service = SupabaseService(settings.supabase_url, settings.supabase_key)
    material_service = MaterialService(
        llm_service=openai_service,
        supabase_service=supabase_service,
        cache_ttl_seconds=settings.material_cache_ttl_seconds,
    )
    lock_manager = UserLockManager()

    bot = Bot(token=settings.telegram_token, default=DefaultBotProperties())
    dp = _build_dispatcher(material_service, lock_manager, settings.free_tier_notice)

    health_task = asyncio.create_task(_run_health_server())
    try:
        await dp.start_polling(bot, handle_signals=True)
    finally:
        health_task.cancel()
        with suppress(Exception):
            await health_task
        with suppress(Exception):
            await bot.session.close()


def run() -> None:
    asyncio.run(run_async())
