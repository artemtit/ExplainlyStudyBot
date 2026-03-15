from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress

from aiohttp import web
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

try:
    from aiogram.fsm.storage.redis import RedisStorage
except Exception:
    RedisStorage = None
from aiogram.types import BotCommand
from aiogram.types.error_event import ErrorEvent

from bot.config import configure_logging, load_settings
from bot.handlers.flashcards import build_router as build_flashcards_router
from bot.handlers.profile import build_router as build_profile_router
from bot.handlers.progress import build_router as build_progress_router
from bot.handlers.settings import build_router as build_settings_router
from bot.handlers.start import build_router as build_start_router
from bot.handlers.study import build_router as build_study_router
from bot.handlers.tests import build_router as build_tests_router
from bot.ai.content_generator import AiContentGenerator
from bot.ai.llm_client import OpenAIService
from bot.infrastructure.redis_cache import RedisMaterialCache
from bot.infrastructure.material_repository import SupabaseMaterialRepository
from bot.infrastructure.progress_repository import SupabaseProgressRepository
from bot.infrastructure.stats_repository import SupabaseStatsRepository
from bot.infrastructure.supabase_client import create_supabase_client
from bot.infrastructure.user_repository import SupabaseUserRepository
from bot.learning_engine.engine import LearningEngine
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


def _build_storage():
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url and RedisStorage is not None:
        try:
            return RedisStorage.from_url(redis_url)
        except Exception:
            logger.exception("Failed to initialize RedisStorage, falling back to MemoryStorage")
    return MemoryStorage()


def _build_dispatcher(
    learning_engine: LearningEngine,
    lock_manager: UserLockManager,
    free_tier_notice: bool,
    support_url: str | None,
) -> Dispatcher:
    dp = Dispatcher(storage=_build_storage())

    dp.include_router(build_start_router(learning_engine, lock_manager, support_url))
    dp.include_router(build_study_router(learning_engine, lock_manager, free_tier_notice))
    dp.include_router(build_flashcards_router(learning_engine, lock_manager))
    dp.include_router(build_tests_router(learning_engine, lock_manager))
    dp.include_router(build_progress_router(learning_engine, lock_manager))
    dp.include_router(build_profile_router(learning_engine, lock_manager, support_url))
    dp.include_router(build_settings_router(learning_engine, lock_manager, support_url))

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
        groq_api_key=settings.groq_api_key,
        groq_model=settings.groq_model,
        timeout_seconds=settings.generation_timeout_seconds,
    )
    supabase_client = create_supabase_client(settings.supabase_url, settings.supabase_key)
    user_repo = SupabaseUserRepository(supabase_client)
    material_repo = SupabaseMaterialRepository(supabase_client)
    progress_repo = SupabaseProgressRepository(supabase_client)
    stats_repo = SupabaseStatsRepository(supabase_client)
    content_generator = AiContentGenerator(openai_service)
    cache = RedisMaterialCache.from_env(ttl_seconds=settings.material_cache_ttl_seconds)
    learning_engine = LearningEngine(
        content_generator=content_generator,
        material_repo=material_repo,
        user_repo=user_repo,
        progress_repo=progress_repo,
        stats_repo=stats_repo,
        cache=cache,
    )
    lock_manager = UserLockManager()

    bot = Bot(token=settings.telegram_token, default=DefaultBotProperties())
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Запуск и главное меню"),
                BotCommand(command="help", description="Справка по боту"),
                BotCommand(command="menu", description="Главное меню"),
                BotCommand(command="settings", description="Настройки"),
                BotCommand(command="reset", description="Сбросить прогресс"),
                BotCommand(command="notifications", description="Уведомления"),
                BotCommand(command="home", description="Главное меню (алиас)"),
                BotCommand(command="restart", description="Перезапуск бота"),
                BotCommand(command="topic", description="Новая тема"),
                BotCommand(command="recent", description="Недавние темы"),
                BotCommand(command="study", description="Начать обучение"),
                BotCommand(command="learn", description="Начать обучение (алиас)"),
                BotCommand(command="cards", description="Флэшкарты"),
                BotCommand(command="flashcards", description="Флэшкарты (алиас)"),
                BotCommand(command="tests", description="Тесты"),
                BotCommand(command="test", description="Тесты (алиас)"),
                BotCommand(command="practice", description="Практика"),
                BotCommand(command="lesson", description="Урок"),
                BotCommand(command="continue", description="Продолжить обучение"),
                BotCommand(command="last", description="Последняя тема"),
                BotCommand(command="profile", description="Профиль пользователя"),
                BotCommand(command="cancel", description="Отменить текущее действие"),
                BotCommand(command="support", description="Связаться с поддержкой"),
                BotCommand(command="progress", description="Показать прогресс"),
                BotCommand(command="stats", description="Статистика (то же, что прогресс)"),
                BotCommand(command="streak", description="Текущая серия"),
                BotCommand(command="about", description="О боте"),
                BotCommand(command="feedback", description="Отправить отзыв"),
                BotCommand(command="report", description="Сообщить о проблеме"),
            ]
        )
    except Exception:
        logger.exception("Failed to set bot commands")
    dp = _build_dispatcher(learning_engine, lock_manager, settings.free_tier_notice, settings.support_url)

    health_task = asyncio.create_task(_run_health_server())
    try:
        await dp.start_polling(bot, handle_signals=True)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutdown requested, stopping polling")
    finally:
        health_task.cancel()
        with suppress(Exception):
            await health_task
        with suppress(Exception):
            await bot.session.close()
        with suppress(Exception):
            await cache.close()


def run() -> None:
    try:
        asyncio.run(run_async())
    except KeyboardInterrupt:
        pass
