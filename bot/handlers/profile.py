from __future__ import annotations

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers.start import show_main_menu
from bot.learning_engine.engine import LearningEngine
from bot.ui.formatting import SEPARATOR
from bot.ui.keyboards import BTN_PROFILE, create_profile_keyboard
from bot.utils.locks import UserLockManager
from bot.utils.telegram_utils import edit_or_send

PROFILE_TEXT = (
    f"{SEPARATOR}\n\U0001F464 \u041F\u0440\u043E\u0444\u0438\u043B\u044C\n{SEPARATOR}\n\n"
    "ID: {user_id}\n"
    "Username: {username}\n\n"
    "\u041F\u043E\u0434\u043F\u0438\u0441\u043A\u0430: Free\n"
    "\u0422\u0435\u043C \u0438\u0437\u0443\u0447\u0435\u043D\u043E: {topics}\n"
    "\u0422\u0435\u0441\u0442\u043E\u0432 \u043F\u0440\u043E\u0439\u0434\u0435\u043D\u043E: {tests}\n"
    "\u0424\u043B\u044D\u0448\u043A\u0430\u0440\u0442 \u043F\u0440\u043E\u0441\u043C\u043E\u0442\u0440\u0435\u043D\u043E: {cards}\n"
    "\u0421\u0435\u0440\u0438\u044F: {streak}\n"
    "\u041F\u043E\u0441\u043B\u0435\u0434\u043D\u044F\u044F \u0442\u0435\u043C\u0430: {last_topic}"
)
SUBSCRIPTION_TEXT = (
    f"{SEPARATOR}\n\u2B50 \u041F\u043E\u0434\u043F\u0438\u0441\u043A\u0430\n{SEPARATOR}\n\n"
    "\u042D\u0442\u0430 \u0444\u0443\u043D\u043A\u0446\u0438\u044F \u043F\u043E\u043A\u0430 \u0432 \u0440\u0430\u0437\u0440\u0430\u0431\u043E\u0442\u043A\u0435."
)


def _format_profile(user_id: int, username: str | None, stats: dict) -> str:
    topics = stats.get("topics_learned", 0)
    tests = stats.get("tests_passed", 0)
    cards = stats.get("flashcards_reviewed", 0)
    streak = stats.get("daily_streak", 0)
    last_topic = stats.get("last_topic") or "\u2014"
    return PROFILE_TEXT.format(
        user_id=user_id,
        username=username or "\u2014",
        topics=topics,
        tests=tests,
        cards=cards,
        streak=streak,
        last_topic=last_topic,
    )


async def _open_profile(target: Message | CallbackQuery, service: LearningEngine) -> None:
    user = target.from_user
    stats = await service.get_user_stats(user.id)
    text = _format_profile(user.id, user.username, stats)
    if isinstance(target, CallbackQuery):
        await edit_or_send(target, text, reply_markup=create_profile_keyboard())
    else:
        await target.answer(text, reply_markup=create_profile_keyboard())


def build_router(material_service: LearningEngine, lock_manager: UserLockManager) -> Router:
    router = Router(name="profile")

    @router.message(F.text == BTN_PROFILE)
    async def profile_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await _open_profile(message, material_service)

    @router.callback_query(F.data == "profile:back")
    async def profile_back_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await show_main_menu(call)

    @router.callback_query(F.data == "profile:subscription")
    async def profile_subscription_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await edit_or_send(call, SUBSCRIPTION_TEXT, reply_markup=create_profile_keyboard())

    return router
