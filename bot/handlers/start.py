from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from bot.learning_engine.engine import LearningEngine
from bot.ui.formatting import SEPARATOR, format_progress
from bot.ui.keyboards import BTN_HELP, BTN_SUPPORT, create_main_menu, create_progress_keyboard
from bot.utils.locks import UserLockManager

logger = logging.getLogger(__name__)

WELCOME_TEXT = (
    f"{SEPARATOR}\n"
    "\U0001F44B \u041F\u0440\u0438\u0432\u0435\u0442! \u042F ExplainlyStudy.\n"
    f"{SEPARATOR}\n\n"
    "\u0412\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u0440\u0435\u0436\u0438\u043C \u043E\u0431\u0443\u0447\u0435\u043D\u0438\u044F."
)
CANCEL_TEXT = (
    f"{SEPARATOR}\n"
    "\u2705 \u0414\u0435\u0439\u0441\u0442\u0432\u0438\u0435 \u043E\u0442\u043C\u0435\u043D\u0435\u043D\u043E\n"
    f"{SEPARATOR}\n\n"
    "\u0412\u043E\u0437\u0432\u0440\u0430\u0449\u0430\u044E \u0432 \u043C\u0435\u043D\u044E."
)
HELP_TEXT = (
    f"{SEPARATOR}\n"
    "\u2753 \u041F\u043E\u043C\u043E\u0449\u044C\n"
    f"{SEPARATOR}\n\n"
    "\u041A\u043E\u0440\u043E\u0442\u043A\u043E \u043E \u0432\u043E\u0437\u043C\u043E\u0436\u043D\u043E\u0441\u0442\u044F\u0445:\n"
    "\u2022 \u041D\u0430\u043F\u0438\u0448\u0438 \u0442\u0435\u043C\u0443 \u2014 \u043F\u043E\u043B\u0443\u0447\u0438\u0448\u044C \u0443\u0440\u043E\u043A, \u043A\u0430\u0440\u0442\u043E\u0447\u043A\u0438 \u0438 \u0442\u0435\u0441\u0442.\n"
    "\u2022 \u0412\u044B\u0431\u0438\u0440\u0430\u0439 \u0444\u043E\u0440\u043C\u0430\u0442 \u043E\u0431\u0443\u0447\u0435\u043D\u0438\u044F \u0432 \u043C\u0435\u043D\u044E.\n"
    "\u2022 \u041F\u0440\u043E\u0434\u043E\u043B\u0436\u0438 \u0442\u0435\u043C\u0443 \u0447\u0435\u0440\u0435\u0437 \u043A\u043D\u043E\u043F\u043A\u0443 \u00AB\u041F\u0440\u043E\u0434\u043E\u043B\u0436\u0438\u0442\u044C\u00BB.\n\n"
    "\u041A\u043E\u043C\u0430\u043D\u0434\u044B:\n"
    "\u2022 /menu \u2014 \u0432\u0435\u0440\u043D\u0443\u0442\u044C\u0441\u044F \u0432 \u0433\u043B\u0430\u0432\u043D\u043E\u0435 \u043C\u0435\u043D\u044E.\n\u2022 /home \u2014 \u0442\u043e \u0436\u0435, \u0447\u0442\u043e /menu.\n\u2022 /topic \u2014 \u043d\u043e\u0432\u0430\u044f \u0442\u0435\u043c\u0430.\n\u2022 /restart \u2014 \u043f\u0435\u0440\u0435\u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0431\u043e\u0442.\n\u2022 /settings \u2014 \u043d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438.\n\u2022 /cards \u2014 \u0444\u043b\u044d\u0448\u043a\u0430\u0440\u0442\u044b.\n"
    "\u2022 /cancel \u2014 \u043E\u0442\u043C\u0435\u043D\u0438\u0442\u044C \u0442\u0435\u043A\u0443\u0449\u0435\u0435 \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0435.\n\n"
    "\u2022 /support \u2014 \u0441\u0432\u044F\u0437\u0430\u0442\u044C\u0441\u044F \u0441 \u043F\u043E\u0434\u0434\u0435\u0440\u0436\u043A\u043E\u0439.\n\n"
    "\u2022 /progress \u2014 \u043F\u043E\u0441\u043C\u043E\u0442\u0440\u0435\u0442\u044C \u043F\u0440\u043E\u0433\u0440\u0435\u0441\u0441.\n\n\u2022 /stats \u2014 \u0442\u043e \u0436\u0435, \u0447\u0442\u043e /progress.\n\n\u2022 /about \u2014 \u043e \u0431 \u043e \u0442 \u0431 \u043e \u0442 \u0435.\n\n\u2022 /feedback \u2014 \u043e\u0431\u0440\u0430\u0442\u043d\u0430\u044f \u0441\u0432\u044f\u0437\u044c.\n\n\u2022 /report \u2014 \u0441\u043e\u043e\u0431\u0449\u0438\u0442\u044c \u043e \u043f\u0440\u043e\u0431\u043b\u0435\u043c\u0435.\n\n"
    "\u0415\u0441\u043B\u0438 \u0447\u0442\u043E-\u0442\u043E \u043D\u0435 \u0440\u0430\u0431\u043E\u0442\u0430\u0435\u0442, \u043D\u0430\u043F\u0438\u0448\u0438 \u0432 \u043F\u043E\u0434\u0434\u0435\u0440\u0436\u043A\u0443."
)
SUPPORT_TEXT = (
    f"{SEPARATOR}\n"
    "\U0001F198 \u041F\u043E\u0434\u0434\u0435\u0440\u0436\u043A\u0430\n"
    f"{SEPARATOR}\n\n"
    "\u041D\u0430\u043F\u0438\u0448\u0438 \u043D\u0430\u043C \u0432 \u043F\u043E\u0434\u0434\u0435\u0440\u0436\u043A\u0443."
)
SUPPORT_MISSING_TEXT = (
    f"{SEPARATOR}\n"
    "\U0001F198 \u041F\u043E\u0434\u0434\u0435\u0440\u0436\u043A\u0430\n"
    f"{SEPARATOR}\n\n"
    "\u041A\u0430\u043D\u0430\u043B \u043F\u043E\u0434\u0434\u0435\u0440\u0436\u043A\u0438 \u043F\u043E\u043A\u0430 \u043D\u0435 \u043D\u0430\u0441\u0442\u0440\u043E\u0435\u043D."
)
FEEDBACK_TEXT = (
    f"{SEPARATOR}\n"
    "\u270D \u041E\u0442\u0437\u044B\u0432\n"
    f"{SEPARATOR}\n\n"
    "\u041D\u0430\u043F\u0438\u0448\u0438\u0442\u0435 \u043E\u0442\u0437\u044B\u0432 \u0432 \u043F\u043E\u0434\u0434\u0435\u0440\u0436\u043A\u0443 \u2014 \u043C\u044B \u0432\u0441\u0435 \u0447\u0438\u0442\u0430\u0435\u043C."
)
ABOUT_TEXT = (
    f"{SEPARATOR}\n"
    "\u2139 \u041E \u0431\u043E\u0442\u0435\n"
    f"{SEPARATOR}\n\n"
    "ExplainlyStudy \u2014 \u043F\u043E\u043C\u043E\u0449\u043D\u0438\u043A \u0434\u043B\u044F \u0443\u0447\u0451\u0431\u044B: \u0443\u0440\u043E\u043A\u0438, \u0444\u043B\u044D\u0448\u043A\u0430\u0440\u0442\u044B, \u0442\u0435\u0441\u0442\u044B \u0438 \u043F\u0440\u0430\u043A\u0442\u0438\u043A\u0430 \u043F\u043E \u043B\u044E\u0431\u043E\u0439 \u0442\u0435\u043C\u0435."
)
UNKNOWN_COMMAND_TEXT = (
    f"{SEPARATOR}\n"
    "\u26A0 \u041D\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043D\u0430\u044F \u043A\u043E\u043C\u0430\u043D\u0434\u0430\n"
    f"{SEPARATOR}\n\n"
    "\u0414\u043E\u0441\u0442\u0443\u043F\u043D\u043E: /start, /help, /menu, /home, /topic, /study, /restart, /settings, /cards, /flashcards, /tests, /test, /practice, /lesson, /continue, /profile, /cancel, "
    "/support, /progress, /stats, /about, /feedback, /report"
)


async def show_main_menu(target: Message | CallbackQuery, *, text: str | None = None) -> None:
    message = target.message if isinstance(target, CallbackQuery) else target
    await message.answer(text or WELCOME_TEXT, reply_markup=create_main_menu())
    if isinstance(target, CallbackQuery):
        await target.answer()


def _build_help_keyboard(support_url: str | None) -> InlineKeyboardMarkup | None:
    if not support_url:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="\U0001F198 \u041F\u043E\u0434\u0434\u0435\u0440\u0436\u043A\u0430", url=support_url)]]
    )


def build_router(material_service: LearningEngine, lock_manager: UserLockManager, support_url: str | None) -> Router:
    router = Router(name="start")

    @router.message(CommandStart())
    async def start_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await material_service.ensure_user(message.from_user.id, message.from_user.username)
            await state.clear()
            await show_main_menu(message)

    @router.message(Command("cancel"))
    async def cancel_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await state.clear()
            await show_main_menu(message, text=CANCEL_TEXT)

    @router.message(Command("menu"))
    async def menu_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await state.clear()
            await show_main_menu(message)

    @router.message(Command("home"))
    async def home_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await state.clear()
            await show_main_menu(message)

    @router.message(Command("restart"))
    async def restart_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await state.clear()
            await show_main_menu(message)

    @router.message(Command("help"))
    async def help_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await state.clear()
            await message.answer(HELP_TEXT, reply_markup=_build_help_keyboard(support_url))
            await show_main_menu(message)

    @router.message(F.text == BTN_HELP)
    async def help_button_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await state.clear()
            await message.answer(HELP_TEXT, reply_markup=_build_help_keyboard(support_url))
            await show_main_menu(message)

    @router.message(Command("support"))
    async def support_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            if support_url:
                await message.answer(SUPPORT_TEXT, reply_markup=_build_help_keyboard(support_url))
            else:
                await message.answer(SUPPORT_MISSING_TEXT)

    @router.message(Command("feedback"))
    async def feedback_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await message.answer(FEEDBACK_TEXT, reply_markup=_build_help_keyboard(support_url))

    @router.message(Command("report"))
    async def report_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await message.answer(FEEDBACK_TEXT, reply_markup=_build_help_keyboard(support_url))

    @router.message(Command("about"))
    async def about_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await message.answer(ABOUT_TEXT)

    @router.message(F.text == BTN_SUPPORT)
    async def support_button_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            if support_url:
                await message.answer(SUPPORT_TEXT, reply_markup=_build_help_keyboard(support_url))
            else:
                await message.answer(SUPPORT_MISSING_TEXT)

    @router.message(Command("progress"))
    async def progress_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            stats = await material_service.get_user_stats(message.from_user.id)
            resume = await material_service.load_resume_state(message.from_user.id)
            await message.answer(
                format_progress(stats),
                reply_markup=create_progress_keyboard(can_continue=bool(resume)),
            )

    @router.message(Command("stats"))
    async def stats_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            stats = await material_service.get_user_stats(message.from_user.id)
            resume = await material_service.load_resume_state(message.from_user.id)
            await message.answer(
                format_progress(stats),
                reply_markup=create_progress_keyboard(can_continue=bool(resume)),
            )

    @router.message(F.text.startswith("/"))
    async def unknown_command_handler(message: Message, state: FSMContext) -> None:
        async with await lock_manager.get(message.from_user.id):
            await message.answer(UNKNOWN_COMMAND_TEXT)
            await show_main_menu(message)

    return router
















