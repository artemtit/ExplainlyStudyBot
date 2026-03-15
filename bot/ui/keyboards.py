from __future__ import annotations

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

BTN_START_LEARNING = "\U0001F4DA \u041D\u0430\u0447\u0430\u0442\u044C \u043E\u0431\u0443\u0447\u0435\u043D\u0438\u0435"
BTN_CONTINUE = "\U0001F525 \u041F\u0440\u043E\u0434\u043E\u043B\u0436\u0438\u0442\u044C"
BTN_FLASHCARDS = "\U0001F9E0 \u0424\u043B\u044D\u0448\u043A\u0430\u0440\u0442\u044B"
BTN_TEST = "\U0001F9EA \u0422\u0435\u0441\u0442"
BTN_PROGRESS = "\U0001F4CA \u041F\u0440\u043E\u0433\u0440\u0435\u0441\u0441"
BTN_SETTINGS = "\u2699 \u041D\u0430\u0441\u0442\u0440\u043E\u0439\u043A\u0438"
BTN_PROFILE = "\U0001F464 \u041F\u0440\u043E\u0444\u0438\u043B\u044C"
BTN_SUPPORT = "\U0001F198 \u041F\u043E\u0434\u0434\u0435\u0440\u0436\u043A\u0430"

BTN_BACK_MENU = "\u2B05 \u041C\u0435\u043D\u044E"
BTN_BACK = "\u2B05 \u041D\u0430\u0437\u0430\u0434"

BTN_SHOW_ANSWER = "\U0001F441 \u041F\u043E\u043A\u0430\u0437\u0430\u0442\u044C \u043E\u0442\u0432\u0435\u0442"
BTN_NEXT = "\u27A1 \u0414\u0430\u043B\u044C\u0448\u0435"
BTN_TEST_NEW = "\U0001F501 \u041D\u043E\u0432\u044B\u0439 \u0442\u0435\u0441\u0442"
BTN_REVIEW_WRONG = "\U0001F4D8 \u041F\u043E\u0432\u0442\u043E\u0440\u0438\u0442\u044C \u043E\u0448\u0438\u0431\u043A\u0438"
BTN_REVIEW_LESSON = "\U0001F4D8 \u041A \u0443\u0440\u043E\u043A\u0443"

BTN_REVIEW_TOPIC = "\U0001F4D8 \u041E\u0431\u0437\u043E\u0440 \u0442\u0435\u043C\u044B"
BTN_PRACTICE = "\U0001F9E9 \u041F\u0440\u0430\u043A\u0442\u0438\u043A\u0430"
BTN_SHOW_SOLUTION = "\U0001F9E0 \u041F\u043E\u043A\u0430\u0437\u0430\u0442\u044C \u0440\u0435\u0448\u0435\u043D\u0438\u0435"

BTN_SETTINGS_NOTIFICATIONS = "\U0001F514 \u0423\u0432\u0435\u0434\u043E\u043C\u043B\u0435\u043D\u0438\u044F"
BTN_SETTINGS_RESET = "\u26A0 \u0421\u0431\u0440\u043E\u0441\u0438\u0442\u044C \u043F\u0440\u043E\u0433\u0440\u0435\u0441\u0441"
BTN_SETTINGS_RESET_CONFIRM = "\u2705 \u0421\u0431\u0440\u043E\u0441\u0438\u0442\u044C"
BTN_SETTINGS_RESET_CANCEL = "\u2B05 \u041D\u0430\u0437\u0430\u0434"
BTN_EXPLAIN_SIMPLE = "\u041e\u0447\u0435\u043d\u044c \u043f\u0440\u043e\u0441\u0442\u043e"
BTN_EXPLAIN_NORMAL = "\u041d\u043e\u0440\u043c\u0430\u043b\u044c\u043d\u043e"
BTN_EXPLAIN_HARD = "\u0421\u043b\u043e\u0436\u043d\u043e"


def _reply_button(text: str) -> KeyboardButton:
    return KeyboardButton(text=text)


def _inline_button(text: str, callback_data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=callback_data)


def _inline_url_button(text: str, url: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, url=url)


def _truncate_text(text: str, limit: int = 48) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return f"{clean[: limit - 3]}..."


def create_main_menu() -> ReplyKeyboardMarkup:
    keyboard = [
        [
            _reply_button(BTN_START_LEARNING),
            _reply_button(BTN_CONTINUE),
        ],
        [
            _reply_button(BTN_PROGRESS),
            _reply_button(BTN_PROFILE),
        ],
    ]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        input_field_placeholder="\u0412\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u0440\u0435\u0436\u0438\u043C",
    )


def create_lesson_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [_inline_button(BTN_FLASHCARDS, "lesson:flashcards")],
        [_inline_button(BTN_TEST, "lesson:test")],
        [_inline_button(BTN_BACK, "lesson:back")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_flashcards_keyboard(*, show_answer: bool) -> InlineKeyboardMarkup:
    keyboard: list[list[InlineKeyboardButton]] = []
    if show_answer:
        keyboard.append([_inline_button(BTN_NEXT, "flash:next")])
    else:
        keyboard.append([_inline_button(BTN_SHOW_ANSWER, "flash:show")])
    keyboard.append([_inline_button(BTN_BACK, "flash:back")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_test_keyboard(options: list[str], question_index: int) -> InlineKeyboardMarkup:
    letters = ["A", "B", "C", "D"]
    keyboard: list[list[InlineKeyboardButton]] = []
    for idx, option in enumerate(options[:4]):
        letter = letters[idx]
        keyboard.append(
            [
                _inline_button(
                    f"{letter}) {option}",
                    f"test:answer:{question_index}:{letter}",
                )
            ]
        )
    keyboard.append([_inline_button(BTN_BACK, "test:back")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_test_result_keyboard() -> InlineKeyboardMarkup:
    keyboard = [[_inline_button(BTN_NEXT, "test:next")]]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_test_done_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [_inline_button(BTN_TEST_NEW, "test:new")],
        [_inline_button(BTN_REVIEW_WRONG, "test:review_wrong")],
        [_inline_button(BTN_PRACTICE, "test:practice")],
        [_inline_button(BTN_BACK_MENU, "test:menu")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_test_review_done_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [_inline_button(BTN_TEST_NEW, "test:new")],
        [_inline_button(BTN_REVIEW_LESSON, "test:review")],
        [_inline_button(BTN_BACK_MENU, "test:menu")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_progress_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[_inline_button(BTN_BACK_MENU, "progress:back")]])


def create_profile_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [_inline_button("\u2B50 \u041F\u043E\u0434\u043F\u0438\u0441\u043A\u0430", "profile:subscription")],
        [_inline_button(BTN_SETTINGS, "profile:settings")],
        [_inline_button(BTN_BACK_MENU, "profile:back")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_settings_keyboard(*, support_url: str | None = None) -> InlineKeyboardMarkup:
    keyboard: list[list[InlineKeyboardButton]] = [
        [_inline_button(BTN_SETTINGS_NOTIFICATIONS, "settings:notifications")],
        [_inline_button(BTN_SETTINGS_RESET, "settings:reset")],
    ]
    if support_url:
        keyboard.append([_inline_url_button(BTN_SUPPORT, support_url)])
    keyboard.append([_inline_button(BTN_BACK_MENU, "settings:back")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_reset_confirm_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [_inline_button(BTN_SETTINGS_RESET_CONFIRM, "settings:reset:confirm")],
        [_inline_button(BTN_SETTINGS_RESET_CANCEL, "settings:reset:cancel")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_practice_keyboard(*, show_solution: bool) -> InlineKeyboardMarkup:
    keyboard: list[list[InlineKeyboardButton]] = []
    if not show_solution:
        keyboard.append([_inline_button(BTN_SHOW_SOLUTION, "practice:show")])
    keyboard.append([_inline_button(BTN_BACK, "practice:back")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_explanation_level_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [_inline_button(BTN_EXPLAIN_SIMPLE, "explain_level:simple")],
        [_inline_button(BTN_EXPLAIN_NORMAL, "explain_level:normal")],
        [_inline_button(BTN_EXPLAIN_HARD, "explain_level:hard")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def create_recent_topics_keyboard(topics: list[str]) -> InlineKeyboardMarkup:
    keyboard: list[list[InlineKeyboardButton]] = []
    for idx, topic in enumerate(topics):
        label = _truncate_text(topic, limit=48)
        keyboard.append([_inline_button(label, f"recent:pick:{idx}")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)
