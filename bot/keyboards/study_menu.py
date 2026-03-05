from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.utils.strings import (
    BTN_FINISH_LESSON,
    BTN_NEXT_CARD,
    BTN_NEXT_STAGE,
    BTN_SHOW_ANSWER,
    BTN_SHOW_SOLUTION,
    BTN_TO_FORMATS,
    BTN_TO_LESSON,
    MENU_BACK,
    STUDY_CARDS,
    STUDY_LESSON,
    STUDY_PRACTICE,
    STUDY_TEST,
)
from bot.utils.text_format import next_stage_id


def study_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=STUDY_LESSON, callback_data="lesson")],
            [InlineKeyboardButton(text=STUDY_CARDS, callback_data="cards")],
            [InlineKeyboardButton(text=STUDY_TEST, callback_data="test")],
            [InlineKeyboardButton(text=STUDY_PRACTICE, callback_data="practice")],
            [InlineKeyboardButton(text=MENU_BACK, callback_data="back_to_start")],
        ]
    )


def study_topics_kb(topics: list[str]) -> InlineKeyboardMarkup:
    keyboard: list[list[InlineKeyboardButton]] = []
    for idx, topic in enumerate(topics):
        keyboard.append([InlineKeyboardButton(text=f"{idx + 1}. {topic}", callback_data=f"topic_idx:{idx}")])
    keyboard.append([InlineKeyboardButton(text="\u2795 \u041D\u043E\u0432\u0430\u044F \u0442\u0435\u043C\u0430", callback_data="new_topic")])
    keyboard.append([InlineKeyboardButton(text=MENU_BACK, callback_data="back_to_start")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def stage_nav_kb(stage_id: str) -> InlineKeyboardMarkup:
    keyboard: list[list[InlineKeyboardButton]] = []
    next_stage = next_stage_id(stage_id)
    if next_stage:
        keyboard.append([InlineKeyboardButton(text=BTN_NEXT_STAGE, callback_data=f"next_stage:{next_stage}")])
    else:
        keyboard.append([InlineKeyboardButton(text=BTN_FINISH_LESSON, callback_data="finish_lesson")])
    keyboard.append([InlineKeyboardButton(text=BTN_TO_FORMATS, callback_data="back_to_formats")])
    if stage_id != "lesson":
        keyboard.append([InlineKeyboardButton(text=BTN_TO_LESSON, callback_data="lesson")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def card_question_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=BTN_SHOW_ANSWER, callback_data="card_answer")],
            [InlineKeyboardButton(text=BTN_TO_FORMATS, callback_data="back_to_formats")],
        ]
    )


def card_answer_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=BTN_NEXT_CARD, callback_data="card_next")],
            [InlineKeyboardButton(text=BTN_TO_FORMATS, callback_data="back_to_formats")],
        ]
    )


def test_options_kb(options: list[str], question_index: int) -> InlineKeyboardMarkup:
    keyboard: list[list[InlineKeyboardButton]] = []
    letters = ["A", "B", "C", "D"]
    for idx, option in enumerate(options[:4]):
        letter = letters[idx]
        keyboard.append([InlineKeyboardButton(text=f"{letter}) {option}", callback_data=f"answer:{question_index}:{letter}")])
    keyboard.append([InlineKeyboardButton(text=BTN_TO_FORMATS, callback_data="back_to_formats")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def practice_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=BTN_SHOW_SOLUTION, callback_data="solution")],
            [InlineKeyboardButton(text=BTN_TO_FORMATS, callback_data="back_to_formats")],
        ]
    )


def profile_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=MENU_BACK, callback_data="back_to_start")]]
    )
