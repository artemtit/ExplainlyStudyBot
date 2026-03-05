from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.utils.strings import MENU_PROFILE, MENU_STUDY, MENU_SUPPORT


def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=MENU_STUDY, callback_data="study")],
            [InlineKeyboardButton(text=MENU_PROFILE, callback_data="profile")],
            [InlineKeyboardButton(text=MENU_SUPPORT, url="https://t.me/ligr5")],
        ]
    )
