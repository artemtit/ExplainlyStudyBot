from __future__ import annotations

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from bot.handlers.common import ensure_material_data
from bot.keyboards.study_menu import card_answer_kb, card_question_kb, stage_nav_kb
from bot.utils.locks import UserLockManager
from bot.utils.strings import CARD_ANSWER_TEMPLATE, CARD_NOT_FOUND, CARD_QUESTION_TEMPLATE, CARDS_DONE_TEXT, CARDS_NONE_TEXT
from bot.utils.telegram_utils import edit_or_send
from bot.utils.text_format import format_stage_header


async def _render_card(call: CallbackQuery, state: FSMContext) -> None:
    payload = await ensure_material_data(call, state)
    if payload is None:
        return

    material, topic = payload
    cards = material.get("cards", [])
    if not isinstance(cards, list):
        cards = []

    data = await state.get_data()
    index = int(data.get("card_index", 0))
    header = format_stage_header(topic, "cards")

    if not cards:
        await edit_or_send(call, f"{header}\n\n{CARDS_NONE_TEXT}", reply_markup=stage_nav_kb("cards"))
        return

    if index >= len(cards):
        await edit_or_send(call, f"{header}\n\n{CARDS_DONE_TEXT}", reply_markup=stage_nav_kb("cards"))
        return

    card = cards[index] if isinstance(cards[index], dict) else {}
    question = str(card.get("question") or "—")
    body = CARD_QUESTION_TEMPLATE.format(pos=index + 1, total=len(cards), question=question)
    await edit_or_send(call, f"{header}\n\n{body}", reply_markup=card_question_kb())


async def open_cards(call: CallbackQuery, state: FSMContext, *, reset_index: bool) -> None:
    await state.update_data(current_stage="cards")
    if reset_index:
        await state.update_data(card_index=0)
    await _render_card(call, state)


def build_router(lock_manager: UserLockManager) -> Router:
    router = Router(name="cards")

    @router.callback_query(F.data == "cards")
    async def cards_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            await open_cards(call, state, reset_index=True)

    @router.callback_query(F.data == "card_answer")
    async def card_answer_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            payload = await ensure_material_data(call, state)
            if payload is None:
                return

            material, topic = payload
            cards = material.get("cards", [])
            if not isinstance(cards, list):
                cards = []

            data = await state.get_data()
            index = int(data.get("card_index", 0))
            header = format_stage_header(topic, "cards")

            if index < 0 or index >= len(cards):
                await edit_or_send(call, f"{header}\n\n{CARD_NOT_FOUND}", reply_markup=stage_nav_kb("cards"))
                return

            card = cards[index] if isinstance(cards[index], dict) else {}
            answer = str(card.get("answer") or "—")
            body = CARD_ANSWER_TEMPLATE.format(pos=index + 1, total=len(cards), answer=answer)
            await edit_or_send(call, f"{header}\n\n{body}", reply_markup=card_answer_kb())

    @router.callback_query(F.data == "card_next")
    async def card_next_handler(call: CallbackQuery, state: FSMContext) -> None:
        async with await lock_manager.get(call.from_user.id):
            await call.answer()
            data = await state.get_data()
            await state.update_data(card_index=int(data.get("card_index", 0)) + 1)
            await _render_card(call, state)

    return router
