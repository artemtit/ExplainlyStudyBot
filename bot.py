# bot.py — ExplainlyStudyBot
# ПОЛНАЯ ВЕРСИЯ. UX И FSM НЕ УПРОЩЕНЫ. ИСПРАВЛЕНЫ ТОЛЬКО НЕДОЧЕТЫ.

import os
import json
import asyncio
import hashlib
import random
from typing import List, Dict, Any, Optional
from aiohttp import web

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.filters import CommandStart
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

from prompts import build_system_prompt, build_user_prompt

# ----------------- ENV -----------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not all([TELEGRAM_TOKEN, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise RuntimeError("Ошибка .env")

# ----------------- CLIENTS -----------------
bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties())
dp = Dispatcher(storage=MemoryStorage())

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------- FSM -----------------
class StudyState(StatesGroup):
    material = State()
    card_index = State()
    test_index = State()
    test_score = State()

# ----------------- UTILS -----------------
LETTERS = ["A", "B", "C", "D"]

def normalize_topic(topic: str) -> str:
    return " ".join(topic.lower().strip().split())

def topic_hash(topic: str) -> str:
    return hashlib.sha256(normalize_topic(topic).encode()).hexdigest()

def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def safe_json_parse(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSON not found")
    return json.loads(text[start:end + 1])

def normalize_tests(tests: List[dict]) -> None:
    """
    ФИКС: функция теперь РЕАЛЬНО используется
    Приводит tests к стабильному виду
    """
    for t in tests:
        options = t.get("options")

        if isinstance(options, dict):
            opts = []
            for L in LETTERS:
                if L in options:
                    opts.append(options[L])
            options = opts or list(options.values())

        if not isinstance(options, list):
            options = []

        options = options[:4]
        while len(options) < 4:
            options.append("—")

        t["options"] = options

        correct = t.get("correct") or t.get("answer")

        if isinstance(correct, int) and 0 <= correct < 4:
            t["correct"] = LETTERS[correct]
        elif isinstance(correct, str):
            c = correct.strip().upper()
            if c in LETTERS:
                t["correct"] = c
            else:
                found = False
                for i, opt in enumerate(options):
                    if correct.lower() in opt.lower():
                        t["correct"] = LETTERS[i]
                        found = True
                        break
                if not found:
                    t["correct"] = random.choice(LETTERS)
        else:
            t["correct"] = random.choice(LETTERS)

def validate_material(material: dict) -> None:
    if not isinstance(material, dict):
        raise ValueError("material not dict")

    if "lesson" not in material or "cards" not in material or "tests" not in material:
        raise ValueError("material structure broken")

    if len(material["cards"]) != 5 or len(material["tests"]) != 5:
        raise ValueError("wrong cards/tests count")

    normalize_tests(material["tests"])

# ----------------- DB -----------------
def ensure_user(user_id: int, username: Optional[str]):
    supabase.table("users").upsert({
        "id": user_id,
        "username": username
    }).execute()

def save_request(user_id: int, topic: str):
    supabase.table("requests").insert({
        "user_id": user_id,
        "topic": topic
    }).execute()

def get_last_requests(user_id: int, limit: int = 3) -> List[str]:
    res = (
        supabase.table("requests")
        .select("topic")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [r["topic"] for r in (res.data or [])]

def get_material_from_db(topic: str) -> Optional[dict]:
    h = topic_hash(topic)
    res = (
        supabase.table("materials")
        .select("content")
        .eq("topic_hash", h)
        .limit(1)
        .execute()
    )
    return res.data[0]["content"] if res.data else None

def save_material_to_db(topic: str, material: dict):
    supabase.table("materials").upsert({
        "topic": normalize_topic(topic),
        "topic_hash": topic_hash(topic),
        "content": material,
        "model": OPENAI_MODEL
    }).execute()

# ----------------- OPENAI -----------------
async def generate_material(topic: str) -> Dict[str, Any]:
    """
    Надёжная генерация учебного материала.
    НЕ меняет UX.
    НЕ упрощает FSM.
    Исправляет проблемы Responses API и LLM.
    """

    def sync_call() -> Any:
        return openai_client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": build_system_prompt(),
                },
                {
                    "role": "user",
                    "content": build_user_prompt(topic),
                },
            ],
            max_output_tokens=1800,
        )

    last_error: Optional[Exception] = None

    for attempt in range(3):
        try:
            response = await asyncio.to_thread(sync_call)

            # 1. Пытаемся получить текст напрямую (output_text может быть None)
            text = getattr(response, "output_text", None)

            # 2. Если пусто — пытаемся собрать текст из output[]
            if not text and hasattr(response, "output"):
                parts = []
                for item in response.output:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    if not isinstance(content, list):
                        continue
                    for c in content:
                        if c.get("type") == "output_text":
                            parts.append(c.get("text", ""))
                text = "\n".join(parts).strip()

            if not text:
                raise ValueError("LLM returned empty text")

            # 3. Парсим JSON
            material = safe_json_parse(text)

            # 4. МЯГКАЯ структурная проверка (НЕ валим за мелочи)
            if not isinstance(material, dict):
                raise ValueError("material is not dict")

            if "lesson" not in material:
                raise ValueError("missing lesson")
            if "cards" not in material or not isinstance(material["cards"], list):
                raise ValueError("missing cards")
            if "tests" not in material or not isinstance(material["tests"], list):
                raise ValueError("missing tests")
            if "practice" not in material:
                raise ValueError("missing practice")

            # 5. Нормализация количеств (НЕ ломает UX)
            material["cards"] = material["cards"][:5]
            material["tests"] = material["tests"][:5]

            while len(material["cards"]) < 5:
                material["cards"].append({
                    "question": "—",
                    "answer": "—"
                })

            while len(material["tests"]) < 5:
                material["tests"].append({
                    "question": "—",
                    "options": ["—", "—", "—", "—"],
                    "correct": "A"
                })

            # 6. Нормализация тестов (критично)
            normalize_tests(material["tests"])

            # 7. Финальная страховка
            for t in material["tests"]:
                if "correct" not in t or t["correct"] not in ["A", "B", "C", "D"]:
                    t["correct"] = "A"

            return material

        except Exception as e:
            last_error = e
            # можно логировать при необходимости
            # print(f"[generate_material] attempt {attempt+1} failed:", e)
            await asyncio.sleep(0.5)


    raise RuntimeError(f"Не удалось сгенерировать материал: {last_error}")


# ----------------- KEYBOARDS -----------------
def main_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📘 Учёба", callback_data="study")],
        [InlineKeyboardButton(text="👤 Профиль", callback_data="profile")],
    ])

def study_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📖 Мини-урок", callback_data="lesson")],
        [InlineKeyboardButton(text="🧠 Карточки", callback_data="cards")],
        [InlineKeyboardButton(text="📝 Тест", callback_data="test")],
        [InlineKeyboardButton(text="🧪 Практика", callback_data="practice")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_start")],
    ])

def test_kb(options: List[str], idx: int):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text=f"{LETTERS[i]}) {opt}",
            callback_data=f"answer:{idx}:{LETTERS[i]}"
        )] for i, opt in enumerate(options)
    ] + [[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")]])

def finish_lesson_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Завершить урок", callback_data="finish_lesson")]
    ])

# ----------------- HELPERS -----------------
async def show_start_screen(msg: Message):
    await msg.answer(
        "👋 Привет! Отправь тему для изучения.",
        reply_markup=main_menu()
    )

async def get_material_or_restart(call: CallbackQuery, state: FSMContext) -> Optional[dict]:
    data = await state.get_data()
    material = data.get("material")
    if not material:
        await call.message.answer("⚠️ Сессия устарела. Выбери тему заново.")
        await show_start_screen(call.message)
        return None
    return material

# ----------------- HANDLERS -----------------
@dp.message(CommandStart())
async def start(message: Message):
    ensure_user(message.from_user.id, message.from_user.username)
    await show_start_screen(message)

@dp.message()
async def handle_message(message: Message, state: FSMContext):
    if not message.text:
        await message.answer("❗ Только текст.")
        return

    topic = message.text.strip()
    ensure_user(message.from_user.id, message.from_user.username)
    save_request(message.from_user.id, topic)

    material = get_material_from_db(topic)
    if not material:
        await message.answer("⏳ Генерирую материал...")
        material = await generate_material(topic)
        save_material_to_db(topic, material)
    else:
        await message.answer("📦 Материал из базы.")

    await state.set_state(StudyState.material)
    await state.update_data(material=material, topic=topic)

    await message.answer(f"📌 Тема: {topic}", reply_markup=study_menu())

# ---------- форматы ----------
@dp.callback_query(F.data == "lesson")
async def lesson(call: CallbackQuery, state: FSMContext):
    await call.answer()
    material = await get_material_or_restart(call, state)
    if not material:
        return

    lesson = material["lesson"]
    text = f"<b>{escape_html(lesson.get('title', 'Урок'))}</b>\n\n"
    for sec in lesson.get("sections", []):
        text += f"<b>{escape_html(sec.get('header',''))}</b>\n"
        text += f"{escape_html(sec.get('text',''))}\n\n"

    await call.message.answer(text, parse_mode="HTML",
                              reply_markup=InlineKeyboardMarkup(
                                  inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")]]
                              ))

@dp.callback_query(F.data == "back_to_formats")
async def back_to_formats(call: CallbackQuery, state: FSMContext):
    await call.answer()
    data = await state.get_data()
    topic = data.get("topic")
    if not topic:
        await show_start_screen(call.message)
        return
    await call.message.answer(f"📌 Тема: {topic}", reply_markup=study_menu())

# ---------- карточки ----------
@dp.callback_query(F.data == "cards")
async def start_cards(call: CallbackQuery, state: FSMContext):
    await call.answer()
    await state.update_data(card_index=0)
    await send_card(call, state)

async def send_card(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    cards = data["material"]["cards"]
    idx = data.get("card_index", 0)

    if idx >= len(cards):
        await call.message.answer("🎉 Карточки закончились.", reply_markup=finish_lesson_kb())
        return

    card = cards[idx]
    await call.message.answer(
        f"🧠 Карточка {idx+1}/5\n\n❓ {card['question']}",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Показать ответ", callback_data="card_answer")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")]
        ])
    )

@dp.callback_query(F.data == "card_answer")
async def card_answer(call: CallbackQuery, state: FSMContext):
    await call.answer()
    data = await state.get_data()
    idx = data["card_index"]
    card = data["material"]["cards"][idx]

    await call.message.answer(
        f"✅ Ответ:\n{card['answer']}",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Следующая", callback_data="card_next")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")]
        ])
    )

@dp.callback_query(F.data == "card_next")
async def card_next(call: CallbackQuery, state: FSMContext):
    await call.answer()
    idx = (await state.get_data()).get("card_index", 0) + 1
    await state.update_data(card_index=idx)
    await send_card(call, state)

# ---------- тест ----------
@dp.callback_query(F.data == "test")
async def start_test(call: CallbackQuery, state: FSMContext):
    await call.answer()
    await state.update_data(test_index=0, test_score=0)
    await send_test(call, state)

async def send_test(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    tests = data["material"]["tests"]
    idx = data.get("test_index", 0)

    if idx >= len(tests):
        await call.message.answer(
            f"🎉 Тест завершён!\nРезультат: {data['test_score']} / {len(tests)}",
            reply_markup=finish_lesson_kb()
        )
        return

    t = tests[idx]
    await call.message.answer(
        f"📝 Вопрос {idx+1}/5\n\n{t['question']}",
        reply_markup=test_kb(t["options"], idx)
    )

@dp.callback_query(F.data.startswith("answer:"))
async def answer(call: CallbackQuery, state: FSMContext):
    await call.answer()
    _, idx_s, chosen = call.data.split(":")
    idx = int(idx_s)

    data = await state.get_data()
    test = data["material"]["tests"][idx]
    correct = test["correct"]
    score = data["test_score"]

    if chosen == correct:
        score += 1
        await call.message.answer("✅ Верно!")
    else:
        ci = LETTERS.index(correct)
        await call.message.answer(
            f"❌ Неверно.\nПравильно: {correct}) {test['options'][ci]}"
        )

    await state.update_data(test_index=idx + 1, test_score=score)
    await send_test(call, state)

# ---------- практика ----------
@dp.callback_query(F.data == "practice")
async def practice(call: CallbackQuery, state: FSMContext):
    await call.answer()
    p = (await state.get_data())["material"]["practice"]
    await call.message.answer(
        f"🧪 Практика:\n{p.get('problem','—')}",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Показать решение", callback_data="solution")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")]
        ])
    )

@dp.callback_query(F.data == "solution")
async def solution(call: CallbackQuery, state: FSMContext):
    await call.answer()
    p = (await state.get_data())["material"]["practice"]
    await call.message.answer(
        f"✅ Решение:\n{p.get('solution','—')}",
        reply_markup=finish_lesson_kb()
    )

# ---------- завершение ----------
@dp.callback_query(F.data == "finish_lesson")
async def finish_lesson(call: CallbackQuery, state: FSMContext):
    await call.answer()
    topic = (await state.get_data()).get("topic")
    await state.clear()
    await call.message.answer(f"Тема завершена: {topic}", reply_markup=main_menu())

# ---------- WEB ----------
async def run_webserver():
    app = web.Application()
    app.add_routes([web.get("/health", lambda _: web.Response(text="ok"))])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8000)))
    await site.start()
    while True:
        await asyncio.sleep(3600)

async def main():
    await asyncio.gather(
        run_webserver(),
        dp.start_polling(bot)
    )

if __name__ == "__main__":
    asyncio.run(main())

# python bot.py
#pip install -r requirements.txt