# bot.py — полный рабочий файл ExplainlyStudy
import os
import json
import asyncio
import hashlib
import random
from typing import List, Dict, Any, Optional

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

# prompts.py должен быть в проекте
from prompts import build_system_prompt, build_user_prompt

# ----------------- Load env -----------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not all([TELEGRAM_TOKEN, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise RuntimeError("Проверь .env — нужны TELEGRAM_TOKEN, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY")

# ----------------- Clients -----------------
bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties())  # не указываем parse_mode здесь
dp = Dispatcher(storage=MemoryStorage())

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------- FSM -----------------
class StudyState(StatesGroup):
    material = State()
    card_index = State()
    test_index = State()
    test_score = State()

# ----------------- Utils -----------------
def normalize_topic(topic: str) -> str:
    return " ".join(topic.lower().strip().split())

def topic_hash(topic: str) -> str:
    return hashlib.sha256(normalize_topic(topic).encode()).hexdigest()

def safe_json_parse(text: str) -> dict:
    """
    Извлекает JSON-объект из текста (между первым { и последним }).
    Бросает ValueError/JSONDecodeError при проблемах.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSON not found in model response")
    raw = text[start:end + 1]
    return json.loads(raw)

def get_correct_answer(test: dict) -> str:
    if "correct" in test:
        return test["correct"]
    if "answer" in test:
        return test["answer"]
    return "A"  # fallback, но мы стараемся избежать постоянного "A" через нормализацию

def normalize_tests(tests: List[dict]) -> None:
    """
    Приводит тесты к более стабильной форме:
    - options: list (если dict -> преобразует в list по A..D)
    - correct: если отсутствует или невалиден -> выбирается случайная буква из доступных
    Изменяет список на месте.
    """
    letters = ["A", "B", "C", "D", "E"]
    for t in tests:
        opts = t.get("options")
        # если dict -> собрать по буквам
        if isinstance(opts, dict):
            opts_list = []
            for L in letters:
                if L in opts:
                    opts_list.append(opts[L])
            if not opts_list:
                # fallback: взять значения
                opts_list = list(opts.values())
            t["options"] = opts_list
        elif isinstance(opts, list):
            # ok
            pass
        else:
            # отсутствуют варианты — сделаем пустой список (на всякий случай)
            t["options"] = []

        # нормализуем correct: если есть и это число/строка, приведём к букве
        correct = t.get("correct") or t.get("answer")
        if isinstance(correct, int):
            # индекс -> буква
            if 0 <= correct < len(t["options"]):
                t["correct"] = letters[correct]
            else:
                t["correct"] = random.choice(letters[:max(1, len(t["options"]))]) if t["options"] else "A"
        elif isinstance(correct, str):
            corr = correct.strip().upper()
            if corr in letters[:len(t["options"])]:
                t["correct"] = corr
            else:
                # возможно модель вернула текст ответа -> попробуем найти совпадение в options
                found = False
                for i, opt in enumerate(t["options"]):
                    if corr == opt.upper() or corr in opt.upper():
                        t["correct"] = letters[i]
                        found = True
                        break
                if not found:
                    # случайная буква из доступных
                    t["correct"] = random.choice(letters[:max(1, len(t["options"]))]) if t["options"] else "A"
        else:
            # не задано -> случайная буква
            t["correct"] = random.choice(letters[:max(1, len(t["options"]))]) if t["options"] else "A"

# ----------------- DB helpers (Supabase) -----------------
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
    if res.data:
        return res.data[0]["content"]
    return None

def save_material_to_db(topic: str, material: dict):
    supabase.table("materials").insert({
        "topic": normalize_topic(topic),
        "topic_hash": topic_hash(topic),
        "content": material
    }).execute()

# ----------------- OpenAI call -----------------
async def generate_material(topic: str) -> Dict[str, Any]:
    """
    Вызывает модель и пытается распарсить JSON до 3 попыток.
    После парсинга нормализуем тесты.
    """
    def sync_call():
        return openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(topic)},
            ],
            temperature=0.2,
            max_tokens=1800,
        )

    last_text = ""
    for _ in range(3):
        resp = await asyncio.to_thread(sync_call)
        try:
            text = resp.choices[0].message.content
        except Exception:
            text = str(resp)
        last_text = text
        try:
            material = safe_json_parse(text)
            # нормализуем части материала: тесты, карточки, уроки, практика — минимум проверок
            if "tests" in material and isinstance(material["tests"], list):
                normalize_tests(material["tests"])
            return material
        except Exception:
            continue
    raise RuntimeError("Не удалось распарсить JSON от модели. Последний ответ: " + (last_text[:400] if last_text else "empty"))

# ----------------- Formatting -----------------
def format_lesson(lesson: dict) -> str:
    """
    Возвращает HTML-строку урока — будем отправлять с parse_mode="HTML".
    Обрезаем вложенные теги, используем <b>, <code>.
    """
    title = lesson.get("title", "Мини-урок")
    parts = [f"<b>{escape_html(title)}</b>\n"]
    for sec in lesson.get("sections", []):
        header = sec.get("header")
        if header:
            parts.append(f"🔹 <b>{escape_html(header)}</b>")
        text = sec.get("text", "")
        if text:
            parts.append(escape_html(text))
        for kp in sec.get("key_points", []):
            parts.append(f"• {escape_html(kp)}")
        formula = sec.get("formula")
        if formula:
            parts.append(f"📐 Формула: <code>{escape_html(formula)}</code>")
        parts.append("")  # blank line
    return "\n".join(parts)

def escape_html(s: str) -> str:
    """Простая экранизация для HTML-отправки."""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))

# ----------------- Keyboards -----------------
def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📘 Учёба", callback_data="study")],
        [InlineKeyboardButton(text="👤 Профиль", callback_data="profile")],
        [InlineKeyboardButton(text="🆘 Поддержка", url="https://t.me/ligr5")],
    ])

def study_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📖 Мини-урок", callback_data="lesson")],
        [InlineKeyboardButton(text="🧠 Карточки", callback_data="cards")],
        [InlineKeyboardButton(text="📝 Тест", callback_data="test")],
        [InlineKeyboardButton(text="🧪 Практика", callback_data="practice")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_start")],
    ])

def study_topics_kb(topics: List[str]) -> InlineKeyboardMarkup:
    kb = []
    for i, t in enumerate(topics):
        kb.append([InlineKeyboardButton(text=f"📌 {t}", callback_data=f"topic_idx:{i}")])
    kb.append([InlineKeyboardButton(text="➕ Новая тема", callback_data="new_topic")])
    kb.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_start")])
    return InlineKeyboardMarkup(inline_keyboard=kb)

def test_kb(options: List[str], idx: int) -> InlineKeyboardMarkup:
    kb = []
    letters = ["A", "B", "C", "D", "E"]
    for i, text in enumerate(options):
        if i >= len(letters):
            break
        letter = letters[i]
        kb.append([InlineKeyboardButton(text=f"{letter}) {text}", callback_data=f"answer:{idx}:{letter}")])
    kb.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")])
    return InlineKeyboardMarkup(inline_keyboard=kb)

def finish_lesson_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Завершить урок", callback_data="finish_lesson")]
    ])

# ----------------- UI helpers -----------------
async def show_start_screen(target):
    # target — Message (мы всегда передаём message)
    await target.answer(
        "👋 Привет! Я ExplainlyStudy.\nОтправь тему текстом — помогу её изучить.",
        reply_markup=main_menu()
    )

async def get_material_or_restart(call: CallbackQuery, state: FSMContext) -> Optional[dict]:
    data = await state.get_data()
    material = data.get("material")
    if not material:
        await call.message.answer("⚠️ Сессия устарела. Пожалуйста, выбери тему заново.")
        await show_start_screen(call.message)
        return None
    return material

# ----------------- Handlers -----------------
@dp.message(CommandStart())
async def start(message: Message):
    ensure_user(message.from_user.id, message.from_user.username)
    await show_start_screen(message)

@dp.message()
async def handle_message(message: Message, state: FSMContext):
    if not message.text:
        await message.answer("❗ Я пока понимаю только текстовые темы.")
        return

    topic = message.text.strip()
    ensure_user(message.from_user.id, message.from_user.username)
    save_request(message.from_user.id, topic)

    # проверка в БД
    material = get_material_from_db(topic)
    if material:
        await message.answer("📦 Материал загружен из базы.")
    else:
        await message.answer("⏳ Генерирую материал...")
        try:
            material = await generate_material(topic)
        except Exception as e:
            await message.answer("❌ Не удалось сгенерировать материал. Попробуй переформулировать тему.")
            return
        save_material_to_db(topic, material)

    # сохраняем topic и material в state
    await state.set_state(StudyState.material)
    await state.update_data(material=material, topic=topic)

    await message.answer(f"📌 Тема: {topic}\nВыбери формат обучения:", reply_markup=study_menu())

@dp.callback_query(F.data == "study")
async def open_study(call: CallbackQuery, state: FSMContext):
    await call.answer()
    topics = get_last_requests(call.from_user.id)
    await state.update_data(last_topics=topics)
    if not topics:
        await call.message.answer("📘 У тебя пока нет тем. Отправь новую тему текстом.", reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="➕ Новая тема", callback_data="new_topic")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_start")],
        ]))
        return
    await call.message.answer("📘 Выбери тему:", reply_markup=study_topics_kb(topics))

@dp.callback_query(F.data.startswith("topic_idx:"))
async def select_old_topic(call: CallbackQuery, state: FSMContext):
    await call.answer()
    idx = int(call.data.replace("topic_idx:", ""))
    data = await state.get_data()
    last_topics = data.get("last_topics", [])
    if idx < 0 or idx >= len(last_topics):
        await call.message.answer("❗ Неверный индекс темы.")
        return
    topic = last_topics[idx]
    material = get_material_from_db(topic)
    if not material:
        await call.message.answer("⏳ Генерирую материал...")
        try:
            material = await generate_material(topic)
        except Exception:
            await call.message.answer("❌ Ошибка генерации.")
            return
        save_material_to_db(topic, material)
    else:
        await call.message.answer("📦 Материал загружен из базы.")
    await state.set_state(StudyState.material)
    await state.update_data(material=material, topic=topic)
    await call.message.answer(f"📌 Тема: {topic}\nВыбери формат обучения:", reply_markup=study_menu())

@dp.callback_query(F.data == "new_topic")
async def new_topic(call: CallbackQuery):
    await call.answer()
    await call.message.answer("✍️ Отправь новую тему текстом.")

@dp.callback_query(F.data == "back_to_start")
async def back_to_start(call: CallbackQuery):
    await call.answer()
    await show_start_screen(call.message)

@dp.callback_query(F.data == "profile")
async def profile(call: CallbackQuery):
    await call.answer()
    await call.message.answer(f"👤 Профиль\nID: {call.from_user.id}\nСтатус: Free")

# ---------- Форматы ----------
@dp.callback_query(F.data == "lesson")
async def lesson(call: CallbackQuery, state: FSMContext):
    await call.answer()
    material = await get_material_or_restart(call, state)
    if not material:
        return
    lesson_obj = material.get("lesson", {})
    # Отправляем HTML (format_lesson вернёт экранированный HTML)
    await call.message.answer(format_lesson(lesson_obj), parse_mode="HTML", reply_markup=InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")]
    ]))

@dp.callback_query(F.data == "back_to_formats")
async def back_to_formats(call: CallbackQuery, state: FSMContext):
    await call.answer()
    data = await state.get_data()
    topic = data.get("topic")
    if not topic:
        await show_start_screen(call.message)
        return
    await call.message.answer(f"📌 Тема: {topic}\nВыбери формат обучения:", reply_markup=study_menu())

# ---------- Карточки ----------
@dp.callback_query(F.data == "cards")
async def start_cards(call: CallbackQuery, state: FSMContext):
    await call.answer()
    material = await get_material_or_restart(call, state)
    if not material:
        return
    await state.update_data(card_index=0)
    await send_card(call, state)

async def send_card(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    cards = data.get("material", {}).get("cards", [])
    idx = data.get("card_index", 0)
    total = len(cards)
    if idx >= total:
        await call.message.answer("🎉 Карточки закончились.", reply_markup=finish_lesson_kb())
        return
    card = cards[idx]
    await call.message.answer(
        f"🧠 Карточка {idx + 1} / {total}\n\n❓ {card.get('question', '—')}",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Показать ответ", callback_data="card_answer")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")],
        ])
    )

@dp.callback_query(F.data == "card_answer")
async def card_answer(call: CallbackQuery, state: FSMContext):
    await call.answer()
    data = await state.get_data()
    idx = data.get("card_index", 0)
    cards = data.get("material", {}).get("cards", [])
    if idx < 0 or idx >= len(cards):
        await call.message.answer("❗ Карточка не найдена.")
        return
    answer_text = cards[idx].get("answer", "—")
    await call.message.answer(
        "✅ Ответ:\n" + answer_text,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Следующая", callback_data="card_next")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")],
        ])
    )

@dp.callback_query(F.data == "card_next")
async def card_next(call: CallbackQuery, state: FSMContext):
    await call.answer()
    data = await state.get_data()
    idx = data.get("card_index", 0) + 1
    await state.update_data(card_index=idx)
    await send_card(call, state)

# ---------- Тест ----------
@dp.callback_query(F.data == "test")
async def start_test(call: CallbackQuery, state: FSMContext):
    await call.answer()
    material = await get_material_or_restart(call, state)
    if not material:
        return
    await state.update_data(test_index=0, test_score=0)
    await send_test(call, state)

async def send_test(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    material = data.get("material", {})
    tests = material.get("tests", [])
    idx = data.get("test_index", 0)
    total = len(tests)
    if idx >= total:
        await call.message.answer(f"🎉 Тест завершён!\nРезультат: {data.get('test_score', 0)} / {total}", reply_markup=finish_lesson_kb())
        return
    t = tests[idx]
    question = t.get("question", "—")
    options = t.get("options", [])
    # Если options — dict, преобразуем
    if isinstance(options, dict):
        letters = ["A", "B", "C", "D", "E"]
        opts_list = []
        for L in letters:
            if L in options:
                opts_list.append(options[L])
        if not opts_list:
            opts_list = list(options.values())
        options = opts_list
    await call.message.answer(f"📝 Вопрос {idx + 1} из {total}\n\n{question}", reply_markup=test_kb(options, idx))

@dp.callback_query(F.data.startswith("answer:"))
async def answer(call: CallbackQuery, state: FSMContext):
    await call.answer()
    try:
        _, idx_s, chosen = call.data.split(":")
        idx = int(idx_s)
    except Exception:
        await call.message.answer("❗ Некорректные данные ответа.")
        return
    material = await get_material_or_restart(call, state)
    if not material:
        return
    tests = material.get("tests", [])
    if idx < 0 or idx >= len(tests):
        await call.message.answer("❗ Неверный индекс вопроса.")
        return
    data = await state.get_data()
    score = data.get("test_score", 0)
    test = tests[idx]
    correct = get_correct_answer(test)
    if chosen == correct:
        score += 1
        await call.message.answer("✅ Верно!")
    else:
        await call.message.answer(f"❌ Неверно. Правильный ответ: {correct}")
    await state.update_data(test_index=idx + 1, test_score=score)
    await send_test(call, state)

# ---------- Практика ----------
@dp.callback_query(F.data == "practice")
async def practice(call: CallbackQuery, state: FSMContext):
    await call.answer()
    material = await get_material_or_restart(call, state)
    if not material:
        return
    p = material.get("practice", {})
    problem = p.get("problem", "—")
    await call.message.answer("🧪 Практика:\n" + problem, reply_markup=InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Показать решение", callback_data="solution")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")]
    ]))

@dp.callback_query(F.data == "solution")
async def solution(call: CallbackQuery, state: FSMContext):
    await call.answer()
    material = await get_material_or_restart(call, state)
    if not material:
        return
    p = material.get("practice", {})
    await call.message.answer("✅ Решение:\n" + p.get("solution", "—"), reply_markup=finish_lesson_kb())

# ---------- Завершение урока ----------
@dp.callback_query(F.data == "finish_lesson")
async def finish_lesson(call: CallbackQuery, state: FSMContext):
    await call.answer()
    data = await state.get_data()
    topic = data.get("topic")
    # Очистим состояние урока (но можно сохранить историю)
    await state.clear()
    if not topic:
        await show_start_screen(call.message)
        return
    await call.message.answer(f"Вы завершили тему: {topic}\nЧто дальше?", reply_markup=main_menu())

# ----------------- Run -----------------
async def main():
    print("ExplainlyStudy — полный рабочий MVP запущен")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
# Запуск: python bot.py
# Установка зависимостей: pip install -r requirements.txt
