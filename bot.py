# bot.py — ExplainlyStudy (refactored for stability, async-safety, UX)
import os
import json
import asyncio
import hashlib
import random
import logging
from typing import List, Dict, Any, Optional
from contextlib import suppress
import requests
from openai import OpenAI

from aiohttp import web
from dotenv import load_dotenv
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
from aiogram.exceptions import TelegramBadRequest

from prompts import build_system_prompt, build_user_prompt

# ----------------- Logging -----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("explainly")

# ----------------- Load env -----------------
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
REDIS_DSN = os.getenv("REDIS_DSN")

if not all([TELEGRAM_TOKEN, GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise RuntimeError(
        "Проверь .env — нужны TELEGRAM_TOKEN, GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY"
    )

openai_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
# ----------------- FSM storage -----------------
storage = None
if REDIS_DSN:
    try:
        from aiogram.fsm.storage.redis import RedisStorage, DefaultKeyBuilder
        from redis.asyncio import from_url as redis_from_url

        redis = redis_from_url(REDIS_DSN, decode_responses=True)
        storage = RedisStorage(redis=redis, key_builder=DefaultKeyBuilder(with_bot_id=True, with_destiny=True))
        logger.info("Using RedisStorage for FSM")
    except Exception as e:
        logger.exception("Failed to init RedisStorage, fallback to MemoryStorage: %s", e)
        storage = MemoryStorage()
else:
    storage = MemoryStorage()

# ----------------- Clients -----------------
bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties())  # explicit parse_mode per message
dp = Dispatcher(storage=storage)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------- FSM -----------------
class StudyState(StatesGroup):
    material = State()
    card_index = State()
    test_index = State()
    test_score = State()
    accepting_answer = State()

# ----------------- Concurrency guards -----------------
class UserLockManager:
    def __init__(self) -> None:
        self._locks: Dict[int, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def get(self, user_id: int) -> asyncio.Lock:
        # Double-checked pattern to avoid race on dict creation
        lock = self._locks.get(user_id)
        if lock is None:
            async with self._global_lock:
                lock = self._locks.get(user_id)
                if lock is None:
                    lock = asyncio.Lock()
                    self._locks[user_id] = lock
        return lock

locks = UserLockManager()

# ----------------- Utils -----------------
def normalize_topic(topic: str) -> str:
    return " ".join(topic.lower().strip().split())

def topic_hash(topic: str) -> str:
    return hashlib.sha256(normalize_topic(topic).encode()).hexdigest()

def strip_code_fences(text: str) -> str:
    t = text.strip()
    # Remove common Markdown fences ```json ... ``` or ``` ... ```
    if t.startswith("```") and t.endswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2:
            inner = "\n".join(lines[1:-1])
            return inner.strip()
    return t


def safe_json_parse(text: str) -> dict:
    """
    Извлекает JSON-объект из текста (между первым { и последним }).
    Бросает ValueError/JSONDecodeError при проблемах.
    """
    t = strip_code_fences(text)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON not found in model response")
    raw = t[start:end + 1]
    return json.loads(raw)

def get_correct_answer(test: dict) -> str:
    if "correct" in test:
        return str(test["correct"]).strip().upper()
    if "answer" in test:
        return str(test["answer"]).strip().upper()
    return "A"

def normalize_tests(tests: List[dict]) -> None:
    """
    Приводит тесты к стабильной форме:
    - options: list (если dict -> преобразует в list по A..D)
    - correct: если отсутствует или невалиден -> выбирается случайная буква из доступных
    Изменяет список на месте.
    """
    letters = ["A", "B", "C", "D", "E"]
    for t in tests:
        opts = t.get("options")
        if isinstance(opts, dict):
            opts_list = []
            for L in letters:
                if L in opts:
                    opts_list.append(opts[L])
            if not opts_list:
                opts_list = list(opts.values())
            t["options"] = opts_list
        elif isinstance(opts, list):
            pass
        else:
            t["options"] = []

        correct = t.get("correct") or t.get("answer")
        if isinstance(correct, int):
            if 0 <= correct < len(t["options"]):
                t["correct"] = letters[correct]
            else:
                t["correct"] = letters[0] if t["options"] else "A"
        elif isinstance(correct, str):
            corr = correct.strip().upper()
            if corr in letters[: len(t["options"])]:
                t["correct"] = corr
            else:
                # попытка сопоставить по тексту
                found = False
                for i, opt in enumerate(t["options"]):
                    if corr == str(opt).strip().upper() or corr in str(opt).strip().upper():
                        t["correct"] = letters[i]
                        found = True
                        break
                if not found:
                    t["correct"] = letters[0] if t["options"] else "A"
        else:
            t["correct"] = letters[0] if t["options"] else "A"

def normalize_material(material: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure material has expected shapes and limits to avoid UI issues."""
    material = material or {}
    # lesson
    material.setdefault("lesson", {})
    material["lesson"].setdefault("title", "Мини-урок")
    material["lesson"].setdefault("sections", [])
    if not isinstance(material["lesson"]["sections"], list):
        material["lesson"]["sections"] = []

    # cards: limit to 5
    cards = material.get("cards") or []
    if not isinstance(cards, list):
        cards = []
    material["cards"] = cards[:5]

    # tests: normalize + limit to 5
    tests = material.get("tests") or []
    if not isinstance(tests, list):
        tests = []
    tests = tests[:5]
    normalize_tests(tests)
    material["tests"] = tests

    # practice
    material.setdefault("practice", {})
    material["practice"].setdefault("problem", "—")
    material["practice"].setdefault("solution", "—")
    return material

# ----------------- DB helpers (Supabase) — async-safe wrappers -----------------
async def _sb_to_thread(func_desc: str, fn, *args, **kwargs):
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except Exception:
        logger.exception("Supabase error during %s", func_desc)
        raise

def _sb_upsert_user_sync(user_id: int, username: Optional[str]):
    return supabase.table("users").upsert({"id": user_id, "username": username}).execute()

async def ensure_user(user_id: int, username: Optional[str]):
    return await _sb_to_thread("upsert_user", _sb_upsert_user_sync, user_id, username)

def _sb_save_request_sync(user_id: int, topic: str):
    return supabase.table("requests").insert({"user_id": user_id, "topic": topic}).execute()

async def save_request(user_id: int, topic: str):
    return await _sb_to_thread("save_request", _sb_save_request_sync, user_id, topic)

def _sb_get_last_requests_sync(user_id: int, limit: int):
    res = (
        supabase.table("requests")
        .select("topic")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [r.get("topic") for r in (res.data or [])]

async def get_last_requests(user_id: int, limit: int = 3) -> List[str]:
    return await _sb_to_thread("get_last_requests", _sb_get_last_requests_sync, user_id, limit)

def _sb_get_material_sync(topic_hash_val: str):
    res = (
        supabase.table("materials").select("content").eq("topic_hash", topic_hash_val).limit(1).execute()
    )
    if res.data:
        return res.data[0].get("content")
    return None

async def get_material_from_db(topic: str) -> Optional[dict]:
    h = topic_hash(topic)
    return await _sb_to_thread("get_material", _sb_get_material_sync, h)

def _sb_save_material_sync(topic: str, material: dict):
    return (
        supabase.table("materials").insert({
            "topic": normalize_topic(topic),
            "topic_hash": topic_hash(topic),
            "content": material,
        }).execute()
    )

async def save_material_to_db(topic: str, material: dict):
    return await _sb_to_thread("save_material", _sb_save_material_sync, topic, material)

# ----------------- Generation via Hugging Face Inference API -----------------
async def generate_material(topic: str) -> Dict[str, Any]:

    def _sync_call() -> str:
        completion = openai_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(topic)},
            ],
            temperature=0.7,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )

        return completion.choices[0].message.content

    retries = 2
    base_delay = 0.5

    for attempt in range(retries):
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(_sync_call),
                timeout=30
            )

            if not raw:
                logger.warning("Groq returned empty response (attempt %d)", attempt + 1)
                continue

            return safe_json_parse(raw)

        except asyncio.TimeoutError:
            logger.warning("Groq timeout (attempt %d)", attempt + 1)

        except Exception as e:
            logger.exception("Groq generation failed (attempt %d): %s", attempt + 1, e)

        if attempt < retries - 1:
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)

    raise RuntimeError("Groq generation failed after retries")

# ----------------- Formatting -----------------
def escape_html(s: str) -> str:
    return (
        str(s).replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def format_lesson(lesson: dict) -> str:
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
        parts.append("")
    return "\n".join(parts)

def split_text_by_limit(text: str, limit: int = 4000) -> List[str]:
    chunks = []
    current = []
    current_len = 0
    for line in text.splitlines(True):  # keepends
        ln = len(line)
        if current_len + ln > limit and current:
            chunks.append("".join(current))
            current = [line]
            current_len = ln
        else:
            current.append(line)
            current_len += ln
    if current:
        chunks.append("".join(current))
    return chunks or [text[:limit]]

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
async def show_start_screen(target: Message):
    await target.answer(
        "👋 Привет! Я ExplainlyStudy.\nОтправь тему текстом — помогу её изучить.",
        reply_markup=main_menu(),
    )

async def get_material_or_restart(call: CallbackQuery, state: FSMContext) -> Optional[dict]:
    data = await state.get_data()
    material = data.get("material")
    if not material:
        await call.message.answer("⚠️ Сессия устарела. Пожалуйста, выбери тему заново.")
        await show_start_screen(call.message)
        return None
    return material

async def edit_or_send(call: CallbackQuery, text: str, reply_markup: Optional[InlineKeyboardMarkup] = None, parse_mode: Optional[str] = None):
    with suppress(TelegramBadRequest):
        return await call.message.edit_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    return await call.message.answer(text, reply_markup=reply_markup, parse_mode=parse_mode)

# ----------------- Input validation -----------------
def validate_topic(raw: str) -> Optional[str]:
    if not raw:
        return None
    topic = raw.strip()
    if len(topic) < 2:
        return None
    if len(topic) > 200:
        topic = topic[:200]
    return topic

# ----------------- Handlers -----------------
@dp.message(CommandStart())
async def start(message: Message):
    uid = message.from_user.id
    async with await locks.get(uid):
        try:
            await ensure_user(uid, message.from_user.username)
        except Exception:
            logger.exception("ensure_user failed")
        await show_start_screen(message)

@dp.message()
async def handle_message(message: Message, state: FSMContext):
    uid = message.from_user.id
    async with await locks.get(uid):
        if not message.text:
            await message.answer("❗ Я пока понимаю только текстовые темы.")
            return

        topic = validate_topic(message.text)
        if not topic:
            await message.answer("✍️ Напиши тему одним коротким предложением (2–200 символов).")
            return

        try:
            await ensure_user(uid, message.from_user.username)
        except Exception:
            logger.exception("ensure_user failed")
        with suppress(Exception):
            await save_request(uid, topic)

        # проверка в БД
        try:
            material = await get_material_from_db(topic)
        except Exception:
            logger.exception("get_material_from_db failed")
            material = None

        if material:
            await message.answer("📦 Материал загружен из базы.")
        else:
            await message.answer("⏳ Генерирую материал...")
            try:
                material = await generate_material(topic)
            except Exception:
                logger.exception("generate_material failed")
                await message.answer("❌ Не удалось сгенерировать материал. Попробуй переформулировать тему.")
                return
            try:
                await save_material_to_db(topic, material)
            except Exception:
                logger.exception("save_material_to_db failed")

        material = normalize_material(material)

        # сохраняем topic и material в state
        await state.set_state(StudyState.material)
        await state.update_data(material=material, topic=topic)

        await message.answer(f"📌 Тема: {topic}\nВыбери формат обучения:", reply_markup=study_menu())

@dp.callback_query(F.data == "study")
async def open_study(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        try:
            topics = await get_last_requests(uid)
        except Exception:
            logger.exception("get_last_requests failed")
            topics = []
        await state.update_data(last_topics=topics)
        if not topics:
            await edit_or_send(
                call,
                "📘 У тебя пока нет тем. Отправь новую тему текстом.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="➕ Новая тема", callback_data="new_topic")],
                    [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_start")],
                ]),
            )
            return
        await edit_or_send(call, "📘 Выбери тему:", reply_markup=study_topics_kb(topics))

@dp.callback_query(F.data.startswith("topic_idx:"))
async def select_old_topic(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        try:
            idx = int(call.data.replace("topic_idx:", ""))
        except Exception:
            await call.message.answer("❗ Неверный индекс темы.")
            return
        data = await state.get_data()
        last_topics = data.get("last_topics", [])
        if idx < 0 or idx >= len(last_topics):
            await call.message.answer("❗ Неверный индекс темы.")
            return
        topic = last_topics[idx]
        try:
            material = await get_material_from_db(topic)
        except Exception:
            logger.exception("get_material_from_db failed")
            material = None
        if not material:
            await edit_or_send(call, "⏳ Генерирую материал...")
            try:
                material = await generate_material(topic)
                await save_material_to_db(topic, material)
            except Exception:
                logger.exception("generation/save failed")
                await call.message.answer("❌ Ошибка генерации.")
                return
        else:
            await call.message.answer("📦 Материал загружен из базы.")
        material = normalize_material(material)
        await state.set_state(StudyState.material)
        await state.update_data(material=material, topic=topic)
        await call.message.answer(f"📌 Тема: {topic}\nВыбери ф��рмат обучения:", reply_markup=study_menu())

@dp.callback_query(F.data == "new_topic")
async def new_topic(call: CallbackQuery):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        await edit_or_send(call, "✍️ Отправь новую тему текстом.")

@dp.callback_query(F.data == "back_to_start")
async def back_to_start(call: CallbackQuery):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        await edit_or_send(call, "👋 Привет! Я ExplainlyStudy.\nОтправь тему текстом — помогу её изучить.", reply_markup=main_menu())

@dp.callback_query(F.data == "profile")
async def profile(call: CallbackQuery):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        await edit_or_send(call, f"👤 Профиль\nID: {uid}\nСтатус: Free")

# ---------- Форматы ----------
@dp.callback_query(F.data == "lesson")
async def lesson(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        material = await get_material_or_restart(call, state)
        if not material:
            return
        lesson_obj = material.get("lesson", {})
        text = format_lesson(lesson_obj)
        parts = split_text_by_limit(text, limit=3800)  # keep a margin for markup
        # First chunk replaces, others append
        if parts:
            await edit_or_send(
                call,
                parts[0],
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")]]),
            )
        for extra in parts[1:]:
            await call.message.answer(extra, parse_mode="HTML")

@dp.callback_query(F.data == "back_to_formats")
async def back_to_formats(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        data = await state.get_data()
        topic = data.get("topic")
        if not topic:
            await show_start_screen(call.message)
            return
        await edit_or_send(call, f"📌 Тема: {topic}\nВыбери формат обучения:", reply_markup=study_menu())

# ---------- Карточки ----------
@dp.callback_query(F.data == "cards")
async def start_cards(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
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
        await edit_or_send(call, "🎉 Карточки закончились.", reply_markup=finish_lesson_kb())
        return
    card = cards[idx]
    await edit_or_send(
        call,
        f"🧠 Карточка {idx + 1} / {total}\n\n❓ {card.get('question', '—')}",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Показать ответ", callback_data="card_answer")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")],
        ]),
    )

@dp.callback_query(F.data == "card_answer")
async def card_answer(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        data = await state.get_data()
        idx = data.get("card_index", 0)
        cards = data.get("material", {}).get("cards", [])
        if idx < 0 or idx >= len(cards):
            await edit_or_send(call, "❗ Карточка не найдена.")
            return
        answer_text = cards[idx].get("answer", "—")
        await edit_or_send(
            call,
            "✅ Ответ:\n" + answer_text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Следующая", callback_data="card_next")],
                [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")],
            ]),
        )

@dp.callback_query(F.data == "card_next")
async def card_next(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        data = await state.get_data()
        idx = data.get("card_index", 0) + 1
        await state.update_data(card_index=idx)
        await send_card(call, state)

# ---------- Тест ----------
@dp.callback_query(F.data == "test")
async def start_test(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
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
        await state.update_data(accepting_answer=False)
        await edit_or_send(
            call,
            f"🎉 Тест завершён!\nРезультат: {data.get('test_score', 0)} / {total}",
            reply_markup=finish_lesson_kb(),
        )
        return
    t = tests[idx]
    question = t.get("question", "—")
    options = t.get("options", [])
    if isinstance(options, dict):  # safety
        letters = ["A", "B", "C", "D", "E"]
        opts_list = []
        for L in letters:
            if L in options:
                opts_list.append(options[L])
        if not opts_list:
            opts_list = list(options.values())
        options = opts_list
    await state.update_data(accepting_answer=True)
    await edit_or_send(
        call,
        f"📝 Вопрос {idx + 1} из {total}\n\n{question}",
        reply_markup=test_kb(options, idx),
    )

@dp.callback_query(F.data.startswith("answer:"))
async def answer(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
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
        current_idx = data.get("test_index", 0)
        accepting = data.get("accepting_answer", False)
        if not accepting or idx != current_idx:
            # Ignore stale/duplicate clicks
            return
        score = data.get("test_score", 0)
        test = tests[idx]
        correct = get_correct_answer(test)
        if chosen == correct:
            score += 1
            await call.message.answer("✅ Верно!")
        else:
            await call.message.answer(f"❌ Неверно. Правильный ответ: {correct}")
        await state.update_data(test_index=idx + 1, test_score=score, accepting_answer=False)
        await send_test(call, state)

# ---------- Практика ----------
@dp.callback_query(F.data == "practice")
async def practice(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        material = await get_material_or_restart(call, state)
        if not material:
            return
        p = material.get("practice", {})
        problem = p.get("problem", "—")
        await edit_or_send(
            call,
            "🧪 Практика:\n" + problem,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Показать решение", callback_data="solution")],
                [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_formats")],
            ]),
        )

@dp.callback_query(F.data == "solution")
async def solution(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        material = await get_material_or_restart(call, state)
        if not material:
            return
        p = material.get("practice", {})
        await edit_or_send(call, "✅ Решение:\n" + p.get("solution", "—"), reply_markup=finish_lesson_kb())

# ---------- Завершение урока ----------
@dp.callback_query(F.data == "finish_lesson")
async def finish_lesson(call: CallbackQuery, state: FSMContext):
    uid = call.from_user.id
    async with await locks.get(uid):
        await call.answer()
        data = await state.get_data()
        topic = data.get("topic")
        await state.clear()
        if not topic:
            await show_start_screen(call.message)
            return
        await edit_or_send(call, f"Вы завершили тему: {topic}\nЧто дальше?", reply_markup=main_menu())

# ---- простой HTTP-сервер для Render ----
async def run_webserver():
    app = web.Application()

    async def health(request):
        return web.Response(text="ok")

    app.add_routes([web.get("/", health), web.get("/health", health)])
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", os.environ.get("RENDER_PORT", 8000)))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("Web server started on 0.0.0.0:%s", port)
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Web server task cancelled, cleaning up")
        with suppress(Exception):
            await runner.cleanup()
        raise

# ---- объ��диняем polling и http-сервер ----
async def main():
    logger.info("ExplainlyStudy — starting webserver and polling")
    web_task = asyncio.create_task(run_webserver())
    try:
        await dp.start_polling(bot, handle_signals=True)
    except Exception:
        logger.exception("Polling crashed")
        raise
    finally:
        web_task.cancel()
        with suppress(Exception):
            await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())

# Запуск: python bot.py
# Установка зависимостей: pip install -r requirements.txt