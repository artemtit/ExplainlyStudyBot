# bot.py
# Рабочая версия бота, совместимая с aiogram v3.x и openai 2.x
# Python 3.10+
# Запуск: в активированном venv -> python bot.py

import os
import asyncio
import json
from typing import Any, Dict, List

from dotenv import load_dotenv

# OpenAI 2.x client
from openai import OpenAI

# aiogram v3
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram.client.default import DefaultBotProperties

# локальные промпты
from prompts import build_system_prompt, build_user_prompt

# Загрузка переменных окружения из .env
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # можно поменять в .env
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1200"))

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не найден в .env")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не найден в .env")

# Инициализация OpenAI клиента (2.x)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Инициализация бота и диспетчера (aiogram v3)
bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties()
)
dp = Dispatcher()

# ----- Вспомогательные функции ----- #

async def call_openai(messages: List[Dict[str, str]], model: str = OPENAI_MODEL, temperature: float = 0.2) -> str:
    """
    Вызывает OpenAI Chat Completions (через синхронный клиент в to_thread).
    Возвращает строку с текстом ответа модели.
    """
    def sync_call():
        # Используем client.chat.completions.create для OpenAI 2.x
        resp = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=OPENAI_MAX_TOKENS
        )
        # Попытка извлечь контент. Формат ответа: resp.choices[0].message.content
        try:
            return resp.choices[0].message.content
        except Exception:
            # В некоторых вариантах SDK контент может быть в другой структуре
            # Приводим весь объект к str как запасной вариант
            return str(resp)

    return await asyncio.to_thread(sync_call)

def parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Попытка строго распарсить JSON из ответа модели.
    Если в тексте есть лишние символы вокруг JSON — извлекаем первую {...}.
    Выбрасывает ValueError при неудаче.
    """
    text = text.strip()
    # Прямая попытка
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Поиск границ JSON-объекта
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            fragment = text[start:end+1]
            try:
                return json.loads(fragment)
            except json.JSONDecodeError as e:
                raise ValueError(f"Не удалось распарсить JSON: {e}") from e
        raise ValueError("Не удалось найти JSON в ответе модели.")

def format_cards(cards: List[Dict[str, str]]) -> str:
    lines = ["УЧЕБНЫЕ КАРТОЧКИ (вопрос — короткий ответ):"]
    for i, c in enumerate(cards, 1):
        q = c.get("question", "").strip()
        a = c.get("answer", "").strip()
        lines.append(f"{i}. Вопрос: {q}\n   Короткий ответ: {a}")
    return "\n\n".join(lines)

def format_tests(tests: List[Dict[str, Any]]) -> str:
    lines = ["ТЕСТОВЫЕ ВОПРОСЫ (A–D). В конце указан правильный вариант:"]
    for i, t in enumerate(tests, 1):
        q = t.get("question", "").strip()
        options = t.get("options", []) or []
        correct = t.get("correct", "").strip()
        lines.append(f"{i}. {q}")
        letters = ["A", "B", "C", "D"]
        for idx, letter in enumerate(letters):
            opt = options[idx].strip() if idx < len(options) else ""
            lines.append(f"   {letter}) {opt}")
        if correct:
            lines.append(f"   Правильный вариант: {correct}")
    return "\n".join(lines)

def format_lesson(lesson: str) -> str:
    return f"МИНИ-УРОК:\n{lesson.strip()}"


def format_practice(practice: Dict[str, str]) -> str:
    problem = practice.get("problem", "").strip()
    solution = practice.get("solution", "").strip()
    return f"ПРАКТИЧЕСКОЕ ЗАДАНИЕ:\n{problem}\n\nРЕШЕНИЕ:\n{solution}"


# ----- Генерация материалов через LLM ----- #

async def generate_learning_material(topic: str) -> Dict[str, Any]:
    """
    Собирает промпты, вызывает LLM и возвращает готовую структуру с материалами.
    """
    system = build_system_prompt()
    user = build_user_prompt(topic)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    raw = await call_openai(messages)
    try:
        parsed = parse_json_strict(raw)
    except ValueError as e:
        # Если не получилось распарсить — возвращаем детальную ошибку
        raise RuntimeError(f"Ошибка парсинга ответа LLM: {e}\n\nТекст ответа модели:\n{raw}")

    # Базовая валидация структуры (MVP)
    if "cards" not in parsed or "tests" not in parsed or "lesson" not in parsed or "practice" not in parsed:
        raise RuntimeError("LLM вернула JSON, но отсутствуют обязательные поля (cards, tests, lesson, practice).")
    return parsed

# ----- Обработчики ботa (aiogram v3: декораторы через @dp.message(...)) ----- #

@dp.message(CommandStart())
async def cmd_start(message: Message):
    """
    Обработчик /start
    """
    text = (
        "Привет. Отправь тему (одно предложение или короткий текст), и бот сгенерирует:\n"
        "1) 5 учебных карточек (вопрос + короткий ответ);\n"
        "2) 5 тестовых вопросов с вариантами A–D и правильным вариантом;\n"
        "3) короткий мини-урок (3–5 абзацев простым языком);\n"
        "4) одно практическое задание с решением.\n\n"
        "Ответ будет на языке входного текста."
    )
    await message.answer(text)

@dp.message(Command("help"))
async def cmd_help(message: Message):
    """
    Обработчик /help — переадресует к стартовому тексту
    """
    await cmd_start(message)

@dp.message()
async def handle_topic(message: Message):
    """
    Обработчик обычного текста — основная логика: принимает тему и возвращает материалы.
    Игнорируем команды, чтобы не конфликтовать с фильтрами.
    """
    text = (message.text or "").strip()
    if not text:
        await message.answer("Тема пустая. Отправьте короткое предложение или фразу — тему для объяснения.")
        return
    # Игнорируем если текст — команда (начинается с '/'), чтобы команды обрабатывались отдельно
    if text.startswith("/"):
        return

    # Информируем пользователя
    await message.answer("Генерирую материалы. Это может занять 5–20 секунд...")

    try:
        data = await generate_learning_material(text)
    except Exception as e:
        await message.answer(f"Ошибка при генерации материалов: {e}")
        return

    # Формируем части ответа
    parts = []
    lesson = data.get("lesson", "")
    if lesson:
        parts.append(format_lesson(lesson))
    cards = data.get("cards", [])
    if cards:
        parts.append(format_cards(cards))
    tests = data.get("tests", [])
    if tests:
        parts.append(format_tests(tests))
    practice = data.get("practice", {})
    if practice:
        parts.append(format_practice(practice))

    # Отправляем по частям, разбивая длинные сообщения
    for part in parts:
        max_len = 3800
        if len(part) <= max_len:
            await message.answer(part)
        else:
            paragraphs = part.split("\n\n")
            buffer = ""
            for p in paragraphs:
                chunk = (p + "\n\n")
                if len(buffer) + len(chunk) > max_len:
                    if buffer:
                        await message.answer(buffer)
                    buffer = chunk
                else:
                    buffer += chunk
            if buffer.strip():
                await message.answer(buffer)

# ----- Запуск бота ----- #

async def main():
    try:
        print("Бот запускается...")
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
