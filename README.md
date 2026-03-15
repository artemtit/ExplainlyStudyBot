# ExplainlyStudyBot
Telegram-бот для быстрого обучения: генерирует карточки, тесты, мини-урок и практическое задание по любой теме с помощью LLM.

**Быстрый старт**
1. `python -m venv venv`
2. `.\venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. Скопируйте `.env.example` в `.env` и заполните значения.
5. `python bot.py`

**Переменные окружения**
- `TELEGRAM_TOKEN` — токен Telegram-бота (обязателен).
- `SUPABASE_URL` — URL проекта Supabase (обязателен).
- `SUPABASE_KEY` — сервисный ключ Supabase (обязателен).
- `GROQ_API_KEY` — ключ Groq для LLM (опционально, без него генерация отключена).
- `GROQ_MODEL` — модель Groq (по умолчанию `groq/compound-mini`).
- `SUPPORT_URL` — ссылка на поддержку.
- `LOG_LEVEL` — уровень логирования (по умолчанию `INFO`).
- `FREE_TIER_NOTICE` — показывать уведомление о лимитах (`1` или `0`).
- `GENERATION_TIMEOUT_SECONDS` — таймаут LLM-запросов в секундах.
- `MATERIAL_CACHE_TTL_SECONDS` — TTL кэша материалов.
- `REDIS_URL` — URL Redis (опционально).
- `LLM_MAX_CONCURRENT` — глобальный лимит параллельных LLM-запросов.
- `METRICS_PORT` — порт метрик Prometheus (по умолчанию `8001`).
- `PORT` — порт health-check сервера (по умолчанию `8000`).

**Тесты**
- `python -m pytest`

**План разработки**
- См. `PROMPTS.md` для последовательных шагов.
