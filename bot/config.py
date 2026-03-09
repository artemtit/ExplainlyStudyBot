from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    supabase_url: str
    supabase_key: str
    openrouter_api_key: str
    groq_api_key: str | None
    openrouter_model: str
    groq_model: str
    support_url: str
    log_level: str
    free_tier_notice: bool
    generation_timeout_seconds: int
    material_cache_ttl_seconds: int


def load_settings() -> Settings:
    load_dotenv()

    telegram_token = os.getenv("TELEGRAM_TOKEN", "").strip()
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip() or None

    required = {
        "TELEGRAM_TOKEN": telegram_token,
        "SUPABASE_URL": supabase_url,
        "SUPABASE_KEY": supabase_key,
        "OPENROUTER_API_KEY": openrouter_api_key,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing required env vars: {missing_text}")

    return Settings(
        telegram_token=telegram_token,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        openrouter_api_key=openrouter_api_key,
        groq_api_key=groq_api_key,
        openrouter_model=os.getenv("OPENROUTER_MODEL", "qwen/qwen3.5-flash-02-23").strip(),
        groq_model=os.getenv("GROQ_MODEL", "mixtral-8x7b-32768").strip(),
        support_url=os.getenv("SUPPORT_URL", "https://t.me/ligr5").strip(),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        free_tier_notice=os.getenv("FREE_TIER_NOTICE", "1").strip() != "0",
        generation_timeout_seconds=int(os.getenv("GENERATION_TIMEOUT_SECONDS", "180").strip()),
        material_cache_ttl_seconds=int(os.getenv("MATERIAL_CACHE_TTL_SECONDS", "3600").strip()),
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
