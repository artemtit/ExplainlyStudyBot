from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    bot_token: str
    llm_api_key: str
    database_url: str


def load_settings() -> Settings:
    bot_token = os.getenv("BOT_TOKEN", "")
    llm_api_key = os.getenv("LLM_API_KEY", "")
    database_url = os.getenv("DATABASE_URL", "")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN is not set")
    if not llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set")
    return Settings(
        bot_token=bot_token,
        llm_api_key=llm_api_key,
        database_url=database_url,
    )
