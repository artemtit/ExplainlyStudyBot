from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from supabase import Client

logger = logging.getLogger(__name__)


class SupabaseRepository:
    def __init__(self, client: Client) -> None:
        self._client = client

    async def _to_thread(self, label: str, fn, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return await asyncio.to_thread(fn, *args, **kwargs)
            except httpx.HTTPError as exc:
                last_exc = exc
                delay = 0.5 * (2**attempt)
                logger.warning("Supabase transient failure [%s], attempt %d/3: %s", label, attempt + 1, exc)
                await asyncio.sleep(delay)
            except Exception:
                logger.exception("Supabase call failed [%s]", label)
                raise

        logger.error("Supabase failed after retries [%s]: %s", label, last_exc)
        raise RuntimeError(f"Supabase request failed: {label}") from last_exc


__all__ = ["SupabaseRepository"]
