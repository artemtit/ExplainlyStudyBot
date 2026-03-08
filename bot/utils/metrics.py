from __future__ import annotations

from bot.analytics.metrics import (  # re-export for backward compatibility
    DEFAULT_METRICS_PORT,
    ensure_metrics_server_started,
    inc_cache_hit,
    inc_cache_miss,
    inc_redis_error,
    log_json,
    record_llm_request,
)

__all__ = [
    "DEFAULT_METRICS_PORT",
    "ensure_metrics_server_started",
    "inc_cache_hit",
    "inc_cache_miss",
    "inc_redis_error",
    "log_json",
    "record_llm_request",
]
