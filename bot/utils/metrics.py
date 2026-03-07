from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from prometheus_client import Counter, Histogram, start_http_server

DEFAULT_METRICS_PORT = 8001

_llm_requests_total = Counter(
    "llm_requests_total",
    "Total number of LLM requests",
    ["status"],
)
_llm_request_latency = Histogram(
    "llm_request_latency_seconds",
    "LLM request latency in seconds",
)
_redis_cache_hits = Counter(
    "redis_cache_hits_total",
    "Redis cache hits",
)
_redis_cache_misses = Counter(
    "redis_cache_misses_total",
    "Redis cache misses",
)
_redis_errors = Counter(
    "redis_errors_total",
    "Redis errors",
)

_metrics_lock = threading.Lock()
_metrics_started = False


def ensure_metrics_server_started() -> None:
    global _metrics_started
    if _metrics_started:
        return
    with _metrics_lock:
        if _metrics_started:
            return
        port = _load_metrics_port()
        start_http_server(port)
        _metrics_started = True


def _load_metrics_port() -> int:
    raw = os.getenv("METRICS_PORT", "").strip()
    if not raw:
        return DEFAULT_METRICS_PORT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_METRICS_PORT
    return value if value > 0 else DEFAULT_METRICS_PORT


def record_llm_request(*, duration_seconds: float, success: bool) -> None:
    status = "success" if success else "error"
    _llm_requests_total.labels(status=status).inc()
    _llm_request_latency.observe(duration_seconds)


def inc_cache_hit() -> None:
    _redis_cache_hits.inc()


def inc_cache_miss() -> None:
    _redis_cache_misses.inc()


def inc_redis_error() -> None:
    _redis_errors.inc()


def log_json(logger, level: int, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields, "ts": time.time()}
    logger.log(level, json.dumps(payload, ensure_ascii=False))
