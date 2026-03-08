from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMRequestEvent:
    timestamp: float
    prompt_name: str
    prompt_version: str
    prompt_variant: str
    model: str | None
    provider: str | None
    latency_ms: float
    tokens_input: int | None
    tokens_output: int | None
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "prompt_variant": self.prompt_variant,
            "model": self.model,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "cache_hit": self.cache_hit,
        }


class LLMTelemetry:
    def __init__(self) -> None:
        self._backend_available = True
        try:
            from bot.analytics.metrics import ensure_metrics_server_started

            ensure_metrics_server_started()
        except Exception:
            self._backend_available = False
            logger.warning("Analytics backend unavailable, using logger fallback")

    def emit(self, event: LLMRequestEvent) -> None:
        if self._backend_available:
            try:
                from bot.analytics.metrics import log_json

                log_json(logger, logging.INFO, "llm_request", **event.to_dict())
                return
            except Exception:
                self._backend_available = False
                logger.warning("Analytics backend failed, using logger fallback")

        logger.info("llm_request %s", event.to_dict())

    def track(
        self,
        *,
        prompt_name: str,
        prompt_version: str,
        prompt_variant: str,
        model: str | None,
        provider: str | None,
        latency_ms: float,
        tokens_input: int | None,
        tokens_output: int | None,
        cache_hit: bool,
    ) -> None:
        event = LLMRequestEvent(
            timestamp=time.time(),
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_variant=prompt_variant,
            model=model,
            provider=provider,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cache_hit=cache_hit,
        )
        self.emit(event)


__all__ = ["LLMTelemetry", "LLMRequestEvent"]
