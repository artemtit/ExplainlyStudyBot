from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from bot.ai.llm_cache import LLMResponseCache
from bot.ai.telemetry import LLMTelemetry
from bot.ai.prompts.base import PromptInstance
from bot.core.ports import LlmClient

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RoutingRule:
    primary_provider: str
    primary_model: str
    fallback_provider: str | None = None
    fallback_model: str | None = None
    second_fallback_provider: str | None = None
    second_fallback_model: str | None = None

    def routes(self, *, override_model: str | None = None) -> list[tuple[str, str]]:
        primary_model = override_model or self.primary_model
        routes: list[tuple[str, str]] = [(self.primary_provider, primary_model)]
        if self.fallback_provider and self.fallback_model:
            routes.append((self.fallback_provider, self.fallback_model))
        if self.second_fallback_provider and self.second_fallback_model:
            routes.append((self.second_fallback_provider, self.second_fallback_model))
        return routes


class LLMRouter:
    """Route LLM requests based on prompt metadata and routing rules."""

    def __init__(
        self,
        *,
        primary: LlmClient,
        fallback: LlmClient | None = None,
        cache: LLMResponseCache | None = None,
        telemetry: LLMTelemetry | None = None,
        rules: dict[str, RoutingRule] | None = None,
        default_model: str | None = None,
        default_fallback_model: str | None = None,
        provider_priority: list[str] | None = None,
        timeout_seconds: int = 40,
        max_retries: int = 2,
        retry_delay: float = 1.5,
    ) -> None:
        self._clients: dict[str, LlmClient] = {"openrouter": primary}
        if fallback is not None:
            self._clients["groq"] = fallback

        self._rules = rules or self._default_rules()
        self._default_model = default_model
        self._default_fallback_model = default_fallback_model
        self._provider_priority = provider_priority or ["openrouter", "groq"]
        self._cache = cache or LLMResponseCache()
        self._telemetry = telemetry or LLMTelemetry()
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    async def generate_json(
        self,
        *,
        prompt: PromptInstance,
        parse_response: Callable[[str], T],
    ) -> T:
        started_at = time.monotonic()
        routes = self._resolve_routes(prompt)
        model_for_cache = routes[0][1] if routes else None

        cached = await self._cache.get(prompt, model=model_for_cache)
        if cached is not None:
            try:
                result = parse_response(cached)
                latency_ms = (time.monotonic() - started_at) * 1000
                self._telemetry.track(
                    prompt_name=prompt.name,
                    prompt_version=prompt.version,
                    prompt_variant=prompt.variant,
                    model=self._telemetry_model(model_for_cache),
                    provider="cache",
                    latency_ms=latency_ms,
                    tokens_input=None,
                    tokens_output=None,
                    cache_hit=True,
                )
                return result
            except Exception:
                logger.warning(
                    "Cached response parse failed, bypassing cache",
                    extra={"prompt": prompt.name, "model": model_for_cache},
                )

        attempts = self._build_attempts(routes)
        last_exc: Exception | None = None

        for provider_name, client, route_model in attempts:
            for attempt in range(self._max_retries + 1):
                try:
                    raw_holder: dict[str, str] = {}
                    usage_holder: dict[str, int] = {}

                    def parse_and_capture(raw: str) -> T:
                        raw_holder["raw"] = raw
                        return parse_response(raw)

                    def capture_usage(usage: dict[str, int] | None) -> None:
                        if usage:
                            usage_holder.update(usage)

                    result = await asyncio.wait_for(
                        client.generate_json(
                            system_prompt=prompt.system,
                            user_prompt=prompt.user,
                            parse_response=parse_and_capture,
                            model=route_model,
                            temperature=prompt.temperature,
                            max_tokens=prompt.max_tokens,
                            usage_collector=capture_usage,
                            provider=provider_name,
                        ),
                        timeout=self._timeout_seconds,
                    )
                    raw = raw_holder.get("raw")
                    if raw is not None:
                        await self._cache.set(prompt, model=route_model, response=raw)
                    latency_ms = (time.monotonic() - started_at) * 1000
                    self._telemetry.track(
                        prompt_name=prompt.name,
                        prompt_version=prompt.version,
                        prompt_variant=prompt.variant,
                        model=self._telemetry_model(route_model),
                        provider=provider_name,
                        latency_ms=latency_ms,
                        tokens_input=usage_holder.get("prompt_tokens"),
                        tokens_output=usage_holder.get("completion_tokens"),
                        cache_hit=False,
                    )
                    return result
                except Exception as exc:
                    last_exc = exc
                    if attempt < self._max_retries and self._is_retryable(exc):
                        logger.warning(
                            "LLM retry %d/%d: prompt=%s model=%s provider=%s error=%s",
                            attempt + 1,
                            self._max_retries,
                            prompt.name,
                            route_model,
                            provider_name,
                            exc,
                        )
                        await asyncio.sleep(self._retry_delay)
                        continue
                    logger.warning(
                        "LLM route failed: prompt=%s model=%s variant=%s error=%s",
                        prompt.name,
                        route_model,
                        prompt.variant,
                        exc,
                    )
                    break

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM routing failed without attempts")

    def _resolve_routes(self, prompt: PromptInstance) -> list[tuple[str, str]]:
        rule = self._rules.get(prompt.name)
        if rule:
            return rule.routes()
        routes: list[tuple[str, str]] = []
        if self._default_model:
            routes.append((self._provider_priority[0], self._default_model))
        if self._default_fallback_model and len(self._provider_priority) > 1:
            routes.append((self._provider_priority[1], self._default_fallback_model))
        return routes

    def _build_attempts(
        self,
        routes: list[tuple[str, str]],
    ) -> list[tuple[str, LlmClient, str | None]]:
        attempts: list[tuple[str, LlmClient, str | None]] = []
        for provider, model in routes:
            client = self._clients.get(provider)
            if client is None:
                continue
            attempts.append((provider, client, model))
        return attempts

    @staticmethod
    def _telemetry_model(model: str | None) -> str | None:
        if model is None:
            return None
        if model.endswith(":free"):
            return model[:-5]
        return model

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
            return True
        status = getattr(exc, "status_code", None)
        if status in {402, 408, 429, 500, 502, 503, 504}:
            return True
        return False

    @staticmethod
    def _default_rules() -> dict[str, RoutingRule]:
        return {
            "lesson_generation": RoutingRule(
                primary_provider="openrouter",
                primary_model="arcee-ai/trinity-large-preview:free",
                fallback_provider="groq",
                fallback_model="llama-3.1-8b-instant",
                second_fallback_provider="groq",
                second_fallback_model="compound-mini",
            ),
            "tests_generation": RoutingRule(
                primary_provider="openrouter",
                primary_model="arcee-ai/trinity-large-preview:free",
                fallback_provider="groq",
                fallback_model="llama-3.1-8b-instant",
                second_fallback_provider="groq",
                second_fallback_model="compound-mini",
            ),
            "flashcards_generation": RoutingRule(
                primary_provider="openrouter",
                primary_model="arcee-ai/trinity-large-preview:free",
                fallback_provider="groq",
                fallback_model="llama-3.1-8b-instant",
                second_fallback_provider="groq",
                second_fallback_model="compound-mini",
            ),
            "test_generation": RoutingRule(
                primary_provider="openrouter",
                primary_model="arcee-ai/trinity-large-preview:free",
                fallback_provider="groq",
                fallback_model="llama-3.1-8b-instant",
                second_fallback_provider="groq",
                second_fallback_model="compound-mini",
            ),
            "card_generation": RoutingRule(
                primary_provider="openrouter",
                primary_model="arcee-ai/trinity-large-preview:free",
                fallback_provider="groq",
                fallback_model="llama-3.1-8b-instant",
                second_fallback_provider="groq",
                second_fallback_model="compound-mini",
            ),
        }


__all__ = ["LLMRouter", "RoutingRule"]
