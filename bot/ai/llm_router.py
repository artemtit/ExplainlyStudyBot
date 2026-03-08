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
from bot.utils.json_parser import (
    JsonParseError,
    balance_delimiters,
    clean_trailing_commas,
    extract_json_object,
)

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
        timeout_seconds: int = 180,
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
        effective_max_tokens = self._cap_max_tokens(prompt.max_tokens)

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
            json_retry_used = False
            raw_holder: dict[str, str] = {}
            usage_holder: dict[str, int] = {}
            for attempt in range(self._max_retries + 1):
                raw_holder.clear()
                usage_holder.clear()
                try:
                    def parse_and_capture(raw: str) -> T:
                        raw_holder["raw"] = raw
                        return self._parse_with_recovery(raw, parse_response)

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
                            max_tokens=effective_max_tokens,
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
                except JsonParseError as exc:
                    last_exc = exc
                    if not json_retry_used:
                        json_retry_used = True
                        logger.warning(
                            "LLM json retry 1/1 provider=%s model=%s error=%s",
                            provider_name,
                            route_model,
                            repr(exc),
                        )
                        continue
                    logger.warning(
                        "LLM route failed: prompt=%s model=%s variant=%s error=%s",
                        prompt.name,
                        route_model,
                        prompt.variant,
                        repr(exc),
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < self._max_retries and self._is_retryable(exc):
                        logger.warning(
                            "LLM retry %s/%s provider=%s model=%s error=%s",
                            attempt + 1,
                            self._max_retries,
                            provider_name,
                            route_model,
                            repr(exc),
                        )
                        await asyncio.sleep(self._retry_delay)
                        continue
                    logger.warning(
                        "LLM route failed: prompt=%s model=%s variant=%s error=%s",
                        prompt.name,
                        route_model,
                        prompt.variant,
                        repr(exc),
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
        if isinstance(exc, JsonParseError):
            return False
        status = getattr(exc, "status_code", None)
        if status == 404:
            return False
        if status == 429:
            return True
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
            return True
        name = exc.__class__.__name__.lower()
        if "timeout" in name or "connection" in name or "network" in name:
            return True
        message = str(exc).lower()
        if "model_not_found" in message:
            return False
        return False

    @staticmethod
    def _cap_max_tokens(max_tokens: int | None) -> int:
        if max_tokens is None:
            return 900
        return min(max_tokens, 900)

    @staticmethod
    def _parse_with_recovery(raw: str, parse_response: Callable[[str], T]) -> T:
        try:
            return parse_response(raw)
        except JsonParseError as exc:
            for candidate in LLMRouter._recover_json_candidates(raw):
                try:
                    return parse_response(candidate)
                except JsonParseError:
                    continue
            raise exc

    @staticmethod
    def _recover_json_candidates(raw: str) -> list[str]:
        candidates: list[str] = []
        stripped = raw.replace("```json", "").replace("```", "").strip()

        def add_candidate(value: str) -> None:
            if value.strip():
                candidates.append(value)

        cleaned = clean_trailing_commas(stripped)
        if cleaned.strip() and cleaned != stripped:
            add_candidate(cleaned)
        extracted = extract_json_object(cleaned)
        if extracted and extracted.strip():
            add_candidate(extracted)
            cleaned_extracted = clean_trailing_commas(extracted)
            add_candidate(cleaned_extracted)
            add_candidate(balance_delimiters(cleaned_extracted))
        return candidates

    @staticmethod
    def _default_rules() -> dict[str, RoutingRule]:
        return {
            "lesson_generation": RoutingRule(
                primary_provider="openrouter",
                primary_model="qwen/qwen3.5-flash-02-23",
                fallback_provider="groq",
                fallback_model="meta-llama/llama-3.1-70b-instruct",
            ),
            "tests_generation": RoutingRule(
                primary_provider="openrouter",
                primary_model="qwen/qwen3.5-flash-02-23",
                fallback_provider="groq",
                fallback_model="meta-llama/llama-3.1-70b-instruct",
            ),
            "flashcards_generation": RoutingRule(
                primary_provider="openrouter",
                primary_model="qwen/qwen3.5-flash-02-23",
                fallback_provider="groq",
                fallback_model="meta-llama/llama-3.1-70b-instruct",
            ),
            "test_generation": RoutingRule(
                primary_provider="openrouter",
                primary_model="qwen/qwen3.5-flash-02-23",
                fallback_provider="groq",
                fallback_model="meta-llama/llama-3.1-70b-instruct",
            ),
            "card_generation": RoutingRule(
                primary_provider="openrouter",
                primary_model="qwen/qwen3.5-flash-02-23",
                fallback_provider="groq",
                fallback_model="meta-llama/llama-3.1-70b-instruct",
            ),
        }


__all__ = ["LLMRouter", "RoutingRule"]
