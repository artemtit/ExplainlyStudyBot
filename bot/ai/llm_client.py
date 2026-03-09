from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from bot.analytics.metrics import ensure_metrics_server_started, log_json, record_llm_request
from bot.utils.llm_limiter import get_global_limiter

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class LlmProvider:
    name: str
    model: str
    client: OpenAI


class OpenAIService:
    def __init__(
        self,
        *,
        openrouter_api_key: str,
        openrouter_model: str,
        groq_api_key: str | None,
        groq_model: str,
        timeout_seconds: int = 180,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._limiter = get_global_limiter()
        ensure_metrics_server_started()

        self._openrouter_provider = LlmProvider(
            name="openrouter",
            model=openrouter_model,
            client=OpenAI(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/explainly-study-bot",
                    "X-Title": "ExplainlyStudyBot",
                },
            ),
        )

        self._groq_provider: LlmProvider | None = None
        if groq_api_key:
            self._groq_provider = LlmProvider(
                name="groq",
                model=groq_model,
                client=OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1"),
            )

        if self._groq_provider is None:
            logger.warning("Groq fallback disabled: GROQ_API_KEY is not configured")

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        parse_response: Callable[[str], T],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        usage_collector: Callable[[dict[str, int] | None], None] | None = None,
        provider: str | None = None,
    ) -> T:
        providers = self._select_providers(provider)

        started_at = time.monotonic()
        last_exc: Exception | None = None
        last_provider: str | None = None

        try:
            for idx, provider in enumerate(providers):
                last_provider = provider.name
                attempts = 2 if idx == 0 else 1
                for attempt in range(attempts):
                    try:
                        raw, usage = await self._call_provider(
                            provider,
                            system_prompt,
                            user_prompt,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        result = parse_response(raw)
                        if usage_collector is not None:
                            usage_collector(usage)
                        record_llm_request(duration_seconds=time.monotonic() - started_at, success=True)
                        return result
                    except Exception as exc:
                        last_exc = exc
                        if self._is_permission_denied(exc):
                            logger.warning("Provider %s denied request (403), switching provider", provider.name)
                            break

                        if self._is_retryable(exc):
                            delay = 0.7 * (2**attempt)
                            logger.warning(
                                "Provider %s failed attempt %d/%d: %s",
                                provider.name,
                                attempt + 1,
                                attempts,
                                repr(exc),
                            )
                            if attempt + 1 < attempts:
                                await asyncio.sleep(delay)
                                continue
                            break

                        logger.error("Provider %s failed with non-retryable error: %s", provider.name, repr(exc))
                        break

            if last_exc is None:
                raise RuntimeError("LLM request failed without exception")
            raise last_exc
        except Exception as exc:
            record_llm_request(duration_seconds=time.monotonic() - started_at, success=False)
            log_json(
                logger,
                logging.ERROR,
                "llm_error",
                provider=last_provider,
                error_type=exc.__class__.__name__,
                error=str(exc),
            )
            if exc.__class__.__name__ == "MaterialValidationError":
                log_json(
                    logger,
                    logging.ERROR,
                    "material_validation_failed",
                    provider=last_provider,
                    error=str(exc),
                )
            raise

    def supports_provider(self, provider: str) -> bool:
        provider = provider.strip().lower()
        if provider == "openrouter":
            return True
        if provider == "groq":
            return self._groq_provider is not None
        return False

    async def _call_provider(
        self,
        provider: LlmProvider,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, dict[str, int] | None]:
        effective_model = model or provider.model
        logger.info("LLM request provider=%s model=%s", provider.name, effective_model)

        async def _invoke() -> str:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._sync_chat_completion,
                    provider,
                    system_prompt,
                    user_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=self._timeout_seconds,
            )

        return await self._limiter.run(_invoke)

    @staticmethod
    def _sync_chat_completion(
        provider: LlmProvider,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, dict[str, int] | None]:
        response = provider.client.chat.completions.create(
            model=model or provider.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature if temperature is not None else 0.7,
            max_tokens=max_tokens if max_tokens is not None else 2000,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content from model")
        usage_payload = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_payload = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
        return content, usage_payload

    @staticmethod
    def _is_permission_denied(exc: Exception) -> bool:
        return isinstance(exc, PermissionDeniedError) or getattr(exc, "status_code", None) == 403

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            status = exc.status_code
            return status == 429
        return False

    def _select_providers(self, provider: str | None) -> list[LlmProvider]:
        if provider is None:
            providers: list[LlmProvider] = [self._openrouter_provider]
            if self._groq_provider is not None:
                providers.append(self._groq_provider)
            return providers

        provider = provider.strip().lower()
        if provider == "openrouter":
            return [self._openrouter_provider]
        if provider == "groq":
            if self._groq_provider is None:
                raise RuntimeError("Groq provider is not configured")
            return [self._groq_provider]
        raise ValueError(f"Unknown provider: {provider}")


__all__ = ["OpenAIService", "LlmProvider"]
