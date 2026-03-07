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

from bot.utils.llm_limiter import get_global_limiter
from bot.utils.metrics import ensure_metrics_server_started, log_json, record_llm_request

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
        timeout_seconds: int = 120,
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
    ) -> T:
        providers: list[LlmProvider] = [self._openrouter_provider]
        if self._groq_provider is not None:
            providers.append(self._groq_provider)

        # last chance retry on primary provider to handle temporary fallback provider outages
        providers.append(self._openrouter_provider)

        started_at = time.monotonic()
        last_exc: Exception | None = None
        last_provider: str | None = None

        try:
            for idx, provider in enumerate(providers):
                last_provider = provider.name
                attempts = 2 if idx == 0 else 1
                for attempt in range(attempts):
                    try:
                        raw = await self._call_provider(provider, system_prompt, user_prompt)
                        result = parse_response(raw)
                        record_llm_request(duration_seconds=time.monotonic() - started_at, success=True)
                        return result
                    except Exception as exc:
                        last_exc = exc
                        if self._is_permission_denied(exc):
                            logger.warning("Provider %s denied request (403), switching provider", provider.name)
                            break

                        if self._is_retryable(exc):
                            delay = 0.7 * (2 ** attempt)
                            logger.warning(
                                "Provider %s failed attempt %d/%d: %s",
                                provider.name,
                                attempt + 1,
                                attempts,
                                exc,
                            )
                            if attempt + 1 < attempts:
                                await asyncio.sleep(delay)
                                continue
                            break

                        logger.error("Provider %s failed with non-retryable error: %s", provider.name, exc)
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

    async def _call_provider(self, provider: LlmProvider, system_prompt: str, user_prompt: str) -> str:
        logger.info("LLM request provider=%s model=%s", provider.name, provider.model)

        async def _invoke() -> str:
            return await asyncio.wait_for(
                asyncio.to_thread(self._sync_chat_completion, provider, system_prompt, user_prompt),
                timeout=self._timeout_seconds,
            )

        return await self._limiter.run(_invoke)

    @staticmethod
    def _sync_chat_completion(provider: LlmProvider, system_prompt: str, user_prompt: str) -> str:
        response = provider.client.chat.completions.create(
            model=provider.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=8000,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content from model")
        return content

    @staticmethod
    def _is_permission_denied(exc: Exception) -> bool:
        return isinstance(exc, PermissionDeniedError) or getattr(exc, "status_code", None) == 403

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, ValueError)):
            return True
        if isinstance(exc, APIStatusError):
            status = exc.status_code
            return status == 429 or status >= 500
        return False
