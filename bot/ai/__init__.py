from __future__ import annotations

from bot.ai.content_generator import AiContentGenerator
from bot.ai.llm_client import LlmProvider, OpenAIService
from bot.ai.llm_router import LLMRouter
from bot.ai.prompts import (
    FlashcardsPrompt,
    LessonPrompt,
    PracticePrompt,
    PromptBuilder,
    PromptInstance,
    PromptRegistry,
    PromptTemplate,
    TestsPrompt,
    build_default_registry,
)
from bot.ai.telemetry import LLMTelemetry, LLMRequestEvent

__all__ = [
    "AiContentGenerator",
    "OpenAIService",
    "LlmProvider",
    "LLMRouter",
    "LLMTelemetry",
    "LLMRequestEvent",
    "PromptTemplate",
    "PromptRegistry",
    "PromptBuilder",
    "PromptInstance",
    "LessonPrompt",
    "TestsPrompt",
    "FlashcardsPrompt",
    "PracticePrompt",
    "build_default_registry",
]
