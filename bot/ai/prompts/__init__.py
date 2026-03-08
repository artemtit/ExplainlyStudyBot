from __future__ import annotations

from bot.ai.prompts.base import PromptBuilder, PromptInstance, PromptRegistry, PromptTemplate
from bot.ai.prompts.flashcards_prompt import FlashcardsPrompt
from bot.ai.prompts.lesson_prompt import LessonPrompt
from bot.ai.prompts.practice_prompt import PracticePrompt
from bot.ai.prompts.tests_prompt import TestsPrompt
from bot.ai.prompts.registry import PROMPT_REGISTRY


def build_default_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register(LessonPrompt())
    registry.register(TestsPrompt())
    registry.register(FlashcardsPrompt())
    registry.register(PracticePrompt())
    return registry


__all__ = [
    "PromptTemplate",
    "PromptRegistry",
    "PromptBuilder",
    "PromptInstance",
    "LessonPrompt",
    "TestsPrompt",
    "FlashcardsPrompt",
    "PracticePrompt",
    "build_default_registry",
    "PROMPT_REGISTRY",
]
