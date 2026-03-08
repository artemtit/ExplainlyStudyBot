from __future__ import annotations

from bot.ai.prompts.base import PromptRegistry
from bot.ai.prompts.lesson_prompt import LessonPrompt
from bot.ai.prompts.tests_prompt import TestsPrompt
from bot.ai.prompts.flashcards_prompt import FlashcardsPrompt
from bot.ai.prompts.practice_prompt import PracticePrompt

def build_default_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register(LessonPrompt())
    registry.register(TestsPrompt())
    registry.register(FlashcardsPrompt())
    registry.register(PracticePrompt())
    return registry

PROMPT_REGISTRY = build_default_registry()

__all__ = ["PROMPT_REGISTRY"]
