from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Callable

from bot.ai.prompts import PROMPT_REGISTRY, PromptBuilder, PromptRegistry
from bot.core.errors import LessonResponseInvalidError
from bot.core.ports import LlmClient
from bot.utils.json_parser import JsonParseError, safe_json_parse

logger = logging.getLogger(__name__)

_MIN_STEPS = 3
_MAX_STEPS = 5


@dataclass(frozen=True, slots=True)
class LearningStep:
    step: int
    question: str
    correct_answer: str
    hint: str
    explanation: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "step": self.step,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "hint": self.hint,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class LearningPlan:
    topic: str
    steps: list[LearningStep]

    def to_dict(self) -> dict[str, object]:
        return {"topic": self.topic, "steps": [step.to_dict() for step in self.steps]}


@dataclass(frozen=True, slots=True)
class SolutionStep:
    step: int
    explanation: str
    formula: str
    result: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "step": self.step,
            "explanation": self.explanation,
            "formula": self.formula,
            "result": self.result,
        }


@dataclass(frozen=True, slots=True)
class SolutionPlan:
    topic: str
    steps: list[SolutionStep]
    final_answer: str

    def to_dict(self) -> dict[str, object]:
        return {
            "topic": self.topic,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
        }


class LessonEngine:
    def __init__(
        self,
        llm_client: LlmClient,
        *,
        prompt_registry: PromptRegistry | None = None,
        max_retries: int = 2,
    ) -> None:
        self._llm = llm_client
        self._registry = prompt_registry or PROMPT_REGISTRY
        self._max_retries = max_retries

    async def generate_steps(self, problem: str) -> LearningPlan:
        prompt = PromptBuilder(self._registry, "tutor_learning").build(problem=problem)
        return await self._generate_with_retry(prompt, lambda payload: self._parse_learning(payload, problem))

    async def generate_solution(self, problem: str) -> SolutionPlan:
        prompt = PromptBuilder(self._registry, "tutor_solution").build(problem=problem)
        return await self._generate_with_retry(prompt, lambda payload: self._parse_solution(payload, problem))

    def check_answer(self, user_answer: str, correct_answer: str) -> bool:
        if not user_answer or not correct_answer:
            return False
        user_parts = self._split_answer(user_answer)
        correct_parts = self._split_answer(correct_answer)
        if user_parts and correct_parts:
            return set(user_parts) == set(correct_parts)
        return self._normalize_answer(user_answer) == self._normalize_answer(correct_answer)

    @staticmethod
    def get_hint(step: LearningStep) -> str:
        return step.hint

    @staticmethod
    def next_step(steps: list[LearningStep], current_step: int) -> LearningStep | None:
        next_index = current_step + 1
        if next_index < len(steps):
            return steps[next_index]
        return None

    async def _generate_with_retry(self, prompt, parse: Callable[[dict], object]):
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            raw = await self._llm.generate_json(system_prompt=prompt.system, user_prompt=prompt.user)
            if not raw or len(raw.strip()) < 300:
                last_error = ValueError("response too short")
                continue
            try:
                payload = safe_json_parse(raw)
            except JsonParseError as exc:
                last_error = exc
                continue
            try:
                return parse(payload)
            except Exception as exc:
                last_error = exc
                continue

        logger.warning("Tutor generation failed after retries", extra={"error": repr(last_error)})
        raise LessonResponseInvalidError("Tutor response failed validation")

    @staticmethod
    def _parse_learning(payload: dict, problem: str) -> LearningPlan:
        steps_raw = payload.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise ValueError("steps missing")
        if len(steps_raw) < _MIN_STEPS:
            raise ValueError("not enough steps")
        steps_raw = steps_raw[:_MAX_STEPS]

        steps: list[LearningStep] = []
        for idx, item in enumerate(steps_raw):
            if not isinstance(item, dict):
                raise ValueError("step is not object")
            step_num = int(item.get("step", idx + 1))
            question = str(item.get("question") or "").strip()
            correct_answer = str(item.get("correct_answer") or "").strip()
            hint = str(item.get("hint") or "").strip()
            explanation = str(item.get("explanation") or "").strip()
            if not (question and correct_answer and hint and explanation):
                raise ValueError("learning step incomplete")
            steps.append(
                LearningStep(
                    step=step_num,
                    question=question,
                    correct_answer=correct_answer,
                    hint=hint,
                    explanation=explanation,
                )
            )

        topic = str(payload.get("topic") or "").strip() or problem
        return LearningPlan(topic=topic, steps=steps)

    @staticmethod
    def _parse_solution(payload: dict, problem: str) -> SolutionPlan:
        steps_raw = payload.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise ValueError("steps missing")
        if len(steps_raw) < _MIN_STEPS:
            raise ValueError("not enough steps")
        steps_raw = steps_raw[:_MAX_STEPS]

        steps: list[SolutionStep] = []
        for idx, item in enumerate(steps_raw):
            if not isinstance(item, dict):
                raise ValueError("step is not object")
            step_num = int(item.get("step", idx + 1))
            explanation = str(item.get("explanation") or "").strip()
            formula = str(item.get("formula") or "").strip()
            result = str(item.get("result") or "").strip()
            if not (explanation and formula and result):
                raise ValueError("solution step incomplete")
            steps.append(
                SolutionStep(
                    step=step_num,
                    explanation=explanation,
                    formula=formula,
                    result=result,
                )
            )

        final_answer = str(payload.get("final_answer") or "").strip()
        if not final_answer:
            raise ValueError("final answer missing")

        topic = str(payload.get("topic") or "").strip() or problem
        return SolutionPlan(topic=topic, steps=steps, final_answer=final_answer)

    @staticmethod
    def _normalize_answer(value: str) -> str:
        return re.sub(r"\s+", "", value.strip().lower())

    @classmethod
    def _split_answer(cls, value: str) -> list[str]:
        if not value:
            return []
        parts = [part for part in re.split(r"[;,]", value) if part.strip()]
        normalized = [cls._normalize_answer(part) for part in parts if cls._normalize_answer(part)]
        return normalized


__all__ = [
    "LessonEngine",
    "LearningPlan",
    "LearningStep",
    "SolutionPlan",
    "SolutionStep",
]
