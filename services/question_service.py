from __future__ import annotations

import logging
from dataclasses import dataclass

from ai.llm_client import LlmClient
from ai.prompts.loader import PromptLoader


@dataclass
class PracticeQuestion:
    question: str
    answer: str


class QuestionService:
    def __init__(self, llm_client: LlmClient) -> None:
        self._llm_client = llm_client
        self._prompts = PromptLoader()
        self._logger = logging.getLogger(__name__)

    async def generate_questions(
        self,
        topic: str,
        *,
        count: int = 3,
        difficulty: str = "\u0441\u0440\u0435\u0434\u043d\u044f\u044f",
    ) -> list[PracticeQuestion]:
        prompt = self._prompts.load(
            "generate_questions",
            topic=topic,
            count=count,
            difficulty=difficulty,
        )
        raw = await self._llm_client.complete(prompt)
        questions = self._parse_questions(raw, fallback_topic=topic, count=count)
        self._logger.info("Generated %s questions for topic=%s", len(questions), topic)
        return questions

    async def validate_answer(self, *, question: PracticeQuestion, user_answer: str) -> bool:
        expected = self._normalize(question.answer)
        provided = self._normalize(user_answer)
        if expected:
            return expected == provided
        prompt = self._prompts.load(
            "evaluate_answer",
            question=question.question,
            answer=question.answer,
            user_answer=user_answer,
        )
        verdict = await self._llm_client.complete(prompt)
        return verdict.strip().upper().startswith("Y")

    async def get_feedback(self, *, question: PracticeQuestion, user_answer: str) -> str:
        prompt = self._prompts.load(
            "feedback",
            question=question.question,
            answer=question.answer,
            user_answer=user_answer,
        )
        return await self._llm_client.complete(prompt)

    async def generate_hint(self, *, question: PracticeQuestion) -> str:
        prompt = self._prompts.load("hint", question=question.question)
        return await self._llm_client.complete(prompt)

    async def generate_exam_questions(
        self,
        topic: str,
        *,
        count: int = 3,
        difficulty: str = "\u0441\u0440\u0435\u0434\u043d\u044f\u044f",
    ) -> list[PracticeQuestion]:
        prompt = self._prompts.load(
            "exam_questions",
            topic=topic,
            count=count,
            difficulty=difficulty,
        )
        raw = await self._llm_client.complete(prompt)
        return self._parse_questions(raw, fallback_topic=topic, count=count)

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.strip().lower().split())

    @staticmethod
    def _parse_questions(raw: str, *, fallback_topic: str, count: int) -> list[PracticeQuestion]:
        questions: list[PracticeQuestion] = []
        for line in raw.splitlines():
            if "Q:" not in line or "A:" not in line:
                continue
            before, after = line.split("A:", 1)
            question_text = before.split("Q:", 1)[-1].strip(" -|")
            answer_text = after.strip(" -|")
            if question_text:
                questions.append(PracticeQuestion(question=question_text, answer=answer_text))
        if not questions:
            questions = [
                PracticeQuestion(
                    question=f"Кратко объясни: {fallback_topic}",
                    answer="",
                )
            ]
        return questions[: max(count, 1)]
