from __future__ import annotations

from dataclasses import dataclass

from ai.llm_client import LlmClient


@dataclass
class PracticeQuestion:
    question: str
    answer: str


class QuestionService:
    def __init__(self, llm_client: LlmClient) -> None:
        self._llm_client = llm_client

    async def generate_questions(self, topic: str, *, count: int = 3) -> list[PracticeQuestion]:
        prompt = (
            "Сгенерируй практические вопросы по теме.\n"
            "Формат: Q: <вопрос> | A: <краткий ответ>\n"
            f"Тема: {topic}\n"
            f"Количество: {count}"
        )
        raw = await self._llm_client.complete(prompt)
        return self._parse_questions(raw, fallback_topic=topic, count=count)

    async def validate_answer(self, *, question: PracticeQuestion, user_answer: str) -> bool:
        expected = self._normalize(question.answer)
        provided = self._normalize(user_answer)
        if expected:
            return expected == provided
        prompt = (
            "Проверь, правильный ли ответ пользователя.\n"
            f"Вопрос: {question.question}\n"
            f"Ответ пользователя: {user_answer}\n"
            "Ответь одним словом: YES или NO."
        )
        verdict = await self._llm_client.complete(prompt)
        return verdict.strip().upper().startswith("Y")

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
