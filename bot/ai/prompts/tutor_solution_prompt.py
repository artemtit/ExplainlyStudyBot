from __future__ import annotations

from bot.ai.prompts.base import PromptTemplate


class TutorSolutionPrompt(PromptTemplate):
    name = "tutor_solution"
    version = "1.0"

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            version=self.version,
            system_template=(
                "Ты AI-репетитор по математике. "
                "Дай решение задачи и объясни каждый шаг. "
                "Разбей решение на 3–5 шагов. "
                "Каждый шаг должен содержать: step, explanation, formula, result. "
                "Верни результат строго в JSON по схеме: "
                "{topic:string, steps:[{step:int, explanation:string, formula:string, result:string}], final_answer:string}. "
                "Не используй markdown и не добавляй текст вне JSON."
            ),
            user_template=(
                "Задача: {problem}\n"
                "Сформируй решение с объяснением."
            ),
            metadata={
                "type": "tutor_solution",
                "language": "ru",
                "temperature": 0.4,
                "max_tokens": 700,
            },
        )


__all__ = ["TutorSolutionPrompt"]
