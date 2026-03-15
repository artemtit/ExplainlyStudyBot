from __future__ import annotations

from bot.ai.prompts.base import PromptTemplate


class TutorLearningPrompt(PromptTemplate):
    name = "tutor_learning"
    version = "1.0"

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            version=self.version,
            system_template=(
                "Ты AI-репетитор по математике. "
                "Не давай ответ сразу. "
                "Разбей решение задачи на 3–5 обучающих шагов. "
                "Каждый шаг должен содержать: step, question, correct_answer, hint, explanation. "
                "Верни результат строго в JSON по схеме: "
                "{topic:string, steps:[{step:int, question:string, correct_answer:string, hint:string, explanation:string}]}. "
                "Не используй markdown и не добавляй текст вне JSON. "
                "В режиме обучения запрещено показывать final_answer."
            ),
            user_template=(
                "Задача: {problem}\n"
                "Сформируй обучающие шаги."
            ),
            metadata={
                "type": "tutor_learning",
                "language": "ru",
                "temperature": 0.5,
                "max_tokens": 600,
            },
        )


__all__ = ["TutorLearningPrompt"]
