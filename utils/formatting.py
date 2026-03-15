from __future__ import annotations


def format_lesson_text(topic: str, explanation: str) -> str:
    return f"Тема: {topic}\n\n{explanation}"


def format_question_text(question: str, *, index: int | None = None, total: int | None = None) -> str:
    prefix = ""
    if index is not None and total is not None:
        prefix = f"Вопрос {index}/{total}: "
    elif index is not None:
        prefix = f"Вопрос {index}: "
    return f"{prefix}{question}"
