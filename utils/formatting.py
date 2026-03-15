from __future__ import annotations


def format_lesson_text(
    topic: str,
    explanation: str,
    *,
    examples: list[str] | None = None,
    summary: list[str] | None = None,
) -> str:
    text = f"*Тема:* {topic}\n\n{explanation}"
    if examples:
        examples_block = "\n".join(f"• {item}" for item in examples)
        text = f"{text}\n\n*Примеры:*\n{examples_block}"
    if summary:
        summary_block = "\n".join(f"• {item}" for item in summary)
        text = f"{text}\n\n*Резюме:*\n{summary_block}"
    return text


def format_question_text(question: str, *, index: int | None = None, total: int | None = None) -> str:
    prefix = ""
    if index is not None and total is not None:
        prefix = f"Вопрос {index}/{total}: "
    elif index is not None:
        prefix = f"Вопрос {index}: "
    return f"{prefix}{question}"


def format_questions_block(questions: list[str]) -> str:
    if not questions:
        return ""
    lines = "\n".join(f"{idx + 1}. {question}" for idx, question in enumerate(questions))
    return f"*Практика (3 вопроса):*\n{lines}"


def format_suggestions_text(suggestions: list[str]) -> str:
    if not suggestions:
        return "Пока нет рекомендаций тем."
    lines = "\n".join(f"• {topic}" for topic in suggestions)
    return f"*Рекомендуемые темы:*\n{lines}"
