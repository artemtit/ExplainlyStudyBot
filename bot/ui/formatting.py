from __future__ import annotations

from typing import Any, Iterable

SEPARATOR = "\u2501" * 14
BULLET = "\u2022"


def _section(title: str, body: str | None) -> str:
    if not body:
        return ""
    return f"{title}\n{body}"


def _join_sections(sections: Iterable[str]) -> str:
    return "\n\n".join([section for section in sections if section])


def format_lesson(topic: str, lesson: dict[str, Any]) -> str:
    title = f"{SEPARATOR}\n\U0001F4D8 \u0422\u0435\u043C\u0430: {topic}\n{SEPARATOR}"
    sections = lesson.get("sections") if isinstance(lesson, dict) else None
    if not isinstance(sections, list):
        sections = []

    rendered: list[str] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        header = str(section.get("header") or "\u0420\u0430\u0437\u0434\u0435\u043b")
        text = str(section.get("text") or "").strip()
        formula = section.get("formula")
        key_points = section.get("key_points") if isinstance(section.get("key_points"), list) else []

        block_parts = [f"{header}\n{text}" if text else header]
        if formula:
            block_parts.append(_section("\u0424\u043E\u0440\u043C\u0443\u043B\u0430", str(formula)))
        if key_points:
            bullets = "\n".join([f"{BULLET} {str(point)}" for point in key_points])
            block_parts.append(_section("\u041A\u043B\u044E\u0447\u0435\u0432\u044B\u0435 \u0438\u0434\u0435\u0438", bullets))

        rendered.append(_join_sections(block_parts))

    content = _join_sections(rendered) if rendered else "\u0423\u0440\u043E\u043A \u0433\u043E\u0442\u043E\u0432."
    return f"{title}\n\n{content}"


def format_flashcard(question: str, answer: str | None, index: int, total: int, *, show_answer: bool) -> str:
    title = f"{SEPARATOR}\n\U0001F9E0 \u0424\u043B\u044D\u0448\u043A\u0430\u0440\u0442\u0430 {index}/{total}\n{SEPARATOR}"
    if show_answer:
        body = _section("\u041E\u0442\u0432\u0435\u0442", answer or "\u2014")
    else:
        body = _section("\u0412\u043E\u043F\u0440\u043E\u0441", question or "\u2014")
    return f"{title}\n\n{body}"


def format_test_question(topic: str, question: str, index: int, total: int) -> str:
    title = f"{SEPARATOR}\n\U0001F9EA \u0422\u0435\u0441\u0442: {topic}\n{SEPARATOR}"
    header = f"\u0412\u043E\u043F\u0440\u043E\u0441 {index}/{total}"
    return f"{title}\n\n{header}\n{question}"


def format_test_result(correct: bool, explanation: str, *, correct_letter: str | None = None) -> str:
    title = f"{SEPARATOR}\n\U0001F9EA \u0420\u0435\u0437\u0443\u043B\u044C\u0442\u0430\u0442\n{SEPARATOR}"
    status = "\u2705 \u0412\u0435\u0440\u043D\u043E" if correct else "\u274C \u041D\u0435\u0432\u0435\u0440\u043D\u043E"
    parts = [status]
    if not correct and correct_letter:
        parts.append(f"\u041F\u0440\u0430\u0432\u0438\u043B\u044C\u043D\u044B\u0439 \u043E\u0442\u0432\u0435\u0442: {correct_letter}")
    if explanation:
        parts.append(_section("\u041E\u0431\u044A\u044F\u0441\u043D\u0435\u043D\u0438\u0435", explanation))
    return f"{title}\n\n{_join_sections(parts)}"


def format_progress(stats: dict[str, Any]) -> str:
    title = f"{SEPARATOR}\n\U0001F4CA \u041F\u0440\u043E\u0433\u0440\u0435\u0441\u0441\n{SEPARATOR}"
    topics = stats.get("topics_learned", 0)
    tests = stats.get("tests_passed", 0)
    cards = stats.get("flashcards_reviewed", 0)
    streak = stats.get("daily_streak", 0)
    last_topic = stats.get("last_topic")
    last_stage = stats.get("last_stage")
    stage_map = {
        "lesson": "\u0423\u0440\u043e\u043a",
        "flashcards": "\u041a\u0430\u0440\u0442\u043e\u0447\u043a\u0438",
        "test": "\u0422\u0435\u0441\u0442",
        "practice": "\u041f\u0440\u0430\u043a\u0442\u0438\u043a\u0430",
        "done": "\u0417\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u043e",
    }
    body_lines = [
        f"\U0001F4D8 \u0422\u0435\u043C \u0438\u0437\u0443\u0447\u0435\u043D\u043E: {topics}",
        f"\U0001F9EA \u0422\u0435\u0441\u0442\u043E\u0432 \u043F\u0440\u043E\u0439\u0434\u0435\u043D\u043E: {tests}",
        f"\U0001F9E0 \u0424\u043B\u044D\u0448\u043A\u0430\u0440\u0442 \u043F\u0440\u043E\u0441\u043C\u043E\u0442\u0440\u0435\u043D\u043E: {cards}",
        f"\U0001F525 \u0414\u043D\u0435\u0432\u043D\u043E\u0439 \u0441\u0442\u0440\u0438\u043A: {streak}",
    ]
    if last_topic:
        body_lines.append(f"\U0001F4CC \u041F\u043E\u0441\u043B\u0435\u0434\u043D\u044F\u044F \u0442\u0435\u043C\u0430: {last_topic}")
    if last_stage:
        stage_label = stage_map.get(str(last_stage), str(last_stage))
        body_lines.append(f"\U0001F9ED \u041F\u043E\u0441\u043B\u0435\u0434\u043D\u0438\u0439 \u044D\u0442\u0430\u043F: {stage_label}")
    body = "\n".join(body_lines)
    return f"{title}\n\n{body}"


def format_practice(problem: str, solution: str | None, *, show_solution: bool) -> str:
    title = f"{SEPARATOR}\n\U0001F9E9 \u041F\u0440\u0430\u043A\u0442\u0438\u043A\u0430\n{SEPARATOR}"
    if show_solution:
        body = _section("\u0420\u0435\u0448\u0435\u043D\u0438\u0435", solution or "\u2014")
    else:
        body = _section("\u0417\u0430\u0434\u0430\u0447\u0430", problem or "\u2014")
    return f"{title}\n\n{body}"


def format_topic_prompt() -> str:
    return f"{SEPARATOR}\n\U0001F4DA \u041D\u043E\u0432\u0430\u044F \u0442\u0435\u043C\u0430\n{SEPARATOR}\n\n\u041D\u0430\u043F\u0438\u0448\u0438\u0442\u0435 \u0442\u0435\u043C\u0443 \u043E\u0434\u043D\u0438\u043C \u043A\u043E\u0440\u043E\u0442\u043A\u0438\u043C \u043F\u0440\u0435\u0434\u043B\u043E\u0436\u0435\u043D\u0438\u0435\u043C."


def format_topic_too_short(min_len: int) -> str:
    return (
        f"{SEPARATOR}\n\u26A0 \u0421\u043B\u0438\u0448\u043A\u043E\u043C \u043A\u043E\u0440\u043E\u0442\u043A\u0430\u044F \u0442\u0435\u043C\u0430\n{SEPARATOR}\n\n"
        f"\u041C\u0438\u043D\u0438\u043C\u0443\u043C {min_len} \u0441\u0438\u043C\u0432\u043E\u043B\u0430. \u041D\u0430\u043F\u0438\u0448\u0438\u0442\u0435 \u0442\u0435\u043C\u0443 \u0431\u043E\u043B\u0435\u0435 \u043F\u043E\u0434\u0440\u043E\u0431\u043D\u043E."
    )


def format_topic_too_long(max_len: int) -> str:
    return (
        f"{SEPARATOR}\n\u26A0 \u0421\u043B\u0438\u0448\u043A\u043E\u043C \u0434\u043B\u0438\u043D\u043D\u0430\u044F \u0442\u0435\u043C\u0430\n{SEPARATOR}\n\n"
        f"\u041C\u0430\u043A\u0441\u0438\u043C\u0443\u043C {max_len} \u0441\u0438\u043C\u0432\u043E\u043B\u043E\u0432. \u0421\u043E\u043A\u0440\u0430\u0442\u0438\u0442\u0435 \u0437\u0430\u043F\u0440\u043E\u0441."
    )


def format_recent_topics(topics: list[str]) -> str:
    rendered = "\n".join([f"{BULLET} {topic}" for topic in topics if topic])
    return (
        f"{SEPARATOR}\n\U0001F4D8 \u041D\u0435\u0434\u0430\u0432\u043D\u0438\u0435 \u0442\u0435\u043C\u044B\n{SEPARATOR}\n\n"
        f"{rendered}\n\n"
        "\u0412\u044B\u0431\u0435\u0440\u0438 \u0442\u0435\u043C\u0443 \u043D\u0438\u0436\u0435 \u0438\u043B\u0438 \u043D\u0430\u043F\u0438\u0448\u0438 \u0441\u0432\u043E\u044E."
    )


def format_explanation_prompt(topic: str) -> str:
    safe_topic = topic or "\u2014"
    return (
        f"{SEPARATOR}\n\U0001F9E0 \u0423\u0440\u043E\u0432\u0435\u043D\u044C \u043E\u0431\u044A\u044F\u0441\u043D\u0435\u043D\u0438\u044F\n{SEPARATOR}\n\n"
        f"\u0422\u0435\u043C\u0430: {safe_topic}\n\u0412\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u0443\u0440\u043E\u0432\u0435\u043D\u044C \u043E\u0431\u044A\u044F\u0441\u043D\u0435\u043D\u0438\u044F."
    )


def format_no_resume() -> str:
    return f"{SEPARATOR}\n\U0001F525 \u041F\u0440\u043E\u0434\u043E\u043B\u0436\u0438\u0442\u044C\n{SEPARATOR}\n\n\u041F\u043E\u043A\u0430 \u043D\u0435\u0442 \u0430\u043A\u0442\u0438\u0432\u043D\u043E\u0439 \u0442\u0435\u043C\u044B. \u041D\u0430\u0447\u043D\u0438\u0442\u0435 \u043E\u0431\u0443\u0447\u0435\u043D\u0438\u0435."


def format_settings() -> str:
    return f"{SEPARATOR}\n\u2699 \u041D\u0430\u0441\u0442\u0440\u043E\u0439\u043A\u0438\n{SEPARATOR}\n\n\u0412\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u0434\u0435\u0439\u0441\u0442\u0432\u0438\u0435:"
