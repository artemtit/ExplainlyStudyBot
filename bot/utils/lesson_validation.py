from __future__ import annotations

from dataclasses import dataclass
import re
import textwrap

_BAD_PHRASES = [
    "попробуйте сформулировать вопрос иначе",
    "не могу ответить",
    "не относится",
    "я не могу помочь",
]

_REFUSAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bне могу\b",
        r"\bне способен\b",
        r"\bне в состоянии\b",
        r"\bне имею доступа\b",
        r"\bкак (?:ии|модель)\b",
        r"\bизвините\b",
        r"\bне предназначен\b",
        r"\bне предоставляю\b",
        r"\bне могу помочь\b",
        r"\bне могу ответить\b",
        r"\bне относится\b",
    ]
]

_FORMULA_HINTS = [
    "$",
    "\\frac",
    "\\sqrt",
    "=",
    "^",
    "√",
    "±",
    "∫",
    "∞",
    "≈",
    "≠",
    "≤",
    "≥",
]

_MATH_KEYWORDS = [
    "алгебр",
    "геометр",
    "тригоном",
    "уравнен",
    "функц",
    "логарифм",
    "интеграл",
    "предел",
    "производн",
    "матриц",
    "вектор",
    "дроб",
    "процент",
    "степен",
    "корен",
    "синус",
    "косинус",
    "тангенс",
    "sin",
    "cos",
    "tan",
    "арифмет",
    "комбинатор",
    "вероятн",
    "статист",
]

_SECTION_HEADERS = {
    "краткое объяснение",
    "формулы",
    "пример",
}

_TOPIC_RE = re.compile(r"^\s*тема\s*[:\-]\s*(.+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LessonRenderData:
    topic: str
    explanation: str
    formulas: str
    example: str


def _normalize_header(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[:\s]+$", "", cleaned)
    return cleaned


def _looks_like_formula(line: str) -> bool:
    lowered = line.lower()
    if "$" in line:
        return True
    return any(token in lowered for token in _FORMULA_HINTS)


def wrap_text(text: str, width: int) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if _looks_like_formula(line):
            lines.append(line)
            continue
        wrapped = textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if wrapped:
            lines.extend(wrapped)
        else:
            lines.append("")
    return "\n".join(lines)


def _section_text(section: dict, *, fallback: str = "") -> str:
    text = str(section.get("text") or "").strip()
    if text:
        return text
    points = section.get("key_points")
    if isinstance(points, list) and points:
        joined = ". ".join([str(point).strip() for point in points if str(point).strip()])
        if joined:
            return joined
    return fallback


def build_lesson_render_data(topic: str, lesson: dict) -> LessonRenderData | None:
    sections = lesson.get("sections") if isinstance(lesson, dict) else None
    if not isinstance(sections, list) or not sections:
        return None

    explanation = ""
    example = ""
    formulas: list[str] = []
    found_explanation = False
    found_formulas = False
    found_example = False

    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        header_raw = str(section.get("header") or "").strip()
        header = _normalize_header(header_raw)
        text = _section_text(section)
        formula = section.get("formula")
        if formula is not None:
            formula_text = str(formula).strip()
            if formula_text:
                formulas.append(formula_text)

        if not explanation and any(key in header for key in ("кратк", "объяс", "обзор", "определ")):
            explanation = text
        if any(key in header for key in ("кратк", "объяс", "обзор", "определ")):
            found_explanation = True

        if not example and any(key in header for key in ("пример", "задач", "примен", "иллюстр")):
            example = text
        if any(key in header for key in ("пример", "задач", "примен", "иллюстр")):
            found_example = True

        if "формул" in header:
            found_formulas = True
            if not formulas and text and _looks_like_formula(text):
                formulas.append(text)

        if not example and "например" in text.lower():
            example = text

    if not (found_explanation and found_formulas and found_example):
        return None

    if not explanation or not example:
        return None

    formulas_text = "\n".join(formulas).strip()
    if not formulas_text:
        formulas_text = "-"

    return LessonRenderData(
        topic=str(topic).strip(),
        explanation=explanation,
        formulas=formulas_text,
        example=example,
    )


def lesson_render_data_to_text(data: LessonRenderData) -> str:
    return (
        f"Тема: {data.topic}\n"
        "Краткое объяснение\n"
        f"{data.explanation}\n\n"
        "Формулы\n"
        f"{data.formulas}\n\n"
        "Пример\n"
        f"{data.example}"
    )


def _extract_topic(text: str) -> str:
    for line in text.splitlines():
        match = _TOPIC_RE.match(line)
        if match:
            return match.group(1).strip()
    return ""


def _is_math_topic(text: str) -> bool:
    topic = _extract_topic(text)
    haystack = topic.lower() if topic else text.lower()
    return any(keyword in haystack for keyword in _MATH_KEYWORDS)


def _contains_math_symbols(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("\\frac", "\\sqrt", "√", "^", "±", "∫", "∞", "≈", "≠", "≤", "≥", "$", "=", "+", "−")):
        return True
    return re.search(r"\d\s*[+\-*/^]\s*\d", lowered) is not None


def _has_required_structure(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines()]
    indices: dict[str, int] = {}

    for idx, line in enumerate(lines):
        if not line:
            continue
        norm = _normalize_header(line)
        if norm in _SECTION_HEADERS and norm not in indices:
            indices[norm] = idx

    if len(indices) != len(_SECTION_HEADERS):
        return False

    try:
        idx_expl = indices["краткое объяснение"]
        idx_form = indices["формулы"]
        idx_ex = indices["пример"]
    except KeyError:
        return False

    if not (idx_expl < idx_form < idx_ex):
        return False

    explanation = "\n".join(lines[idx_expl + 1 : idx_form]).strip()
    formulas = "\n".join(lines[idx_form + 1 : idx_ex]).strip()
    example = "\n".join(lines[idx_ex + 1 :]).strip()

    return bool(explanation and formulas and example)


def validate_lesson_response(text: str) -> bool:
    if not text:
        return False
    cleaned = text.strip()
    if len(cleaned) < 300:
        return False

    lowered = cleaned.lower()
    for phrase in _BAD_PHRASES:
        if phrase in lowered:
            return False

    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(lowered):
            return False

    if not _has_required_structure(cleaned):
        return False

    if _is_math_topic(cleaned) and not _contains_math_symbols(cleaned):
        return False

    return True


__all__ = [
    "LessonRenderData",
    "build_lesson_render_data",
    "lesson_render_data_to_text",
    "validate_lesson_response",
    "wrap_text",
]
