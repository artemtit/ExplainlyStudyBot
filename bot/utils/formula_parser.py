import re

MATH_TOKENS = ("^", "/", "sqrt", "frac")
_LATEX_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\$(.+?)\$", re.DOTALL)
_CYRILLIC_RE = re.compile(r"[\u0410-\u042F\u0430-\u044F\u0401\u0451]")


def extract_latex_formulas(text: str) -> list[str]:
    if not text:
        return []
    formulas = [match.group(0) for match in _LATEX_BLOCK_RE.finditer(text)]
    text_without_blocks = _LATEX_BLOCK_RE.sub("", text)
    formulas.extend(match.group(0) for match in _LATEX_INLINE_RE.finditer(text_without_blocks))
    return formulas


def _has_math_tokens(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in MATH_TOKENS)


def wrap_legacy_formulas(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    wrapped_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            wrapped_lines.append(line)
            continue
        if "$" in line:
            wrapped_lines.append(line)
            continue
        if _has_math_tokens(stripped):
            if _CYRILLIC_RE.search(stripped):
                wrapped_lines.append(f"${stripped}$")
            else:
                wrapped_lines.append(f"$${stripped}$$")
        else:
            wrapped_lines.append(line)
    return "\n".join(wrapped_lines)


def normalize_lesson_payload(payload: dict) -> None:
    lesson = payload.get("lesson")
    if not isinstance(lesson, dict):
        return
    sections = lesson.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            text = section.get("text")
            if isinstance(text, str):
                section["text"] = wrap_legacy_formulas(text)
            formula = section.get("formula")
            if isinstance(formula, str):
                section["formula"] = wrap_legacy_formulas(formula)
    lesson_text = lesson.get("text")
    if isinstance(lesson_text, str):
        lesson["text"] = wrap_legacy_formulas(lesson_text)
