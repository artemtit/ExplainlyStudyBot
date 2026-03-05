from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from bot.utils.strings import HEADER_TEMPLATE, STAGE_TITLES

STAGE_ORDER = ["lesson", "cards", "test", "practice"]

SUPERSCRIPTS = {
    "0": "\u2070",
    "1": "\u00b9",
    "2": "\u00b2",
    "3": "\u00b3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079",
}


def escape_html(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_stage_header(topic: str, stage_id: str) -> str:
    title = STAGE_TITLES.get(stage_id, stage_id)
    total = len(STAGE_ORDER)
    try:
        index = STAGE_ORDER.index(stage_id)
    except ValueError:
        index = 0
    safe_topic = escape_html(topic or "\u2014")
    return HEADER_TEMPLATE.format(topic=safe_topic, stage=title, pos=index + 1, total=total)


def next_stage_id(stage_id: str) -> str | None:
    try:
        index = STAGE_ORDER.index(stage_id)
    except ValueError:
        return None
    next_index = index + 1
    if next_index >= len(STAGE_ORDER):
        return None
    return STAGE_ORDER[next_index]


@dataclass(slots=True)
class FractionNode:
    numerator: list[object]
    denominator: list[object]


def _read_braced(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        return "", start
    depth = 0
    out: list[str] = []
    idx = start
    while idx < len(text):
        ch = text[idx]
        if ch == "{":
            depth += 1
            if depth > 1:
                out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out), idx + 1
            out.append(ch)
        else:
            out.append(ch)
        idx += 1
    return "".join(out), idx


def parse_formula_tokens(text: str) -> list[object]:
    tokens: list[object] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    i = 0
    while i < len(text):
        if text.startswith("\\frac", i):
            flush()
            i += len("\\frac")
            num, i = _read_braced(text, i)
            den, i = _read_braced(text, i)
            tokens.append(FractionNode(parse_formula_tokens(num), parse_formula_tokens(den)))
            continue
        if text.startswith("\\sqrt", i):
            i += len("\\sqrt")
            inner, i = _read_braced(text, i)
            buf.append(f"√({inner})")
            continue
        if text.startswith("\\pm", i):
            buf.append("±")
            i += len("\\pm")
            continue
        if text.startswith("\\cdot", i):
            buf.append("×")
            i += len("\\cdot")
            continue
        if text[i] == "\\":
            i += 1
            continue
        buf.append(text[i])
        i += 1

    flush()
    return tokens


def normalize_text_token(token: str) -> str:
    normalized = token.replace("{", "").replace("}", "")
    for op in ("=", "+", "-", "±", "×", "÷"):
        normalized = normalized.replace(op, f" {op} ")
    normalized = " ".join(normalized.split())
    return normalized


def convert_superscripts(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "^" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt.isdigit():
                out.append(SUPERSCRIPTS.get(nxt, nxt))
                i += 2
                continue
            if nxt == "(" and i + 3 < len(text) and text[i + 3] == ")":
                mid = text[i + 2]
                if mid.isdigit():
                    out.append(SUPERSCRIPTS.get(mid, mid))
                    i += 4
                    continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _box_width(lines: list[str]) -> int:
    return max((len(x) for x in lines), default=0)


def _pad(lines: list[str], width: int) -> list[str]:
    return [x.ljust(width) for x in lines]


def _vpad(lines: list[str], height: int) -> list[str]:
    width = _box_width(lines)
    padded = _pad(lines, width)
    extra = height - len(padded)
    top = extra // 2
    bottom = extra - top
    return ([" " * width] * top) + padded + ([" " * width] * bottom)


def _box_hstack(parts: list[list[str]], sep: str = " ") -> list[str]:
    if not parts:
        return [""]
    height = max(len(part) for part in parts)
    aligned = [_vpad(part, height) for part in parts]
    out: list[str] = []
    for row in range(height):
        out.append(sep.join(part[row] for part in aligned).rstrip())
    return out


def _box_fraction(num: list[str], den: list[str]) -> list[str]:
    width = max(_box_width(num), _box_width(den), 1)
    num_box = [line.center(width) for line in _pad(num, width)]
    den_box = [line.center(width) for line in _pad(den, width)]
    bar = "─" * width
    return num_box + [bar] + den_box


def _tokens_to_box(tokens: list[object]) -> list[str]:
    chunks: list[list[str]] = []
    for token in tokens:
        if isinstance(token, FractionNode):
            chunks.append(_box_fraction(_tokens_to_box(token.numerator), _tokens_to_box(token.denominator)))
            continue
        if isinstance(token, str):
            text = convert_superscripts(normalize_text_token(token))
            if text:
                chunks.append([text])
    return _box_hstack(chunks)


def _contains_fraction(tokens: Iterable[object]) -> bool:
    for token in tokens:
        if isinstance(token, FractionNode):
            return True
    return False


def format_formula(formula: str) -> tuple[str, bool]:
    tokens = parse_formula_tokens(str(formula))
    if _contains_fraction(tokens):
        return "\n".join(_tokens_to_box(tokens)), True

    parts: list[str] = []
    for token in tokens:
        if isinstance(token, str):
            normalized = convert_superscripts(normalize_text_token(token))
            if normalized:
                parts.append(normalized)
    return " ".join(parts).strip(), False


def format_lesson(lesson: dict) -> str:
    title = lesson.get("title") or "\u041c\u0438\u043d\u0438-\u0443\u0440\u043e\u043a"
    sections = lesson.get("sections")
    output: list[str] = [f"<b>{escape_html(title)}</b>"]

    if not isinstance(sections, list) or not sections:
        fallback = lesson.get("text") or lesson.get("content") or "\u041c\u0438\u043d\u0438-\u0443\u0440\u043e\u043a \u043d\u0435 \u0441\u0433\u0435\u043d\u0435\u0440\u0438\u0440\u043e\u0432\u0430\u043b\u0441\u044f."
        output.append(escape_html(fallback))
        return "\n\n".join(output)

    for section in sections:
        if not isinstance(section, dict):
            output.append(escape_html(section))
            continue

        block: list[str] = []
        header = section.get("header")
        if header:
            block.append(f"<b>{escape_html(header)}</b>")

        text = section.get("text")
        if text:
            block.append(escape_html(text))

        points = section.get("key_points")
        if isinstance(points, list):
            for point in points:
                block.append(f"\u2022 {escape_html(point)}")

        formula = section.get("formula")
        if formula:
            formatted_formula, multiline = format_formula(str(formula))
            if multiline:
                block.append(f"\U0001F4D0 \u0424\u043e\u0440\u043c\u0443\u043b\u0430:\n<pre>{escape_html(formatted_formula)}</pre>")
            else:
                block.append(f"\U0001F4D0 \u0424\u043e\u0440\u043c\u0443\u043b\u0430: {escape_html(formatted_formula)}")

        if block:
            output.append("\n".join(block))

    return "\n\n".join(output)


def split_text_by_limit(text: str, limit: int = 3900) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in text.splitlines(keepends=True):
        if len(line) > limit:
            if current:
                chunks.append("".join(current))
                current = []
                current_len = 0
            for i in range(0, len(line), limit):
                chunks.append(line[i : i + limit])
            continue

        if current_len + len(line) > limit:
            chunks.append("".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)

    if current:
        chunks.append("".join(current))
    return chunks
