from __future__ import annotations

import json
import logging
import re
from typing import Any


class JsonParseError(ValueError):
    """Raised when JSON cannot be recovered from model output."""


logger = logging.getLogger(__name__)


def strip_code_fences(text: str) -> str:
    payload = text.strip()
    if payload.startswith("```") and payload.endswith("```"):
        lines = payload.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return payload


def clean_json_payload(text: str) -> str:
    clean = text.replace("```json", "").replace("```", "")
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end >= 0 and end >= start:
        clean = clean[start : end + 1]
    return clean.strip()


def clean_trailing_commas(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(text):
        char = text[i]
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        if char == ",":
            j = i + 1
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j < len(text) and text[j] in "]}":
                i += 1
                continue

        result.append(char)
        i += 1

    return "".join(result)


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    in_string = False
    escaped = False
    depth = 0
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:] if depth > 0 else None


def balance_delimiters(text: str) -> str:
    in_string = False
    escaped = False
    braces = 0
    brackets = 0

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            braces += 1
        elif char == "}":
            braces -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1

    if braces > 0:
        text += "}" * braces
    if brackets > 0:
        text += "]" * brackets
    return text


def _attempt_load(text: str) -> dict[str, Any]:
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise JsonParseError("Top-level JSON must be an object")
    return loaded


def safe_json_parse(raw: str) -> dict[str, Any]:
    text = strip_code_fences(raw).strip()
    if not text:
        raise JsonParseError("Empty model response")

    clean = clean_json_payload(text) or text
    logger.debug("Cleaned JSON: %s", clean)

    candidates: list[str] = [clean]
    if clean != text:
        candidates.append(text)
    cleaned = clean_trailing_commas(clean)
    if cleaned != clean:
        candidates.append(cleaned)

    extracted = extract_json_object(cleaned)
    if extracted:
        candidates.append(extracted)
        candidates.append(balance_delimiters(clean_trailing_commas(extracted)))

    # Recover from occasional smart quotes
    if "“" in clean or "”" in clean:
        normalized_quotes = clean.replace("“", '"').replace("”", '"')
        candidates.append(normalized_quotes)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return _attempt_load(candidate)
        except Exception:
            continue

    snippet = re.sub(r"\s+", " ", text)[:220]
    raise JsonParseError(f"Unable to parse JSON from model output: {snippet}")
