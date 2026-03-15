from __future__ import annotations

from utils.normalization import normalize_topic


def sanitize_text(text: str) -> str:
    return " ".join(text.strip().split())


def validate_topic(text: str, *, min_len: int = 2, max_len: int = 100) -> str | None:
    cleaned = sanitize_text(text)
    if len(cleaned) < min_len or len(cleaned) > max_len:
        return None
    return normalize_topic(cleaned)


_DIFFICULTY_ALIASES = {
    "easy": "easy",
    "simple": "easy",
    "\u043b\u0435\u0433\u043a\u043e": "easy",
    "\u043f\u0440\u043e\u0441\u0442\u043e": "easy",
    "\u043f\u0440\u043e\u0441\u0442\u0430\u044f": "easy",
    "normal": "normal",
    "medium": "normal",
    "\u0441\u0440\u0435\u0434\u043d\u0435": "normal",
    "\u0441\u0440\u0435\u0434\u043d\u044f\u044f": "normal",
    "\u043e\u0431\u044b\u0447\u043d\u043e": "normal",
    "hard": "hard",
    "\u0441\u043b\u043e\u0436\u043d\u043e": "hard",
    "\u0441\u043b\u043e\u0436\u043d\u0430\u044f": "hard",
}


def extract_difficulty_and_topic(args: str, *, default: str = "normal") -> tuple[str, str]:
    cleaned = sanitize_text(args)
    if not cleaned:
        return default, ""
    parts = cleaned.split(" ", 1)
    if len(parts) == 1:
        return default, cleaned
    maybe_level = _DIFFICULTY_ALIASES.get(parts[0].lower())
    if maybe_level:
        return maybe_level, parts[1].strip()
    return default, cleaned
