from __future__ import annotations

from utils.normalization import normalize_topic


def sanitize_text(text: str) -> str:
    return " ".join(text.strip().split())


def validate_topic(text: str, *, min_len: int = 2, max_len: int = 100) -> str | None:
    cleaned = sanitize_text(text)
    if len(cleaned) < min_len or len(cleaned) > max_len:
        return None
    return normalize_topic(cleaned)
