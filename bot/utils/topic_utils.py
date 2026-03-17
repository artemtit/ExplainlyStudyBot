from __future__ import annotations

import hashlib
import re

_TOPIC_RE = re.compile(r"\s+")


def normalize_topic(topic: str) -> str:
    normalized = _TOPIC_RE.sub(" ", topic.strip().lower())
    return normalized


def normalize_difficulty(level: str | None) -> str | None:
    if level is None:
        return None
    level = level.strip().lower()
    return level if level in {"simple", "normal", "hard"} else None


def _material_key(topic: str, difficulty: str | None = None) -> str:
    base = normalize_topic(topic)
    normalized_level = normalize_difficulty(difficulty)
    if normalized_level and normalized_level != "normal":
        return f"{base}|{normalized_level}"
    return base


def topic_hash(topic: str, difficulty: str | None = None) -> str:
    return hashlib.sha256(_material_key(topic, difficulty).encode("utf-8")).hexdigest()


def validate_topic(raw_text: str | None, min_len: int = 2, max_len: int = 200) -> str | None:
    if raw_text is None:
        return None
    topic = raw_text.strip()
    if len(topic) < min_len:
        return None
    if len(topic) > max_len:
        return None
    return topic
