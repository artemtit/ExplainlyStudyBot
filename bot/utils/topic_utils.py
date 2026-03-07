from __future__ import annotations

import hashlib
import re

_TOPIC_RE = re.compile(r"\s+")


def normalize_topic(topic: str) -> str:
    normalized = _TOPIC_RE.sub(" ", topic.strip().lower())
    return normalized


def topic_hash(topic: str) -> str:
    return hashlib.sha256(normalize_topic(topic).encode("utf-8")).hexdigest()


def validate_topic(raw_text: str | None, min_len: int = 2, max_len: int = 200) -> str | None:
    if raw_text is None:
        return None
    topic = raw_text.strip()
    if len(topic) < min_len:
        return None
    if len(topic) > max_len:
        return None
    return topic
