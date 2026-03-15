from __future__ import annotations


_CANONICAL_TOPICS = {
    "quadratic equations": "Квадратные уравнения",
    "pythagorean theorem": "Теорема Пифагора",
    "photosynthesis": "Фотосинтез",
}


def normalize_topic(topic: str) -> str:
    normalized = " ".join(topic.strip().lower().split())
    return _CANONICAL_TOPICS.get(normalized, topic.strip())
