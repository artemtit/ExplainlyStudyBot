from __future__ import annotations


class TopicSuggestionService:
    def __init__(self) -> None:
        self._suggestions = [
            "Квадратные уравнения",
            "Теорема Пифагора",
            "Фотосинтез",
            "Дроби",
            "Законы Ньютона",
            "Электричество",
            "Древний Рим",
            "Клеточное строение",
        ]

    def suggest(self, *, limit: int = 5) -> list[str]:
        return self._suggestions[: max(1, limit)]
