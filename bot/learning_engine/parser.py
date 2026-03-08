from __future__ import annotations

from typing import Any, Mapping

from bot.core.errors import MaterialValidationError
from bot.core.models import Material, QuizQuestion
from bot.utils.json_parser import JsonParseError, safe_json_parse


class MaterialPayloadParser:
    @staticmethod
    def parse(payload: dict[str, Any] | str, *, topic: str | None = None) -> Material:
        data = payload
        if isinstance(payload, str):
            data = safe_json_parse(payload)
        if not isinstance(data, Mapping):
            raise MaterialValidationError("Material payload must be an object")
        return Material.from_payload(data, topic=topic)

    @staticmethod
    def parse_tests(payload: Any, *, topic: str | None = None) -> list[QuizQuestion]:
        if isinstance(payload, Mapping):
            payload = payload.get("tests")
        return QuizQuestion.list_from_raw(payload, topic=topic)


__all__ = ["MaterialPayloadParser", "JsonParseError", "MaterialValidationError"]
