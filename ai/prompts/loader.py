from __future__ import annotations

from pathlib import Path


class PromptLoader:
    def __init__(self, *, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path(__file__).parent

    def load(self, name: str, **variables: str | int) -> str:
        filename = name if name.endswith(".txt") else f"{name}.txt"
        path = self._base_dir / filename
        template = path.read_text(encoding="utf-8")
        return template.format(**variables)
