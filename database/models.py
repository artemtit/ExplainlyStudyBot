from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class User:
    user_id: int
    username: str | None
    created_at: datetime


@dataclass
class Lesson:
    user_id: int
    topic: str
    content: str
    created_at: datetime


@dataclass
class Answer:
    user_id: int
    question: str
    answer: str
    is_correct: bool
    created_at: datetime


TABLES = ("users", "lessons", "answers")
