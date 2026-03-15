from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from database.models import Answer, Lesson, User


@dataclass
class Repository:
    users: list[User] = field(default_factory=list)
    lessons: list[Lesson] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)

    async def store_user(self, *, user_id: int, username: str | None) -> User:
        user = User(user_id=user_id, username=username, created_at=datetime.utcnow())
        self.users.append(user)
        return user

    async def store_lesson(self, *, user_id: int, topic: str, content: str) -> Lesson:
        lesson = Lesson(user_id=user_id, topic=topic, content=content, created_at=datetime.utcnow())
        self.lessons.append(lesson)
        return lesson

    async def store_answer(
        self,
        *,
        user_id: int,
        question: str,
        answer: str,
        is_correct: bool,
    ) -> Answer:
        record = Answer(
            user_id=user_id,
            question=question,
            answer=answer,
            is_correct=is_correct,
            created_at=datetime.utcnow(),
        )
        self.answers.append(record)
        return record
