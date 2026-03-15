from __future__ import annotations

from database.repository import Repository


class HistoryService:
    def __init__(self, repository: Repository) -> None:
        self._repository = repository

    async def store_answer(
        self,
        *,
        user_id: int,
        question: str,
        answer: str,
        is_correct: bool,
    ) -> None:
        await self._repository.store_answer(
            user_id=user_id,
            question=question,
            answer=answer,
            is_correct=is_correct,
        )
