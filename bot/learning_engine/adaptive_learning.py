from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptiveLearning:
    easy_threshold: float = 0.8
    hard_threshold: float = 0.5

    def pick_difficulty(self, score: int, total: int) -> str:
        if total <= 0:
            return "medium"
        ratio = score / total
        if ratio >= self.easy_threshold:
            return "hard"
        if ratio <= self.hard_threshold:
            return "easy"
        return "medium"


__all__ = ["AdaptiveLearning"]
