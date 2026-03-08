from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpacedRepetition:
    min_interval_days: int = 1
    max_interval_days: int = 30

    def next_interval(self, current_interval_days: int, success: bool) -> int:
        if success:
            return min(max(current_interval_days * 2, self.min_interval_days), self.max_interval_days)
        return max(self.min_interval_days, int(current_interval_days / 2) or self.min_interval_days)


__all__ = ["SpacedRepetition"]
