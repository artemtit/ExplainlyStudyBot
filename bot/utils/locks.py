from __future__ import annotations

import asyncio


class UserLockManager:
    def __init__(self) -> None:
        self._locks: dict[int, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def get(self, user_id: int) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is not None:
            return lock

        async with self._lock:
            lock = self._locks.get(user_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[user_id] = lock
            return lock
