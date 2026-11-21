"""Bounded drop-oldest queue: a live pipeline must favour fresh frames over complete ones."""

from __future__ import annotations

import threading
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        self._items: deque[T] = deque()
        self._cond = threading.Condition()
        self._maxsize = maxsize
        self.dropped = 0

    def put(self, item: T) -> bool:
        """Enqueue ``item``; returns True if the oldest item had to be dropped."""
        with self._cond:
            dropped = len(self._items) >= self._maxsize
            if dropped:
                self._items.popleft()
                self.dropped += 1
            self._items.append(item)
            self._cond.notify()
            return dropped

    def get(self, timeout: float | None = None) -> T | None:
        """Pop the oldest item; waits up to ``timeout`` s (None = do not wait)."""
        with self._cond:
            if not self._items and timeout:
                self._cond.wait(timeout)
            return self._items.popleft() if self._items else None

    def __len__(self) -> int:
        return len(self._items)
