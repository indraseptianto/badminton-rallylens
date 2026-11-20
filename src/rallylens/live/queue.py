"""Bounded drop-oldest queue: a live pipeline must favour fresh frames over complete ones."""

from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        self._items: deque[T] = deque()
        self._maxsize = maxsize
        self.dropped = 0

    def put(self, item: T) -> bool:
        """Enqueue ``item``; returns True if the oldest item had to be dropped."""
        dropped = len(self._items) >= self._maxsize
        if dropped:
            self._items.popleft()
            self.dropped += 1
        self._items.append(item)
        return dropped

    def get(self) -> T | None:
        return self._items.popleft() if self._items else None

    def __len__(self) -> int:
        return len(self._items)
