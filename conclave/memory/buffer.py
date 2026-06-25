"""A simple recency buffer."""

from __future__ import annotations

from collections import deque
from typing import Any

from .base import Memory, MemoryRecord


class ConversationBuffer(Memory):
    """Keeps the most recent ``max_items`` records; recall returns the newest."""

    def __init__(self, max_items: int = 50):
        self.max_items = max_items
        self._items: deque[MemoryRecord] = deque(maxlen=max_items)

    def add(self, content: str, **metadata: Any) -> None:
        self._items.append(MemoryRecord(content=content, metadata=metadata))

    def recall(self, query: str, k: int = 5) -> list[str]:
        items = list(self._items)[-k:]
        return [r.content for r in reversed(items)]

    def all(self) -> list[str]:
        return [r.content for r in self._items]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
