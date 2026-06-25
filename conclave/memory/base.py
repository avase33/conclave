"""Memory abstraction.

A memory stores pieces of text and can recall the most relevant ones for a query.
Different implementations trade recency for relevance for compression.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Memory(ABC):
    @abstractmethod
    def add(self, content: str, **metadata: Any) -> None:
        """Store a piece of text."""

    @abstractmethod
    def recall(self, query: str, k: int = 5) -> list[str]:
        """Return up to ``k`` relevant stored items for ``query``."""

    def clear(self) -> None:  # pragma: no cover - trivial default
        pass
