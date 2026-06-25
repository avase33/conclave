"""A shared blackboard for agents to read from and write to.

The blackboard is the simplest possible coordination primitive: a key/value store
with an append-only history of who wrote what. Teams and the orchestrator use it
to pass intermediate results between agents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entry:
    key: str
    value: Any
    author: str
    timestamp: float = field(default_factory=time.time)


class Blackboard:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self.history: list[Entry] = []

    def write(self, key: str, value: Any, author: str = "system") -> None:
        self._data[key] = value
        self.history.append(Entry(key=key, value=value, author=author))

    def read(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def items(self) -> dict[str, Any]:
        return dict(self._data)

    def as_context(self) -> str:
        """Render the current state as text for injection into a prompt."""
        if not self._data:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in self._data.items())

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)
