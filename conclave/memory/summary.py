"""Summarizing memory.

Keeps a rolling natural-language summary of everything added. When the buffer of
un-summarized content grows past a threshold, it asks an LLM provider to compress
it into the running summary. With the offline mock provider this still works
(the mock returns a deterministic summary), so it degrades gracefully.
"""

from __future__ import annotations

from typing import Any

from ..llm.base import LLMProvider
from ..types import Message
from .base import Memory


class SummaryMemory(Memory):
    def __init__(self, provider: LLMProvider, max_chars: int = 2000):
        self.provider = provider
        self.max_chars = max_chars
        self.summary = ""
        self._pending: list[str] = []

    def add(self, content: str, **metadata: Any) -> None:
        self._pending.append(content)
        if sum(len(p) for p in self._pending) >= self.max_chars:
            self._compress()

    def _compress(self) -> None:
        joined = "\n".join(self._pending)
        prompt = (
            "Update the running summary so it captures the key facts and decisions. "
            "Keep it under 200 words.\n\n"
            f"Current summary:\n{self.summary or '(none)'}\n\nNew content:\n{joined}"
        )
        completion = self.provider.complete([Message.user(prompt)])
        self.summary = completion.text.strip() or self.summary
        self._pending.clear()

    def recall(self, query: str, k: int = 5) -> list[str]:
        if self._pending:
            self._compress()
        return [self.summary] if self.summary else []

    def clear(self) -> None:
        self.summary = ""
        self._pending.clear()
