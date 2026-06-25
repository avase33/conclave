"""The provider abstraction.

Every backend — mock, Anthropic, OpenAI — implements :class:`LLMProvider`.
Agents only ever talk to this interface, so swapping models is a one-line change
and the entire framework is testable with the offline mock provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import Completion, Message


class LLMProvider(ABC):
    """Abstract LLM backend.

    Implementations turn a list of :class:`~conclave.types.Message` (and optional
    tool schemas) into a single :class:`~conclave.types.Completion`.
    """

    name: str = "base"
    default_model: str = ""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> Completion:
        """Return a single completion for the given conversation."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
