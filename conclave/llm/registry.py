"""Provider factory.

``get_provider("mock"|"anthropic"|"openai"|"scripted")`` returns a ready
provider, falling back to the offline mock so nothing ever hard-fails for lack
of credentials.
"""

from __future__ import annotations

from ..config import Settings
from .base import LLMProvider
from .mock import HeuristicProvider, MockProvider, ScriptedProvider

__all__ = [
    "LLMProvider",
    "MockProvider",
    "HeuristicProvider",
    "ScriptedProvider",
    "get_provider",
]


def get_provider(name: str | None = None, **kwargs) -> LLMProvider:
    name = (name or Settings.from_env().provider or "mock").lower()

    if name in ("mock", "heuristic"):
        return HeuristicProvider()
    if name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)
    if name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(**kwargs)
    raise ValueError(f"Unknown provider: {name!r}")
