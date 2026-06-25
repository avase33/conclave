"""Runtime configuration, resolved from explicit args then environment.

Conclave is offline-first: with no configuration at all it uses the deterministic
mock provider so everything runs and tests pass without network or API keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    provider: str = "mock"
    model: str = ""
    temperature: float = 0.7
    max_steps: int = 8
    request_timeout: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        """Auto-detect a provider from the environment.

        Preference order: explicit CONCLAVE_PROVIDER, then any available API key,
        otherwise the offline mock.
        """

        provider = os.environ.get("CONCLAVE_PROVIDER")
        if not provider:
            if os.environ.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "mock"

        return cls(
            provider=provider,
            model=os.environ.get("CONCLAVE_MODEL", ""),
            temperature=float(os.environ.get("CONCLAVE_TEMPERATURE", "0.7")),
            max_steps=int(os.environ.get("CONCLAVE_MAX_STEPS", "8")),
            request_timeout=int(os.environ.get("CONCLAVE_TIMEOUT", "60")),
        )
