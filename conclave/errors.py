"""Exception hierarchy for Conclave."""

from __future__ import annotations


class ConclaveError(Exception):
    """Base class for all Conclave errors."""


class ProviderError(ConclaveError):
    """An LLM provider failed or is misconfigured."""


class ToolError(ConclaveError):
    """A tool failed to execute. Caught by agents and fed back to the model."""


class ToolNotFoundError(ToolError):
    """The model asked for a tool that isn't registered."""


class AgentError(ConclaveError):
    """An agent could not complete its task."""


class MaxStepsExceeded(AgentError):
    """The agent hit its step budget without finishing."""


class OrchestrationError(ConclaveError):
    """A workflow / team could not be executed."""


class ConfigError(ConclaveError):
    """Invalid configuration."""
