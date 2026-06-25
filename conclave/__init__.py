"""Conclave — a multi-agent orchestration framework.

Specialized AI agents that plan, delegate, use tools, critique, and collaborate
to solve complex tasks. Offline-first: with no API key it runs on a deterministic
mock provider, so examples and tests work out of the box. Drop in the Anthropic or
OpenAI adapter (or your own) to go live.

Quick start
-----------
>>> from conclave import Agent, default_tools
>>> agent = Agent("assistant", tools=default_tools())
>>> agent.run("What is 21 * 2?").output  # doctest: +SKIP
'Based on the tool result (42), here is the answer: 42.'
"""

from __future__ import annotations

from .agents import Agent, CriticAgent, PlannerAgent, ReActAgent
from .config import Settings
from .errors import ConclaveError
from .llm import HeuristicProvider, LLMProvider, MockProvider, ScriptedProvider, get_provider
from .memory import ConversationBuffer, SummaryMemory, VectorMemory
from .orchestration import Blackboard, Orchestrator, Router, Team
from .tools import Tool, ToolRegistry, default_tools, tool
from .tracing import Tracer
from .types import AgentResult, Completion, Message, Role, ToolCall, ToolResult
from .version import __version__

__all__ = [
    "__version__",
    # agents
    "Agent",
    "ReActAgent",
    "PlannerAgent",
    "CriticAgent",
    # llm
    "LLMProvider",
    "MockProvider",
    "HeuristicProvider",
    "ScriptedProvider",
    "get_provider",
    # tools
    "Tool",
    "ToolRegistry",
    "tool",
    "default_tools",
    # memory
    "ConversationBuffer",
    "VectorMemory",
    "SummaryMemory",
    # orchestration
    "Orchestrator",
    "Team",
    "Router",
    "Blackboard",
    # infra
    "Tracer",
    "Settings",
    "ConclaveError",
    # types
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "Completion",
    "AgentResult",
]
