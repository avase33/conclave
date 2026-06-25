"""Core data structures shared across Conclave.

These are deliberately small, immutable-ish dataclasses with no behaviour beyond
serialization. Keeping the data model tiny and explicit makes the rest of the
framework (providers, agents, tools, tracing) easy to reason about and test.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def new_id(prefix: str = "id") -> str:
    """Short, sortable-ish unique id used for messages, tool calls, spans."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A request, emitted by the model, to invoke a tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call"))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolResult:
    """The outcome of running a tool."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
            "is_error": self.is_error,
        }


@dataclass
class Message:
    """A single turn in a conversation."""

    role: Role
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    id: str = field(default_factory=lambda: new_id("msg"))
    created_at: float = field(default_factory=time.time)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str, name: str | None = None) -> "Message":
        return cls(role=Role.USER, content=content, name=name)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, result: ToolResult) -> "Message":
        return cls(
            role=Role.TOOL,
            content=result.content,
            name=result.name,
            tool_call_id=result.tool_call_id,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data


@dataclass
class Usage:
    """Token accounting, when a provider reports it."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass
class Completion:
    """A single response from an LLM provider."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = "stop"

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class AgentResult:
    """What an agent returns after working a task."""

    output: str
    success: bool = True
    agent: str = ""
    steps: int = 0
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "output": self.output,
            "success": self.success,
            "steps": self.steps,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "metadata": self.metadata,
        }
