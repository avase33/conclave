"""The Agent: a provider + instructions + tools + memory, with a tool-use loop.

``Agent.run(task)`` drives the standard structured loop:

1. Build the conversation (system instructions, optional recalled memory, task).
2. Ask the provider for a completion, passing any available tool schemas.
3. If the model requested tools, execute them, append the results, and repeat.
4. Otherwise return the final answer.

The loop is bounded by ``max_steps`` and fully instrumented through the tracer.
"""

from __future__ import annotations

from typing import Any

from ..errors import MaxStepsExceeded
from ..llm.base import LLMProvider
from ..llm.registry import get_provider
from ..memory.base import Memory
from ..tools.base import Tool, ToolRegistry
from ..tracing.events import EventType
from ..tracing.tracer import Tracer
from ..types import AgentResult, Message, Usage


def _as_registry(tools: ToolRegistry | list[Tool] | None) -> ToolRegistry | None:
    if tools is None:
        return None
    if isinstance(tools, ToolRegistry):
        return tools
    return ToolRegistry(list(tools))


class Agent:
    """A single autonomous worker."""

    def __init__(
        self,
        name: str,
        instructions: str = "You are a helpful assistant.",
        *,
        provider: LLMProvider | None = None,
        tools: ToolRegistry | list[Tool] | None = None,
        memory: Memory | None = None,
        max_steps: int = 8,
        temperature: float | None = None,
        tracer: Tracer | None = None,
        capabilities: list[str] | None = None,
    ):
        self.name = name
        self.instructions = instructions
        self.provider = provider or get_provider()
        self.tools = _as_registry(tools)
        self.memory = memory
        self.max_steps = max_steps
        self.temperature = temperature
        self.tracer = tracer or Tracer()
        self.capabilities = capabilities or []

    # -- prompt construction -------------------------------------------------
    def _system_message(self) -> Message:
        text = self.instructions
        if self.tools:
            text += (
                "\n\nYou can call tools to help. Available tools: "
                + ", ".join(self.tools.names())
                + "."
            )
        return Message.system(text)

    def _seed_messages(self, task: str, context: str | None) -> list[Message]:
        messages = [self._system_message()]
        if self.memory:
            recalled = self.memory.recall(task, k=3)
            if recalled:
                messages.append(Message.system("Relevant context:\n" + "\n".join(recalled)))
        if context:
            messages.append(Message.system("Additional context:\n" + context))
        messages.append(Message.user(task))
        return messages

    # -- main loop -----------------------------------------------------------
    def run(self, task: str, context: str | None = None) -> AgentResult:
        self.tracer.record(EventType.AGENT_START, agent=self.name, task=task)
        messages = self._seed_messages(task, context)
        schemas = self.tools.schemas() if self.tools else None
        usage = Usage()
        steps = 0

        while steps < self.max_steps:
            steps += 1
            completion = self.provider.complete(
                messages, tools=schemas, temperature=self.temperature
            )
            usage = usage + completion.usage
            self.tracer.record(
                EventType.LLM_CALL,
                agent=self.name,
                step=steps,
                model=completion.model,
                has_tool_calls=completion.has_tool_calls,
                text=completion.text[:500],
            )

            if completion.has_tool_calls:
                messages.append(Message.assistant(completion.text, completion.tool_calls))
                for call in completion.tool_calls:
                    self.tracer.record(
                        EventType.TOOL_CALL, agent=self.name, tool=call.name, arguments=call.arguments
                    )
                    result = (
                        self.tools.execute(call)
                        if self.tools
                        else None
                    )
                    if result is None:
                        from ..types import ToolResult

                        result = ToolResult(
                            tool_call_id=call.id,
                            name=call.name,
                            content="No tools are available to this agent.",
                            is_error=True,
                        )
                    self.tracer.record(
                        EventType.TOOL_RESULT,
                        agent=self.name,
                        tool=call.name,
                        is_error=result.is_error,
                        content=result.content[:500],
                    )
                    messages.append(Message.tool(result))
                continue

            # No tool calls => final answer.
            messages.append(Message.assistant(completion.text))
            if self.memory is not None:
                self.memory.add(f"Task: {task}\nAnswer: {completion.text}")
            self.tracer.record(EventType.AGENT_END, agent=self.name, steps=steps)
            return AgentResult(
                output=completion.text,
                success=True,
                agent=self.name,
                steps=steps,
                messages=messages,
                usage=usage,
            )

        self.tracer.record(EventType.ERROR, agent=self.name, error="max_steps_exceeded")
        raise MaxStepsExceeded(
            f"Agent '{self.name}' did not finish within {self.max_steps} steps."
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Agent {self.name!r} tools={self.tools.names() if self.tools else []}>"
