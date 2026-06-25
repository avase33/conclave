"""A classic ReAct agent driven by a text protocol.

Unlike :class:`Agent` (which uses structured/native tool calling), the ReAct
agent works with any text model by parsing a ``Thought / Action / Action Input /
Observation`` transcript. It stops on a ``Final Answer:`` line, or when the model
stops requesting actions.
"""

from __future__ import annotations

import json
import re

from ..tracing.events import EventType
from ..types import AgentResult, Message, ToolCall, Usage
from .base import Agent

_ACTION = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE)
_ACTION_INPUT = re.compile(r"Action Input\s*:\s*(.+)", re.IGNORECASE)
_FINAL = re.compile(r"Final Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)

_REACT_TEMPLATE = """{instructions}

Use this exact format:
Thought: your reasoning
Action: the tool to use, one of [{tools}]
Action Input: the input to the tool
Observation: (filled in for you)
... (repeat Thought/Action/Action Input/Observation as needed)
Final Answer: the answer to the task
"""


class ReActAgent(Agent):
    def _system_message(self) -> Message:
        tools = ", ".join(self.tools.names()) if self.tools else "none"
        return Message.system(_REACT_TEMPLATE.format(instructions=self.instructions, tools=tools))

    def run(self, task: str, context: str | None = None) -> AgentResult:
        self.tracer.record(EventType.AGENT_START, agent=self.name, task=task, mode="react")
        messages = self._seed_messages(task, context)
        usage = Usage()

        for step in range(1, self.max_steps + 1):
            completion = self.provider.complete(messages, temperature=self.temperature)
            usage = usage + completion.usage
            text = completion.text
            messages.append(Message.assistant(text))
            self.tracer.record(EventType.LLM_CALL, agent=self.name, step=step, text=text[:500])

            final = _FINAL.search(text)
            if final:
                answer = final.group(1).strip()
                self.tracer.record(EventType.AGENT_END, agent=self.name, steps=step)
                return AgentResult(output=answer, agent=self.name, steps=step, messages=messages, usage=usage)

            action = _ACTION.search(text)
            if not action or not self.tools:
                # No further action requested: treat the text as the answer.
                self.tracer.record(EventType.AGENT_END, agent=self.name, steps=step)
                return AgentResult(output=text.strip(), agent=self.name, steps=step, messages=messages, usage=usage)

            tool_name = action.group(1).strip().splitlines()[0].strip()
            raw_input = _ACTION_INPUT.search(text)
            args = self._parse_args(raw_input.group(1).strip() if raw_input else "")
            call = ToolCall(name=tool_name, arguments=args)
            self.tracer.record(EventType.TOOL_CALL, agent=self.name, tool=tool_name, arguments=args)
            result = self.tools.execute(call)
            self.tracer.record(EventType.TOOL_RESULT, agent=self.name, tool=tool_name, content=result.content[:500])
            messages.append(Message.user(f"Observation: {result.content}"))

        self.tracer.record(EventType.ERROR, agent=self.name, error="max_steps_exceeded")
        return AgentResult(
            output="(stopped: step budget exhausted)",
            success=False,
            agent=self.name,
            steps=self.max_steps,
            messages=messages,
            usage=usage,
        )

    @staticmethod
    def _parse_args(raw: str) -> dict:
        raw = raw.strip().splitlines()[0].strip() if raw else ""
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"input": parsed}
        except json.JSONDecodeError:
            return {"expression": raw, "input": raw, "text": raw}
