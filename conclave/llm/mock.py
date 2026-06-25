"""Offline, deterministic providers.

These exist so the entire framework — agents, tools, orchestration — runs and is
fully testable with no network access and no API keys.

* :class:`ScriptedProvider` replays a fixed list of responses. Perfect for tests
  where you want to assert an exact agent trajectory.
* :class:`HeuristicProvider` is a tiny rule-based "brain". It inspects the
  conversation and behaves plausibly: it calls a calculator when it sees
  arithmetic, emits a numbered plan when asked to plan, scores work when asked to
  critique, and otherwise answers from the last user turn. It always terminates.
"""

from __future__ import annotations

import re
from typing import Any

from ..types import Completion, Message, Role, ToolCall, Usage
from .base import LLMProvider

_ARITH = re.compile(r"(-?\d+(?:\.\d+)?(?:\s*[-+*/^]\s*-?\d+(?:\.\d+)?)+)")


def _last_user_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == Role.USER:
            return m.content
    return ""


def _estimate_usage(messages: list[Message], output: str) -> Usage:
    prompt = sum(len(m.content) for m in messages) // 4
    return Usage(prompt_tokens=prompt, completion_tokens=max(1, len(output) // 4))


class ScriptedProvider(LLMProvider):
    """Replays a predetermined list of responses, in order.

    Each script item may be a plain string (becomes completion text), a
    :class:`Completion`, or a dict like ``{"tool": "calculator", "args": {...}}``
    to emit a tool call. When the script is exhausted the last item repeats.
    """

    name = "scripted"

    def __init__(self, script: list[Any]):
        if not script:
            raise ValueError("ScriptedProvider needs at least one scripted response")
        self.script = script
        self._i = 0

    def complete(self, messages, tools=None, *, temperature=None, model=None) -> Completion:
        item = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        if isinstance(item, Completion):
            return item
        if isinstance(item, str):
            return Completion(text=item, usage=_estimate_usage(messages, item), model="scripted")
        if isinstance(item, dict) and "tool" in item:
            tc = ToolCall(name=item["tool"], arguments=item.get("args", {}))
            return Completion(tool_calls=[tc], model="scripted", finish_reason="tool_calls")
        raise TypeError(f"Unsupported script item: {item!r}")


class HeuristicProvider(LLMProvider):
    """A deterministic, rule-based stand-in for a real model."""

    name = "mock"
    default_model = "conclave-mock-1"

    def complete(self, messages, tools=None, *, temperature=None, model=None) -> Completion:
        tools = tools or []
        tool_names = {t["name"] for t in tools}
        last = messages[-1] if messages else None
        convo_text = " ".join(m.content for m in messages).lower()
        user_text = _last_user_text(messages)

        # If we just received a tool result, synthesize a final answer from it.
        if last is not None and last.role == Role.TOOL:
            answer = f"Based on the tool result ({last.content}), here is the answer: {last.content}."
            return Completion(text=answer, usage=_estimate_usage(messages, answer), model=self.default_model)

        already_called = any(m.role == Role.ASSISTANT and m.tool_calls for m in messages)

        # Arithmetic → use the calculator tool if available and not yet used.
        if not already_called and "calculator" in tool_names:
            match = _ARITH.search(user_text)
            if match:
                tc = ToolCall(name="calculator", arguments={"expression": match.group(1).strip()})
                return Completion(tool_calls=[tc], model=self.default_model, finish_reason="tool_calls")

        # Planning request → numbered steps.
        if any(k in convo_text for k in ("create a plan", "numbered plan", "break the task", "decompose")):
            plan = (
                "1. Clarify the objective and constraints.\n"
                "2. Gather the information or inputs required.\n"
                "3. Perform the core work.\n"
                "4. Review the result for correctness.\n"
                "5. Produce the final answer."
            )
            return Completion(text=plan, usage=_estimate_usage(messages, plan), model=self.default_model)

        # Critique request → a structured review with a score.
        if any(k in convo_text for k in ("critique", "review the", "evaluate the")):
            review = (
                "Strengths: the response addresses the task directly.\n"
                "Weaknesses: it could include more supporting detail.\n"
                "Score: 8/10"
            )
            return Completion(text=review, usage=_estimate_usage(messages, review), model=self.default_model)

        # Default: a concise synthesized answer.
        snippet = user_text.strip().split("\n")[0][:240] or "the request"
        answer = f"Here is a concise response to: {snippet}"
        return Completion(text=answer, usage=_estimate_usage(messages, answer), model=self.default_model)


# Friendly default alias.
MockProvider = HeuristicProvider
