"""An agent that decomposes a task into ordered steps."""

from __future__ import annotations

import re

from ..types import Message
from .base import Agent

_PLAN_INSTRUCTIONS = (
    "You are a meticulous planner. Given a goal, create a numbered plan of "
    "concrete, ordered steps. Each step should be a single actionable instruction."
)

_NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*)$")


class PlannerAgent(Agent):
    def __init__(self, name: str = "planner", **kwargs):
        kwargs.setdefault("instructions", _PLAN_INSTRUCTIONS)
        # Planning is a single reasoning step; no tools needed.
        kwargs.setdefault("tools", None)
        super().__init__(name, **kwargs)

    def plan(self, goal: str) -> list[str]:
        prompt = f"Create a plan to accomplish this goal:\n\n{goal}"
        completion = self.provider.complete(
            [Message.system(self.instructions), Message.user(prompt)],
            temperature=self.temperature,
        )
        steps: list[str] = []
        for line in completion.text.splitlines():
            m = _NUMBERED.match(line)
            if m:
                steps.append(m.group(1).strip())
        if not steps:
            # Fall back to non-empty lines if the model didn't use a list.
            steps = [ln.strip() for ln in completion.text.splitlines() if ln.strip()]
        return steps
