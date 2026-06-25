"""Capability-based routing.

The router picks the best agent for a task by scoring keyword overlap between the
task and each agent's declared capabilities (and name). It is intentionally
simple and deterministic; replace ``score`` with an LLM call for smarter routing.
"""

from __future__ import annotations

import re

from ..agents.base import Agent

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class Router:
    def __init__(self, agents: list[Agent]):
        if not agents:
            raise ValueError("Router needs at least one agent")
        self.agents = agents

    def score(self, task: str, agent: Agent) -> int:
        task_tokens = _tokens(task)
        agent_tokens = _tokens(" ".join(agent.capabilities) + " " + agent.name)
        return len(task_tokens & agent_tokens)

    def route(self, task: str) -> Agent:
        ranked = sorted(self.agents, key=lambda a: self.score(task, a), reverse=True)
        return ranked[0]

    def rank(self, task: str) -> list[tuple[Agent, int]]:
        return sorted(
            ((a, self.score(task, a)) for a in self.agents),
            key=lambda x: x[1],
            reverse=True,
        )
