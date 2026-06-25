"""Teams: run several agents together, sequentially or in parallel.

* ``sequential`` — agents form a pipeline; each receives the previous agent's
  output as additional context. Good for draft → refine → finalize chains.
* ``parallel`` — every agent independently tackles the same task; results are
  collected for a downstream synthesizer or vote.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..agents.base import Agent
from ..errors import OrchestrationError
from ..tracing.events import EventType
from ..tracing.tracer import Tracer
from ..types import AgentResult, Usage
from .blackboard import Blackboard


@dataclass
class TeamResult:
    output: str
    mode: str
    results: list[AgentResult] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "output": self.output,
            "agents": [r.to_dict() for r in self.results],
            "total_tokens": self.usage.total_tokens,
        }


class Team:
    def __init__(
        self,
        agents: list[Agent],
        mode: str = "sequential",
        *,
        tracer: Tracer | None = None,
        blackboard: Blackboard | None = None,
        max_workers: int = 4,
    ):
        if not agents:
            raise OrchestrationError("A team needs at least one agent")
        if mode not in ("sequential", "parallel"):
            raise OrchestrationError(f"Unknown team mode: {mode!r}")
        self.agents = agents
        self.mode = mode
        self.tracer = tracer or Tracer()
        self.blackboard = blackboard or Blackboard()
        self.max_workers = max_workers

    def run(self, task: str) -> TeamResult:
        self.tracer.record(EventType.RUN_START, task=task, mode=self.mode, agents=[a.name for a in self.agents])
        result = self._run_sequential(task) if self.mode == "sequential" else self._run_parallel(task)
        self.tracer.record(EventType.RUN_END, mode=self.mode, total_tokens=result.usage.total_tokens)
        return result

    def _run_sequential(self, task: str) -> TeamResult:
        results: list[AgentResult] = []
        usage = Usage()
        context = ""
        last_output = ""
        for i, agent in enumerate(self.agents):
            agent.tracer = self.tracer
            res = agent.run(task, context=context or None)
            results.append(res)
            usage = usage + res.usage
            last_output = res.output
            self.blackboard.write(f"{agent.name}_output", res.output, author=agent.name)
            context = f"Output from {agent.name}:\n{res.output}"
            if i < len(self.agents) - 1:
                self.tracer.record(EventType.HANDOFF, frm=agent.name, to=self.agents[i + 1].name)
        return TeamResult(output=last_output, mode="sequential", results=results, usage=usage)

    def _run_parallel(self, task: str) -> TeamResult:
        for a in self.agents:
            a.tracer = self.tracer
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(lambda a: a.run(task), self.agents))
        usage = Usage()
        for r in results:
            usage = usage + r.usage
            self.blackboard.write(f"{r.agent}_output", r.output, author=r.agent)
        combined = "\n\n".join(f"[{r.agent}]\n{r.output}" for r in results)
        return TeamResult(output=combined, mode="parallel", results=results, usage=usage)
