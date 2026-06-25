"""The orchestrator: a manager that coordinates a pool of worker agents.

Workflow:

1. A :class:`PlannerAgent` decomposes the task into ordered steps.
2. A :class:`Router` assigns each step to the most suitable worker.
3. Each worker runs its step with the shared blackboard as context.
4. A synthesizer agent combines all findings into a final answer.
5. (Optional) a :class:`CriticAgent` reviews the answer and triggers a bounded
   refinement loop until it passes or the refinement budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..agents.base import Agent
from ..agents.critic import CriticAgent, Critique
from ..agents.planner import PlannerAgent
from ..llm.base import LLMProvider
from ..llm.registry import get_provider
from ..tracing.events import EventType
from ..tracing.tracer import Tracer
from ..types import AgentResult, Usage
from .blackboard import Blackboard
from .router import Router

_SYNTH_INSTRUCTIONS = (
    "You are a synthesizer. Combine the findings from several agents into a "
    "single, coherent, complete answer to the original task. Resolve overlaps "
    "and present the result clearly."
)


@dataclass
class OrchestratorResult:
    output: str
    plan: list[str] = field(default_factory=list)
    step_results: list[AgentResult] = field(default_factory=list)
    critique: Critique | None = None
    refinements: int = 0
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict:
        return {
            "output": self.output,
            "plan": self.plan,
            "steps": [r.to_dict() for r in self.step_results],
            "critique": None if self.critique is None else {
                "score": self.critique.score,
                "passed": self.critique.passed,
            },
            "refinements": self.refinements,
            "total_tokens": self.usage.total_tokens,
        }


class Orchestrator:
    def __init__(
        self,
        workers: list[Agent],
        *,
        provider: LLMProvider | None = None,
        planner: PlannerAgent | None = None,
        synthesizer: Agent | None = None,
        critic: CriticAgent | None = None,
        tracer: Tracer | None = None,
        blackboard: Blackboard | None = None,
        max_refinements: int = 1,
    ):
        if not workers:
            raise ValueError("Orchestrator needs at least one worker agent")
        self.provider = provider or get_provider()
        self.tracer = tracer or Tracer()
        self.blackboard = blackboard or Blackboard()
        self.workers = workers
        for w in workers:
            w.tracer = self.tracer
        self.router = Router(workers)
        self.planner = planner or PlannerAgent(provider=self.provider, tracer=self.tracer)
        self.synthesizer = synthesizer or Agent(
            "synthesizer", _SYNTH_INSTRUCTIONS, provider=self.provider, tracer=self.tracer
        )
        self.critic = critic
        self.max_refinements = max_refinements

    def run(self, task: str) -> OrchestratorResult:
        self.tracer.record(EventType.RUN_START, task=task, workers=[w.name for w in self.workers])
        usage = Usage()

        plan = self.planner.plan(task) or [task]
        self.tracer.record(EventType.NOTE, note="plan_created", steps=len(plan))

        step_results: list[AgentResult] = []
        for i, step in enumerate(plan, start=1):
            worker = self.router.route(step)
            res = worker.run(step, context=self.blackboard.as_context() or None)
            usage = usage + res.usage
            step_results.append(res)
            self.blackboard.write(f"step_{i}", f"{step} -> {res.output}", author=worker.name)
            self.tracer.record(EventType.STEP, index=i, step=step, worker=worker.name)

        output, synth_usage = self._synthesize(task)
        usage = usage + synth_usage

        result = OrchestratorResult(
            output=output, plan=plan, step_results=step_results, usage=usage
        )

        if self.critic is not None:
            output, result = self._critique_and_refine(task, output, result, usage)

        self.tracer.record(EventType.RUN_END, total_tokens=result.usage.total_tokens)
        return result

    def _synthesize(self, task: str) -> tuple[str, Usage]:
        findings = self.blackboard.as_context()
        prompt = f"Original task:\n{task}\n\nFindings from agents:\n{findings}"
        res = self.synthesizer.run(prompt)
        return res.output, res.usage

    def _critique_and_refine(self, task, output, result, usage):
        assert self.critic is not None
        refinements = 0
        critique = self.critic.review(task, output)
        usage = usage + Usage()  # critic uses its own provider; tokens tracked loosely
        while not critique.passed and refinements < self.max_refinements:
            refinements += 1
            self.tracer.record(EventType.NOTE, note="refining", attempt=refinements, score=critique.score)
            improve = self.synthesizer.run(
                f"Improve this answer based on the critique.\n\nTask:\n{task}\n\n"
                f"Answer:\n{output}\n\nCritique:\n{critique.text}"
            )
            output = improve.output
            usage = usage + improve.usage
            critique = self.critic.review(task, output)

        result.output = output
        result.critique = critique
        result.refinements = refinements
        result.usage = usage
        return output, result
