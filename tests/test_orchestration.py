from conclave.agents import Agent, CriticAgent
from conclave.llm import HeuristicProvider
from conclave.orchestration import Blackboard, Orchestrator, Router, Team


def _provider():
    return HeuristicProvider()


def test_blackboard():
    bb = Blackboard()
    bb.write("k", "v", author="alice")
    assert bb.read("k") == "v"
    assert "k" in bb
    assert "k: v" in bb.as_context()
    assert bb.history[0].author == "alice"


def test_router_picks_by_capability():
    a = Agent("writer", capabilities=["write", "summarize"], provider=_provider())
    b = Agent("mathy", capabilities=["calculate", "math", "numbers"], provider=_provider())
    router = Router([a, b])
    assert router.route("please calculate the math here").name == "mathy"
    assert router.route("write and summarize this").name == "writer"


def test_team_sequential():
    agents = [
        Agent("drafter", provider=_provider(), capabilities=["draft"]),
        Agent("editor", provider=_provider(), capabilities=["edit"]),
    ]
    team = Team(agents, mode="sequential")
    res = team.run("Produce a short note.")
    assert len(res.results) == 2
    assert res.output


def test_team_parallel():
    agents = [Agent(f"a{i}", provider=_provider()) for i in range(3)]
    team = Team(agents, mode="parallel")
    res = team.run("Brainstorm ideas.")
    assert len(res.results) == 3
    assert "[a0]" in res.output


def test_orchestrator_end_to_end():
    workers = [
        Agent("researcher", provider=_provider(), capabilities=["research", "facts", "math"]),
        Agent("writer", provider=_provider(), capabilities=["write", "summarize"]),
    ]
    orch = Orchestrator(workers, provider=_provider(), critic=CriticAgent(provider=_provider()))
    res = orch.run("Explain why testing matters.")
    assert res.output
    assert len(res.plan) >= 1
    assert res.critique is not None
    assert orch.tracer.summary().get("run_end", 0) == 1
