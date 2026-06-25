from conclave.agents import Agent, CriticAgent, PlannerAgent, ReActAgent
from conclave.llm import HeuristicProvider, ScriptedProvider
from conclave.tools import default_tools


def test_agent_answers_without_tools():
    agent = Agent("a", provider=HeuristicProvider())
    res = agent.run("Tell me about agents.")
    assert res.success and res.output
    assert res.steps == 1


def test_agent_uses_calculator_tool():
    agent = Agent("calc", provider=HeuristicProvider(), tools=default_tools())
    res = agent.run("What is 21 * 2?")
    assert "42" in res.output
    assert res.steps == 2  # one tool call, then final answer
    kinds = agent.tracer.summary()
    assert kinds.get("tool_call", 0) == 1


def test_react_agent_with_scripted_provider():
    script = [
        "Thought: I should add them.\nAction: calculator\nAction Input: 2 + 2",
        "Thought: done.\nFinal Answer: 4",
    ]
    agent = ReActAgent("react", provider=ScriptedProvider(script), tools=default_tools())
    res = agent.run("Add 2 and 2.")
    assert res.output == "4"


def test_planner_returns_steps():
    planner = PlannerAgent(provider=HeuristicProvider())
    steps = planner.plan("Launch a new product")
    assert len(steps) >= 3
    assert all(isinstance(s, str) and s for s in steps)


def test_critic_scores_and_passes():
    critic = CriticAgent(provider=HeuristicProvider(), threshold=7.0)
    c = critic.review("write a haiku", "a perfectly fine haiku")
    assert c.score == 8.0
    assert c.passed is True
