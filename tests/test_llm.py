from conclave.llm import HeuristicProvider, ScriptedProvider
from conclave.types import Completion, Message, Role


def test_scripted_provider_replays_in_order():
    p = ScriptedProvider(["first", "second"])
    assert p.complete([Message.user("x")]).text == "first"
    assert p.complete([Message.user("x")]).text == "second"
    # Exhausted -> repeats last.
    assert p.complete([Message.user("x")]).text == "second"


def test_scripted_provider_tool_call():
    p = ScriptedProvider([{"tool": "calculator", "args": {"expression": "1+1"}}])
    c = p.complete([Message.user("x")])
    assert c.has_tool_calls
    assert c.tool_calls[0].name == "calculator"


def test_heuristic_calls_calculator_on_arithmetic():
    p = HeuristicProvider()
    tools = [{"name": "calculator", "description": "", "parameters": {}}]
    c = p.complete([Message.user("What is 12 * 9?")], tools=tools)
    assert c.has_tool_calls
    assert c.tool_calls[0].name == "calculator"
    assert "12" in c.tool_calls[0].arguments["expression"]


def test_heuristic_finalizes_after_tool_result():
    p = HeuristicProvider()
    msgs = [Message.user("compute"), Message(role=Role.TOOL, content="108", tool_call_id="t1", name="calculator")]
    c = p.complete(msgs, tools=[{"name": "calculator", "description": "", "parameters": {}}])
    assert not c.has_tool_calls
    assert "108" in c.text


def test_heuristic_plan_and_critique():
    p = HeuristicProvider()
    plan = p.complete([Message.user("create a plan for launch")])
    assert "1." in plan.text
    crit = p.complete([Message.user("critique this answer")])
    assert "Score:" in crit.text
