import pytest

from conclave.tools import ToolRegistry, calculator, default_tools, tool
from conclave.types import ToolCall


def test_tool_decorator_builds_schema():
    @tool
    def greet(name: str, times: int = 1) -> str:
        "Greet someone."
        return "hi " * times + name

    assert greet.name == "greet"
    assert greet.description == "Greet someone."
    props = greet.parameters["properties"]
    assert props["name"]["type"] == "string"
    assert props["times"]["type"] == "integer"
    assert greet.parameters["required"] == ["name"]  # times has a default


def test_calculator_evaluates():
    assert calculator.run(expression="2 + 3 * 4") == "14"
    assert calculator.run(expression="2 ^ 10") == "1024"  # ^ mapped to **


def test_calculator_rejects_unsafe():
    from conclave.errors import ToolError

    with pytest.raises(ToolError):
        calculator.run(expression="__import__('os').system('echo hi')")


def test_tool_run_ignores_extra_kwargs():
    # ReAct may pass several candidate keys; the tool keeps only what it declares.
    assert calculator.run(expression="5+5", input="ignored", text="ignored") == "10"


def test_registry_execute_and_errors():
    reg = default_tools()
    ok = reg.execute(ToolCall(name="calculator", arguments={"expression": "9*9"}))
    assert ok.content == "81" and not ok.is_error

    missing = reg.execute(ToolCall(name="nope", arguments={}))
    assert missing.is_error
    assert "nope" in missing.content


def test_registry_schemas_and_names():
    reg = default_tools()
    assert "calculator" in reg.names()
    assert all("name" in s and "parameters" in s for s in reg.schemas())
