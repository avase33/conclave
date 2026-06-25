"""Define a custom tool and give it to an agent."""

from conclave import Agent, ToolRegistry, tool


@tool
def reverse_text(text: str) -> str:
    """Reverse a string of text."""
    return text[::-1]


@tool
def shout(text: str) -> str:
    """Return the text in upper case with an exclamation mark."""
    return text.upper() + "!"


def main() -> None:
    tools = ToolRegistry([reverse_text, shout])
    agent = Agent("toolsmith", "Use the available tools to transform text.", tools=tools)

    # Tools can also be invoked directly, without a model:
    print("reverse:", tools.get("reverse_text").run(text="conclave"))
    print("shout:", tools.get("shout").run(text="ship it"))

    result = agent.run("Reverse the phrase 'multi agent systems'.")
    print("Agent:", result.output)


if __name__ == "__main__":
    main()
