"""Quickstart: a single tool-using agent (runs offline)."""

from conclave import Agent, default_tools


def main() -> None:
    agent = Agent(
        "assistant",
        instructions="You are a helpful assistant. Use tools when they help.",
        tools=default_tools(),
    )
    result = agent.run("What is 137 * 24?")
    print("Answer:", result.output)
    print("Steps:", result.steps)
    print("Trace:", agent.tracer.summary())


if __name__ == "__main__":
    main()
