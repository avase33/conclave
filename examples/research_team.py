"""A manager orchestrates a research team with a critic + refinement loop.

Runs offline on the mock provider. Set ANTHROPIC_API_KEY or OPENAI_API_KEY (and
pass provider="anthropic"/"openai") to run it against a real model.
"""

from conclave import Agent, CriticAgent, Orchestrator, Tracer, default_tools, get_provider


def main() -> None:
    provider = get_provider()  # auto: real model if a key is set, else mock
    tracer = Tracer()

    workers = [
        Agent(
            "researcher",
            "Gather the key facts and figures needed to answer the question.",
            provider=provider, tools=default_tools(), tracer=tracer,
            capabilities=["research", "facts", "data", "calculate", "math"],
        ),
        Agent(
            "analyst",
            "Analyze findings and draw out the most important implications.",
            provider=provider, tracer=tracer,
            capabilities=["analyze", "compare", "evaluate", "implications"],
        ),
        Agent(
            "writer",
            "Write a clear, concise final answer from the findings.",
            provider=provider, tracer=tracer,
            capabilities=["write", "summarize", "report", "explain"],
        ),
    ]

    orchestrator = Orchestrator(
        workers,
        provider=provider,
        critic=CriticAgent(provider=provider, tracer=tracer),
        tracer=tracer,
        max_refinements=1,
    )

    result = orchestrator.run("Explain the trade-offs of microservices vs a monolith.")
    print("PLAN:")
    for i, step in enumerate(result.plan, 1):
        print(f"  {i}. {step}")
    print("\nFINAL ANSWER:\n", result.output)
    if result.critique:
        print(f"\nCritic score: {result.critique.score}/10 (passed={result.critique.passed})")
    print("\nTrace summary:", tracer.summary())


if __name__ == "__main__":
    main()
