"""Three agents brainstorm the same prompt in parallel, then results are pooled."""

from conclave import Agent, Team, Tracer, get_provider


def main() -> None:
    provider = get_provider()
    tracer = Tracer()
    personas = [
        ("optimist", "You brainstorm bold, ambitious ideas."),
        ("pragmatist", "You brainstorm practical, low-risk ideas."),
        ("contrarian", "You brainstorm unconventional, against-the-grain ideas."),
    ]
    agents = [Agent(name, inst, provider=provider, tracer=tracer) for name, inst in personas]

    team = Team(agents, mode="parallel", tracer=tracer)
    result = team.run("Suggest ways to reduce onboarding time for new engineers.")
    print(result.output)
    print("\nTrace:", tracer.summary())


if __name__ == "__main__":
    main()
