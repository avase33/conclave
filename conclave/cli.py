"""Command-line interface for Conclave."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import Agent, CriticAgent
from .llm.registry import get_provider
from .orchestration import Orchestrator, Team
from .tools import default_tools
from .tracing import Tracer
from .version import __version__


def _eprint(*a) -> None:
    print(*a, file=sys.stderr)


def _build_demo_workers(provider, tracer) -> list[Agent]:
    return [
        Agent(
            "researcher",
            "You research and gather the key facts needed to answer a question.",
            provider=provider, tools=default_tools(), tracer=tracer,
            capabilities=["research", "facts", "calculate", "math", "data", "information"],
        ),
        Agent(
            "writer",
            "You write clear, well-structured prose from given findings.",
            provider=provider, tracer=tracer,
            capabilities=["write", "summarize", "explain", "draft", "report"],
        ),
    ]


def cmd_run(args) -> int:
    provider = get_provider(args.provider)
    tracer = Tracer()

    if args.mode == "single":
        agent = Agent("assistant", provider=provider, tools=default_tools(), tracer=tracer)
        result = agent.run(args.task)
        output = result.output
    elif args.mode == "team":
        team = Team(_build_demo_workers(provider, tracer), mode=args.team_mode, tracer=tracer)
        output = team.run(args.task).output
    elif args.mode == "orchestrate":
        critic = CriticAgent(provider=provider, tracer=tracer) if args.critic else None
        orch = Orchestrator(_build_demo_workers(provider, tracer), provider=provider, tracer=tracer, critic=critic)
        output = orch.run(args.task).output
    else:  # pragma: no cover
        _eprint(f"unknown mode {args.mode}")
        return 2

    print("\n" + "=" * 60)
    print(output)
    print("=" * 60)
    _eprint(f"\n[trace] {tracer.summary()}")

    if args.trace:
        path = tracer.save(args.trace)
        _eprint(f"[trace] saved to {path}")
    if args.dashboard:
        from .server import serve_trace

        serve_trace(tracer.to_dict(), port=args.port)
    return 0


def cmd_tools(args) -> int:
    reg = default_tools()
    print("Built-in tools:")
    for name in reg.names():
        t = reg.get(name)
        print(f"  • {name}: {t.description}")
    return 0


def cmd_serve(args) -> int:
    from .server import serve_trace

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    serve_trace(trace, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="conclave",
        description="Conclave — a multi-agent orchestration framework. Runs offline by default.",
    )
    p.add_argument("--version", action="version", version=f"Conclave {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a task with one or more agents")
    run.add_argument("task", help="the task / question")
    run.add_argument("--mode", choices=["single", "team", "orchestrate"], default="orchestrate")
    run.add_argument("--team-mode", choices=["sequential", "parallel"], default="sequential")
    run.add_argument("--provider", default=None, help="mock | anthropic | openai (default: auto)")
    run.add_argument("--critic", action="store_true", help="add a critic + refinement loop (orchestrate mode)")
    run.add_argument("--trace", metavar="PATH", help="save the execution trace as JSON")
    run.add_argument("--dashboard", action="store_true", help="serve the trace dashboard after running")
    run.add_argument("--port", type=int, default=8765)
    run.set_defaults(func=cmd_run)

    tools = sub.add_parser("tools", help="list the built-in tools")
    tools.set_defaults(func=cmd_tools)

    serve = sub.add_parser("serve", help="serve a saved trace dashboard")
    serve.add_argument("trace", help="path to a trace JSON file")
    serve.add_argument("--port", type=int, default=8765)
    serve.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
