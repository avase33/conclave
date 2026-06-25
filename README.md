<div align="center">

# 🏛️ Conclave

### A multi-agent orchestration framework where AI agents plan, delegate, use tools, critique, and collaborate.

[![CI](https://github.com/akhil/conclave/actions/workflows/ci.yml/badge.svg)](https://github.com/akhil/conclave/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Runtime deps](https://img.shields.io/badge/runtime%20deps-0-brightgreen)](pyproject.toml)
[![Offline](https://img.shields.io/badge/works-offline-success)](#offline-first)

</div>

---

**Conclave** is a small, readable, batteries-included framework for building
systems of cooperating AI agents. A manager agent breaks a task into steps,
routes each one to the best-suited worker, lets them share state on a blackboard,
synthesizes their findings, and runs a critic-driven refinement loop — all fully
traced and viewable in a built-in dashboard.

It has **zero runtime dependencies** (standard library only) and is
**offline-first**: with no API key it runs on a deterministic mock provider, so
every example and test works the moment you clone it. Add the Anthropic or OpenAI
adapter to go live.

```python
from conclave import Agent, Orchestrator, CriticAgent, default_tools

workers = [
    Agent("researcher", "Gather the key facts.", tools=default_tools(),
          capabilities=["research", "math", "data"]),
    Agent("writer", "Write a clear answer from the findings.",
          capabilities=["write", "summarize", "explain"]),
]

orchestrator = Orchestrator(workers, critic=CriticAgent())
print(orchestrator.run("Compare microservices and monoliths.").output)
```

## ✨ Why Conclave

| | |
|---|---|
| 🧠 **Real orchestration** | Plan → route → execute → synthesize → critique → refine, not just a chat loop. |
| 🧩 **Pluggable everything** | One interface each for providers, tools, memory, and agents. Swap any of them in a line. |
| 🛠️ **Tools from type hints** | `@tool` turns a typed function into a JSON-schema tool automatically. |
| 🧵 **Teams** | Run agents as a pipeline (`sequential`) or a swarm (`parallel`). |
| 🧮 **Memory** | Recency buffer, dependency-free vector recall, and LLM summarization. |
| 🔍 **Full tracing** | Every step is an event; view the whole run in a self-contained HTML dashboard. |
| 📦 **Zero deps, offline** | Standard library only; deterministic mock provider for dev & CI. |

## 🚀 Install & run

```bash
git clone https://github.com/akhil/conclave.git
cd conclave
pip install -e .

# Run a task with a full orchestrated team (offline mock provider):
conclave run "Plan a launch for a new mobile app" --mode orchestrate --critic

# A single tool-using agent:
conclave run "What is 137 * 24?" --mode single

# A parallel brainstorm, then open the trace dashboard in your browser:
conclave run "Ideas to cut onboarding time" --mode team --team-mode parallel --dashboard

# List built-in tools:
conclave tools
```

### Go live with a real model

```bash
export ANTHROPIC_API_KEY=sk-...        # or OPENAI_API_KEY=sk-...
conclave run "Summarize the CAP theorem" --provider anthropic --mode orchestrate
```

## 🧱 Core concepts

### Agents
An `Agent` is a provider + instructions + tools + memory running a bounded
tool-use loop. `ReActAgent` does the same via a text protocol; `PlannerAgent`
decomposes goals; `CriticAgent` scores answers.

### Tools
```python
from conclave import tool, Agent, ToolRegistry

@tool
def fib(n: int) -> int:
    "Return the nth Fibonacci number."
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

agent = Agent("mathy", tools=ToolRegistry([fib]))
```
The JSON schema for `fib` is generated from its signature and docstring.

### Orchestration
`Team` runs agents sequentially or in parallel. `Orchestrator` is a manager that
plans, routes steps via a capability `Router`, coordinates through a `Blackboard`,
synthesizes, and refines with a critic.

### Tracing & dashboard
```python
result = orchestrator.run(task)
orchestrator.tracer.save("trace.json")
# then:  conclave serve trace.json
```

## <a name="offline-first"></a>🔌 Offline-first design

Conclave never hard-fails for lack of credentials. `get_provider()` picks a real
model when an API key is present and otherwise returns a deterministic
`HeuristicProvider`, so:

- `pytest` runs with no network and no keys,
- examples produce real (if synthetic) output,
- CI is fast and hermetic.

`ScriptedProvider` lets tests assert an exact agent trajectory.

## 🗺️ Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full layered diagram.
In short: **Types → LLM / Tools / Memory / Tracing → Agents → Orchestration →
CLI / Dashboard**, each layer depending only on the ones beneath it.

## 🧪 Development

```bash
pip install -e ".[dev]"
pytest -q
```

CI runs the suite on Python 3.10–3.12, smoke-tests the CLI, and executes every
example — all on the offline provider.

## 🧭 Roadmap

- [ ] Streaming completions and token-level callbacks
- [ ] Persistent memory backends (SQLite, file)
- [ ] Hierarchical teams (managers of managers)
- [ ] Native function-calling for more providers (Gemini, Mistral, local llama.cpp)
- [ ] Live (websocket) dashboard during a run

## 📄 License

MIT © Akhil — see [LICENSE](LICENSE).
