# Conclave architecture

Conclave is built in layers, each depending only on the ones below it. Every
layer is swappable, and the whole stack runs offline on a deterministic mock
provider so it is trivially testable.

```
            ┌─────────────────────────────────────────────┐
   CLI /    │  conclave run | tools | serve   (cli.py)     │
 dashboard  │  HTTP trace dashboard           (server/)    │
            └───────────────────┬─────────────────────────┘
            ┌───────────────────▼─────────────────────────┐
Orchestration│ Orchestrator · Team · Router · Blackboard   │
            └───────────────────┬─────────────────────────┘
            ┌───────────────────▼─────────────────────────┐
   Agents   │ Agent · ReActAgent · PlannerAgent · Critic   │
            └───────────────────┬─────────────────────────┘
   ┌────────────────┬───────────┼───────────┬──────────────┐
   ▼                ▼           ▼           ▼              ▼
 Tools           Memory       LLM        Tracing        Types
(registry,    (buffer,     (mock,      (events,      (Message,
 builtins)     vector,      anthropic,   tracer)       ToolCall,
               summary)     openai)                    Completion)
```

## Layer responsibilities

### Types (`conclave/types.py`)
Small, serializable dataclasses — `Message`, `ToolCall`, `ToolResult`,
`Completion`, `AgentResult`, `Usage`. No behaviour, no dependencies. Everything
else speaks in these.

### LLM providers (`conclave/llm/`)
A single `LLMProvider.complete(messages, tools)` interface. `HeuristicProvider`
(the offline default) and `ScriptedProvider` make the framework run and test
without credentials; `AnthropicProvider` and `OpenAIProvider` are stdlib-only
adapters for real models. `get_provider()` auto-selects based on environment.

### Tools (`conclave/tools/`)
The `@tool` decorator turns any typed function into a `Tool` with an
auto-generated JSON schema. `ToolRegistry` executes tool calls and converts any
failure into an error result the agent can read and recover from.

### Memory (`conclave/memory/`)
Three interchangeable backends behind one `Memory` interface: a recency
`ConversationBuffer`, a dependency-free hashing `VectorMemory` for relevance
recall, and a `SummaryMemory` that compresses history via the LLM.

### Agents (`conclave/agents/`)
`Agent` runs the structured tool-use loop (think → call tools → observe →
repeat) bounded by `max_steps`. `ReActAgent` does the same via a text protocol
for models without native tool calling. `PlannerAgent` decomposes goals;
`CriticAgent` scores answers.

### Orchestration (`conclave/orchestration/`)
`Team` runs agents sequentially (a pipeline) or in parallel. `Orchestrator` is a
manager: it plans, routes each step to the best worker via `Router`, shares state
through the `Blackboard`, synthesizes a final answer, and optionally runs a
critic-driven refinement loop.

### Tracing & dashboard (`conclave/tracing/`, `conclave/server/`)
Every meaningful action emits a structured `Event`. The `Tracer` collects them;
the stdlib HTTP server renders them into a self-contained timeline dashboard.

## Design principles

- **Offline-first.** No network or keys required for development or tests.
- **Zero runtime dependencies.** Standard library only.
- **One interface per layer.** Swap any provider, tool, memory, or agent.
- **Everything is observable.** All control flow is captured in the trace.
