"""Tool definitions and registry.

A :class:`Tool` wraps a plain Python callable together with a JSON schema for its
arguments. The :func:`tool` decorator builds that schema automatically from the
function's type hints and docstring, so defining a new capability is just::

    @tool
    def add(a: int, b: int) -> int:
        "Add two integers."
        return a + b
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from ..errors import ToolError, ToolNotFoundError
from ..types import ToolCall, ToolResult

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> str:
    return _JSON_TYPES.get(annotation, "string")


def _build_schema(func: Callable) -> dict[str, Any]:
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[pname] = {"type": _json_type(param.annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable
    tags: list[str] = field(default_factory=list)

    def run(self, **kwargs: Any) -> str:
        # Keep only arguments this tool actually declares, so loosely-parsed
        # tool calls (e.g. from the ReAct text protocol) don't raise on extras.
        allowed = set(self.parameters.get("properties", {}))
        if allowed:
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        try:
            result = self.func(**kwargs)
        except TypeError as exc:
            raise ToolError(f"Bad arguments for tool '{self.name}': {exc}") from exc
        except Exception as exc:  # surface as ToolError so agents can recover
            raise ToolError(f"Tool '{self.name}' failed: {exc}") from exc
        return result if isinstance(result, str) else str(result)

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def tool(func: Callable | None = None, *, name: str | None = None, description: str | None = None,
         tags: list[str] | None = None) -> Tool | Callable[[Callable], Tool]:
    """Decorator that turns a function into a :class:`Tool`."""

    def wrap(fn: Callable) -> Tool:
        doc = (description or inspect.getdoc(fn) or "").strip()
        return Tool(
            name=name or fn.__name__,
            description=doc,
            parameters=_build_schema(fn),
            func=fn,
            tags=tags or [],
        )

    return wrap(func) if func is not None else wrap


class ToolRegistry:
    """A named collection of tools, with safe execution of tool calls."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> Tool:
        self._tools[t.name] = t
        return t

    def add(self, *tools: Tool) -> "ToolRegistry":
        for t in tools:
            self.register(t)
        return self

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolNotFoundError(f"No tool named '{name}'. Available: {', '.join(self.names()) or 'none'}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a tool call, capturing errors as an error ToolResult."""
        try:
            tool_obj = self.get(call.name)
            content = tool_obj.run(**call.arguments)
            return ToolResult(tool_call_id=call.id, name=call.name, content=content)
        except (ToolError, ToolNotFoundError) as exc:
            return ToolResult(tool_call_id=call.id, name=call.name, content=str(exc), is_error=True)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
