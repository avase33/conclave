"""Built-in tools.

The default set is fully offline and side-effect free (calculator, clock, text
utilities). File and HTTP tools are provided too but are opt-in via
:func:`io_tools` so the safe default can't touch the disk or network.
"""

from __future__ import annotations

import ast
import operator
import urllib.request
from datetime import datetime, timezone

from .base import Tool, ToolRegistry, tool

# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (no eval / no builtins).
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+, -, *, /, %, **) and return the result."""
    expr = expression.replace("^", "**")
    try:
        value = _eval_node(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"cannot evaluate '{expression}': {exc}") from exc
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


@tool
def current_time() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@tool
def word_count(text: str) -> str:
    """Count the words and characters in a piece of text."""
    words = len(text.split())
    return f"{words} words, {len(text)} characters"


@tool(tags=["io"])
def read_file(path: str) -> str:
    """Read and return the contents of a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


@tool(tags=["io"])
def write_file(path: str, content: str) -> str:
    """Write text content to a file, returning the number of bytes written."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return f"wrote {len(content.encode('utf-8'))} bytes to {path}"


@tool(tags=["io", "network"])
def http_get(url: str) -> str:
    """Fetch a URL and return up to 4000 characters of the response body."""
    with urllib.request.urlopen(url, timeout=20) as resp:  # pragma: no cover - network
        return resp.read(4000).decode("utf-8", errors="replace")


def default_tools() -> ToolRegistry:
    """Offline, side-effect-free tools suitable as a safe default."""
    return ToolRegistry([calculator, current_time, word_count])


def io_tools() -> list[Tool]:
    """File and network tools — opt in explicitly when you want them."""
    return [read_file, write_file, http_get]
