"""Anthropic Messages API adapter (standard library only).

Reads ``ANTHROPIC_API_KEY`` from the environment. Converts Conclave messages and
tool schemas into the Anthropic wire format and parses the response back into a
:class:`~conclave.types.Completion`, including ``tool_use`` blocks.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from ..errors import ProviderError
from ..types import Completion, Message, Role, ToolCall, Usage
from .base import LLMProvider

_API_URL = "https://api.anthropic.com/v1/messages"


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = "claude-3-5-sonnet-latest"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 60):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set.")
        self.model = model or os.environ.get("CONCLAVE_MODEL") or self.default_model
        self.timeout = timeout

    def _convert(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                system_parts.append(m.content)
            elif m.role == Role.USER:
                out.append({"role": "user", "content": m.content})
            elif m.role == Role.ASSISTANT:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    )
                out.append({"role": "assistant", "content": blocks or m.content})
            elif m.role == Role.TOOL:
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                            }
                        ],
                    }
                )
        return "\n\n".join(p for p in system_parts if p), out

    def complete(self, messages, tools=None, *, temperature=None, model=None) -> Completion:
        system, converted = self._convert(messages)
        payload: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": 1024,
            "messages": converted,
            "temperature": 0.7 if temperature is None else temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t.get("description", ""), "input_schema": t["parameters"]}
                for t in tools
            ]

        body = self._post(payload)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in body.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(id=block["id"], name=block["name"], arguments=block.get("input", {}))
                )
        usage_raw = body.get("usage", {})
        return Completion(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=Usage(usage_raw.get("input_tokens", 0), usage_raw.get("output_tokens", 0)),
            model=body.get("model", payload["model"]),
            finish_reason=body.get("stop_reason", "stop"),
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            _API_URL,
            data=json.dumps(payload).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            raise ProviderError(f"Anthropic API error {exc.code}: {exc.read().decode(errors='replace')}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise ProviderError(f"Could not reach Anthropic API: {exc}") from exc
