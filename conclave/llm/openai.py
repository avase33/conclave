"""OpenAI Chat Completions adapter (standard library only).

Reads ``OPENAI_API_KEY``. Supports function/tool calling.
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

_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 60):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not set.")
        self.model = model or os.environ.get("CONCLAVE_MODEL") or self.default_model
        self.timeout = timeout

    def _convert(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == Role.TOOL:
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == Role.ASSISTANT and m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            else:
                out.append({"role": m.role.value, "content": m.content})
        return out

    def complete(self, messages, tools=None, *, temperature=None, model=None) -> Completion:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": self._convert(messages),
            "temperature": 0.7 if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]

        body = self._post(payload)
        choice = body["choices"][0]["message"]
        tool_calls: list[ToolCall] = []
        for tc in choice.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn["name"], arguments=args))
        usage_raw = body.get("usage", {})
        return Completion(
            text=choice.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage(usage_raw.get("prompt_tokens", 0), usage_raw.get("completion_tokens", 0)),
            model=body.get("model", payload["model"]),
            finish_reason=body["choices"][0].get("finish_reason", "stop"),
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            _API_URL,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            raise ProviderError(f"OpenAI API error {exc.code}: {exc.read().decode(errors='replace')}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise ProviderError(f"Could not reach OpenAI API: {exc}") from exc
