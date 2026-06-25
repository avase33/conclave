"""Trace event model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STEP = "step"
    HANDOFF = "handoff"
    NOTE = "note"
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "timestamp": self.timestamp, "data": self.data}
