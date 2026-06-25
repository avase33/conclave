"""A lightweight, in-memory tracer.

Agents, tools and orchestrators record structured events here. The trace can be
serialized to JSON and rendered by the web dashboard, giving a full picture of
how a multi-agent run unfolded.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .events import Event, EventType


class Tracer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events: list[Event] = []
        self.started_at = time.time()

    def record(self, type: EventType, **data: Any) -> None:
        if self.enabled:
            self.events.append(Event(type=type, data=data))

    @contextmanager
    def span(self, name: str, **data: Any) -> Iterator[None]:
        start = time.time()
        self.record(EventType.AGENT_START, name=name, **data)
        try:
            yield
        except Exception as exc:
            self.record(EventType.ERROR, name=name, error=str(exc))
            raise
        finally:
            self.record(EventType.AGENT_END, name=name, duration=round(time.time() - start, 4))

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.type.value] = counts.get(e.type.value, 0) + 1
        return counts
