"""An agent that reviews another agent's output and scores it."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..types import Message
from .base import Agent

_CRITIC_INSTRUCTIONS = (
    "You are a rigorous critic. Review the response against the task. Note "
    "strengths and weaknesses, then end with a line 'Score: N/10'."
)

_SCORE = re.compile(r"score\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE)


@dataclass
class Critique:
    text: str
    score: float  # 0..10
    passed: bool


class CriticAgent(Agent):
    def __init__(self, name: str = "critic", threshold: float = 7.0, **kwargs):
        kwargs.setdefault("instructions", _CRITIC_INSTRUCTIONS)
        kwargs.setdefault("tools", None)
        super().__init__(name, **kwargs)
        self.threshold = threshold

    def review(self, task: str, response: str) -> Critique:
        prompt = (
            f"Task:\n{task}\n\nResponse to critique:\n{response}\n\n"
            "Critique the response and give a score out of 10."
        )
        completion = self.provider.complete(
            [Message.system(self.instructions), Message.user(prompt)],
            temperature=self.temperature,
        )
        text = completion.text
        m = _SCORE.search(text)
        score = float(m.group(1)) if m else 5.0
        return Critique(text=text, score=score, passed=score >= self.threshold)
