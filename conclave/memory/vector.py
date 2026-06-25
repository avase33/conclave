"""A dependency-free vector memory.

Embeddings are produced by hashing tokens into a fixed-dimensional bag-of-words
vector. This is deterministic and needs no model or network, which keeps the
whole framework runnable offline while still giving sensible relevance ranking
for recall. Swap in real embeddings by subclassing and overriding ``_embed``.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from .base import Memory, MemoryRecord

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class VectorMemory(Memory):
    def __init__(self, dim: int = 256):
        self.dim = dim
        self._records: list[MemoryRecord] = []
        self._vectors: list[list[float]] = []

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def add(self, content: str, **metadata: Any) -> None:
        self._records.append(MemoryRecord(content=content, metadata=metadata))
        self._vectors.append(self._embed(content))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not self._records:
            return []
        q = self._embed(query)
        scored = [
            (rec.content, self._cosine(q, vec))
            for rec, vec in zip(self._records, self._vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def recall(self, query: str, k: int = 5) -> list[str]:
        return [content for content, score in self.search(query, k) if score > 0]

    def clear(self) -> None:
        self._records.clear()
        self._vectors.clear()

    def __len__(self) -> int:
        return len(self._records)
