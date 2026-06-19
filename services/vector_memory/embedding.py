"""Deterministic lightweight text embeddings for vector memory.

This intentionally avoids network calls and large model downloads in the trading
process.  The storage layer can later swap to a stronger embedding provider
without changing callers.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-/]+|[\u4e00-\u9fff]{1,4}")


def tokenize_memory_text(text: str) -> list[str]:
    """Tokenize Chinese/English trading text into stable searchable tokens."""

    normalized = str(text or "").strip().lower()
    if not normalized:
        return []
    return _TOKEN_RE.findall(normalized)


def deterministic_text_vector(text: str, *, dimension: int) -> list[float]:
    """Return a normalized feature-hash vector for text similarity search."""

    dim = max(int(dimension or 0), 16)
    vector = [0.0] * dim
    for token in tokenize_memory_text(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        weight = 1.0 + min(len(token), 16) / 32.0
        vector[bucket] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [round(value / norm, 8) for value in vector]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    """Return cosine similarity in the range [-1, 1]."""

    left_values = list(left)
    right_values = list(right)
    size = min(len(left_values), len(right_values))
    if size <= 0:
        return 0.0
    dot = sum(left_values[i] * right_values[i] for i in range(size))
    left_norm = math.sqrt(sum(value * value for value in left_values[:size]))
    right_norm = math.sqrt(sum(value * value for value in right_values[:size]))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)
