"""Shared types for the optional vector memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class VectorMemoryDocument:
    """One searchable memory item indexed for similarity retrieval."""

    id: str
    kind: str
    text: str
    symbol: str = ""
    action: str = ""
    outcome: str = ""
    pnl_pct: float | None = None
    created_at: datetime | None = None
    source_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorMemoryHit:
    """A retrieved similar memory item."""

    id: str
    score: float
    kind: str
    text: str
    symbol: str = ""
    action: str = ""
    outcome: str = ""
    pnl_pct: float | None = None
    created_at: str | None = None
    source_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
