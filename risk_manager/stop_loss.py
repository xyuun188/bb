"""Result types for governed stop-loss execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StopLossType(StrEnum):
    NONE = "none"
    HARD = "hard"
    TRAILING = "trailing"


@dataclass
class StopLossResult:
    triggered: bool
    stop_type: StopLossType = StopLossType.NONE
    exit_price: float = 0.0
    reason: str = ""
    loss_pct: float = 0.0
