"""Entry guard for recently wicked symbols."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput

ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT = 80.0
ABNORMAL_WICK_ENTRY_BLOCK_RECENT_HOURS = 96.0
ABNORMAL_WICK_ENTRY_BLOCK_MIN_COUNT = 1


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class EntryAbnormalWickGuardPolicy:
    """Block entries on symbols with recent extreme wick risk."""

    remember_temporary_entry_block: Callable[[str | None, str, float], None] | None = None
    max_wick_pct: float = ABNORMAL_WICK_ENTRY_BLOCK_MAX_PCT
    recent_hours: float = ABNORMAL_WICK_ENTRY_BLOCK_RECENT_HOURS
    min_count: int = ABNORMAL_WICK_ENTRY_BLOCK_MIN_COUNT
    temporary_block_minutes: float = 60.0

    def guard_reason(self, decision: DecisionOutput) -> str | None:
        if not decision.is_entry:
            return None

        snapshot = decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        max_wick_pct = _safe_float(snapshot.get("abnormal_wick_max_pct"), 0.0)
        wick_count = int(_safe_float(snapshot.get("abnormal_wick_count_72h"), 0.0))
        recent_hours = _safe_float(snapshot.get("abnormal_wick_recent_hours"), 9999.0)
        if (
            wick_count < self.min_count
            or max_wick_pct < self.max_wick_pct
            or recent_hours > self.recent_hours
        ):
            return None

        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["abnormal_wick_guard"] = {
            "blocked": True,
            "count_72h": wick_count,
            "max_wick_pct": round(max_wick_pct, 4),
            "recent_hours": round(recent_hours, 4),
            "rule": "recent extreme wick can fill stops far from the planned stop price",
        }
        decision.raw_response = raw
        reason = (
            f"{decision.symbol} 最近 {recent_hours:.1f} 小时内出现过异常插针，"
            f"72 小时内共 {wick_count} 次，最大插针约 {max_wick_pct:.1f}%。"
            "这类币可能让止损按远离计划止损价的极端价成交，本次禁止新开仓，等待异常波动消退。"
        )
        if self.remember_temporary_entry_block is not None:
            self.remember_temporary_entry_block(
                decision.symbol,
                reason,
                max(self.temporary_block_minutes, 60.0),
            )
        return reason
