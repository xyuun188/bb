"""Post-crash rebound guard for short entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput

ENTRY_POST_CRASH_REBOUND_1M = 0.030
ENTRY_POST_CRASH_REBOUND_5M_DROP = -0.18
ENTRY_POST_CRASH_REBOUND_20M_DROP = -0.25


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class EntryPostCrashReboundGuardPolicy:
    """Block fresh shorts when a crash trace is followed by a sharp rebound."""

    rebound_1m: float = ENTRY_POST_CRASH_REBOUND_1M
    drop_5m: float = ENTRY_POST_CRASH_REBOUND_5M_DROP
    drop_20m: float = ENTRY_POST_CRASH_REBOUND_20M_DROP

    def guard_reason(self, decision: DecisionOutput) -> str | None:
        if decision.action != Action.SHORT:
            return None
        snapshot = decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        if not snapshot:
            return None
        returns_1 = _safe_float(snapshot.get("returns_1"), 0.0)
        returns_5 = _safe_float(snapshot.get("returns_5"), 0.0)
        returns_20 = _safe_float(snapshot.get("returns_20"), 0.0)
        if not (
            returns_1 >= self.rebound_1m
            and (returns_5 <= self.drop_5m or returns_20 <= self.drop_20m)
        ):
            return None

        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        raw["post_crash_rebound_guard"] = {
            "blocked": True,
            "action": decision.action.value,
            "returns_1": round(returns_1, 6),
            "returns_5": round(returns_5, 6),
            "returns_20": round(returns_20, 6),
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality, 6),
            "policy": "暴跌后 1 分钟强反弹时不追空，等待新一轮行情确认方向。",
        }
        decision.raw_response = raw
        return (
            f"暴跌后反弹保护：该币种刚经历短周期大跌，但最新 1 分钟已反弹 "
            f"{returns_1 * 100:.2f}%；5 分钟/20 分钟仍保留暴跌痕迹"
            f"（5m {returns_5 * 100:.2f}%，20m {returns_20 * 100:.2f}%）。"
            "这类结构容易从插针低点快速反抽，系统不追空，等待下一轮新行情重新判断。"
        )
