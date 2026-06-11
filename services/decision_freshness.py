"""Decision freshness policy.

The execution layer should not submit old AI decisions.  This module owns the
reference-time calculation and expiry thresholds so entry/exit policies do not
need to call TradingService private methods.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_brain.base_model import DecisionOutput

ENTRY_DECISION_MAX_AGE_SECONDS = 300.0
ENTRY_STRONG_OPPORTUNITY_MAX_AGE_SECONDS = 240.0
ENTRY_EXCEPTIONAL_OPPORTUNITY_MAX_AGE_SECONDS = 300.0
EXIT_DECISION_MAX_AGE_SECONDS = 120.0
PROFIT_PROTECTION_EXIT_MAX_AGE_SECONDS = 300.0


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(slots=True)
class DecisionFreshnessPolicy:
    """Evaluate whether an AI decision is still safe to submit."""

    forced_exit_checker: Callable[[DecisionOutput], bool] = lambda _decision: False
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    entry_max_age_seconds: float = ENTRY_DECISION_MAX_AGE_SECONDS
    entry_strong_max_age_seconds: float = ENTRY_STRONG_OPPORTUNITY_MAX_AGE_SECONDS
    entry_exceptional_max_age_seconds: float = ENTRY_EXCEPTIONAL_OPPORTUNITY_MAX_AGE_SECONDS
    exit_max_age_seconds: float = EXIT_DECISION_MAX_AGE_SECONDS
    profit_protection_exit_max_age_seconds: float = PROFIT_PROTECTION_EXIT_MAX_AGE_SECONDS

    def decision_reference_time(self, decision: DecisionOutput) -> datetime:
        snapshot_times: list[datetime] = []
        snapshot = decision.feature_snapshot or {}
        for key in ("timestamp", "feature_timestamp", "market_timestamp"):
            parsed = parse_utc_datetime(snapshot.get(key))
            if parsed is not None:
                snapshot_times.append(parsed)

        raw = _safe_dict(decision.raw_response)
        timing = _safe_dict(raw.get("timing"))
        timing_times: dict[str, datetime] = {}
        for key in ("analysis_started_at", "decision_completed_at"):
            parsed = parse_utc_datetime(timing.get(key))
            if parsed is not None:
                timing_times[key] = parsed

        parsed_decision_time = parse_utc_datetime(decision.timestamp)
        if decision.is_exit:
            for key in ("decision_completed_at", "analysis_started_at"):
                parsed = timing_times.get(key)
                if parsed is not None:
                    return parsed
            if parsed_decision_time is not None:
                return parsed_decision_time
            if snapshot_times:
                return max(snapshot_times)
            return self.clock()

        # Entry execution age measures how long the AI decision itself has been
        # waiting. Market snapshot freshness is checked by the price/data recheck.
        for key in ("decision_completed_at", "analysis_started_at"):
            parsed = timing_times.get(key)
            if parsed is not None:
                return parsed
        if parsed_decision_time is not None:
            return parsed_decision_time
        if snapshot_times:
            return max(snapshot_times)
        return self.clock()

    def decision_age_seconds(self, decision: DecisionOutput) -> float:
        return max((self.clock() - self.decision_reference_time(decision)).total_seconds(), 0.0)

    def max_age_seconds(self, decision: DecisionOutput) -> float:
        max_age = self.entry_max_age_seconds if decision.is_entry else self.exit_max_age_seconds
        if decision.is_entry:
            raw = _safe_dict(decision.raw_response)
            opportunity = _safe_dict(raw.get("opportunity_score"))
            score = _safe_float(opportunity.get("score"), 0.0)
            ai_expected = _safe_float(opportunity.get("ai_expected_return_pct"), 0.0)
            confidence = max(min(float(decision.confidence or 0.0), 1.0), 0.0)
            reward_risk = _safe_float(opportunity.get("reward_risk_ratio"), 0.0)
            if confidence >= 0.82 and score >= 6.0 and ai_expected >= 4.0 and reward_risk >= 1.5:
                return self.entry_exceptional_max_age_seconds
            if confidence >= 0.75 and score >= 3.0 and ai_expected >= 2.0 and reward_risk >= 1.2:
                return self.entry_strong_max_age_seconds
        if decision.is_exit:
            raw = _safe_dict(decision.raw_response)
            close_evidence = _safe_dict(raw.get("close_evidence"))
            execution_profit = _safe_dict(raw.get("execution_profit_protection"))
            if close_evidence.get("profit_protection") or execution_profit.get("allow"):
                return max(max_age, self.profit_protection_exit_max_age_seconds)
        return max_age

    def stale_decision_reason(self, decision: DecisionOutput) -> str | None:
        if decision.is_hold or self.forced_exit_checker(decision):
            return None
        max_age = self.max_age_seconds(decision)
        age = self.decision_age_seconds(decision)
        if age <= max_age:
            return None
        age_source = "AI平仓裁决完成到准备下单" if decision.is_exit else "AI开仓裁决完成到准备下单"
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["stale_decision_check"] = {
            "applied": True,
            "age_seconds": round(age, 3),
            "max_age_seconds": round(max_age, 3),
            "age_source": age_source,
            "reference_time": self.decision_reference_time(decision).isoformat(),
        }
        decision.raw_response = raw
        return (
            f"AI信号已过有效期：{age_source}已经过去 {age:.0f} 秒，"
            f"超过允许 {max_age:.0f} 秒。为避免使用旧裁决下单，本次不执行，等待下一轮重新分析。"
        )
