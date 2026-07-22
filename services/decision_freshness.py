"""Dynamic entry-decision validity derived from return-model horizons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.paper_training import (
    PAPER_TRAINING_VERSION,
    paper_training_contract_reasons,
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


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
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def entry_validity_seconds_from_raw(raw_response: Any) -> float:
    raw = _safe_dict(raw_response)
    paper_training = _safe_dict(raw.get("paper_training"))
    if (
        paper_training.get("version") == PAPER_TRAINING_VERSION
        and not paper_training_contract_reasons(paper_training)
    ):
        valid_for_seconds = _safe_float(
            paper_training.get("valid_for_seconds"),
            0.0,
        )
        if valid_for_seconds > 0:
            return valid_for_seconds
    paper_canary = _safe_dict(raw.get("paper_bootstrap_canary"))
    if (
        paper_canary.get("authorized") is True
        and paper_canary.get("requested") is True
        and paper_canary.get("execution_scope") == "paper_only"
        and paper_canary.get("production_permission") is False
    ):
        observation = _safe_dict(paper_canary.get("selected_observation"))
        horizon_minutes = _safe_float(observation.get("horizon_minutes"), 0.0)
        if horizon_minutes > 0:
            return horizon_minutes * 60.0
    opportunity = _safe_dict(raw.get("opportunity_score"))
    provenance = _safe_dict(opportunity.get("policy_provenance"))
    value = _safe_float(provenance.get("valid_for_seconds"), 0.0)
    return value if value > 0 else 0.0


def entry_reference_time_from_raw(raw_response: Any) -> datetime | None:
    raw = _safe_dict(raw_response)
    paper_training = _safe_dict(raw.get("paper_training"))
    if (
        paper_training.get("version") == PAPER_TRAINING_VERSION
        and not paper_training_contract_reasons(paper_training)
    ):
        provenance = _safe_dict(paper_training.get("policy_provenance"))
        generated_at = parse_utc_datetime(provenance.get("generated_at"))
        if generated_at is not None:
            return generated_at
    paper_canary = _safe_dict(raw.get("paper_bootstrap_canary"))
    if (
        paper_canary.get("authorized") is True
        and paper_canary.get("requested") is True
        and paper_canary.get("execution_scope") == "paper_only"
        and paper_canary.get("production_permission") is False
    ):
        provenance = _safe_dict(paper_canary.get("policy_provenance"))
        generated_at = parse_utc_datetime(
            provenance.get("generated_at") or paper_canary.get("generated_at")
        )
        if generated_at is not None:
            return generated_at
    opportunity = _safe_dict(raw.get("opportunity_score"))
    provenance = _safe_dict(opportunity.get("policy_provenance"))
    return parse_utc_datetime(provenance.get("generated_at"))


@dataclass(slots=True)
class DecisionFreshnessPolicy:
    """Reject entries whose authoritative return horizon is missing or expired."""

    forced_exit_checker: Callable[[DecisionOutput], bool] = lambda _decision: False
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)

    def decision_reference_time(self, decision: DecisionOutput) -> datetime:
        if decision.is_entry:
            generated_at = entry_reference_time_from_raw(decision.raw_response)
            if generated_at is not None:
                return generated_at

        raw = _safe_dict(decision.raw_response)
        timing = _safe_dict(raw.get("timing"))
        for key in ("decision_completed_at", "analysis_started_at"):
            parsed = parse_utc_datetime(timing.get(key))
            if parsed is not None:
                return parsed
        parsed_decision_time = parse_utc_datetime(decision.timestamp)
        if parsed_decision_time is not None:
            return parsed_decision_time
        snapshot = _safe_dict(decision.feature_snapshot)
        for key in ("timestamp", "feature_timestamp", "market_timestamp"):
            parsed = parse_utc_datetime(snapshot.get(key))
            if parsed is not None:
                return parsed
        return self.clock()

    def decision_age_seconds(self, decision: DecisionOutput) -> float:
        return max((self.clock() - self.decision_reference_time(decision)).total_seconds(), 0.0)

    @staticmethod
    def max_age_seconds(decision: DecisionOutput) -> float:
        return entry_validity_seconds_from_raw(decision.raw_response) if decision.is_entry else 0.0

    def stale_decision_reason(self, decision: DecisionOutput) -> str | None:
        if decision.is_hold or decision.is_exit or self.forced_exit_checker(decision):
            return None
        valid_for = self.max_age_seconds(decision)
        age = self.decision_age_seconds(decision)
        raw = _safe_dict(decision.raw_response)
        if valid_for <= 0:
            raw["stale_decision_check"] = {
                "applied": True,
                "age_seconds": round(age, 3),
                "valid_for_seconds": 0.0,
                "reason": "return_horizon_provenance_missing",
                "reference_time": self.decision_reference_time(decision).isoformat(),
            }
            decision.raw_response = raw
            return "收益模型没有提供可审计的动态有效期，本次不执行，等待重新分析。"
        if age <= valid_for:
            return None
        raw["stale_decision_check"] = {
            "applied": True,
            "age_seconds": round(age, 3),
            "valid_for_seconds": round(valid_for, 3),
            "reason": "return_horizon_expired",
            "reference_time": self.decision_reference_time(decision).isoformat(),
        }
        decision.raw_response = raw
        return (
            f"收益分布的动态有效期已过：当前信号年龄 {age:.0f} 秒，"
            f"模型预测周期只覆盖 {valid_for:.0f} 秒。本次不执行，等待重新分析。"
        )
