"""Dynamic pre-execution entry price validation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _feature_snapshot(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is not None and hasattr(value, "to_dict"):
        snapshot = value.to_dict()
        return snapshot if isinstance(snapshot, dict) else {}
    return {}


@dataclass(slots=True)
class EntryPriceGuardPolicy:
    """Require current adverse drift to fit inside authoritative return budget."""

    latest_price_provider: Callable[[str], Awaitable[float]]
    fresh_feature_provider: Callable[[str], Awaitable[Any]]
    market_data_quality_reason_provider: Callable[..., str | None]
    decision_age_seconds_provider: Callable[[DecisionOutput], float]
    async def guard_reason(self, decision: DecisionOutput) -> str | None:
        if not decision.is_entry:
            return None

        snapshot = _safe_dict(decision.feature_snapshot)
        quality_reason = self.market_data_quality_reason_provider(
            snapshot,
            stage_label="pre-order analysis snapshot",
        )
        if quality_reason:
            snapshot = await self._fresh_valid_snapshot(decision.symbol)
            if not snapshot:
                return "Pre-order market data is incomplete; entry fails closed."
            decision.feature_snapshot = snapshot

        snapshot_price = _safe_float(snapshot.get("current_price") or snapshot.get("close"))
        if snapshot_price <= 0:
            return "Pre-order analysis price is missing; entry fails closed."
        latest_price = await self.latest_price_provider(decision.symbol)
        if latest_price <= 0:
            return "Latest pre-order price is unavailable; entry fails closed."

        return_budget = self._return_budget_fraction(decision)
        allowed = return_budget
        if allowed <= 0:
            return "Authoritative fee-after return budget is missing; entry fails closed."

        move = (latest_price - snapshot_price) / snapshot_price
        adverse = self._adverse_move(decision.action, move)
        raw = _safe_dict(decision.raw_response)
        raw["pre_execution_price_check"] = {
            "snapshot_price": snapshot_price,
            "latest_price": latest_price,
            "adverse_move_fraction": round(adverse, 8),
            "return_budget_fraction": round(return_budget, 8),
            "allowed_adverse_move_fraction": round(allowed, 8),
            "decision_age_seconds": round(self.decision_age_seconds_provider(decision), 3),
            "policy_provenance": {
                "source": "authoritative_fee_after_return_lcb",
                "observation_window": "current_pre_order_refresh",
                "sample_count": self._return_sample_count(decision),
                "generated_at": raw.get("generated_at") or "decision_runtime",
                "strategy_version": "2026-07-12.dynamic-price-budget.v1",
                "fallback_reason": "",
            },
        }
        decision.raw_response = raw
        if adverse <= allowed:
            return None

        fresh = await self._fresh_valid_snapshot(decision.symbol)
        fresh_price = _safe_float(fresh.get("current_price") or fresh.get("close"))
        if fresh_price > 0:
            refreshed_gap = abs(latest_price - fresh_price) / fresh_price
            raw["pre_execution_price_recheck"] = {
                "fresh_price": fresh_price,
                "latest_price": latest_price,
                "fresh_gap_fraction": round(refreshed_gap, 8),
                "allowed_adverse_move_fraction": round(allowed, 8),
                "accepted": refreshed_gap <= allowed,
            }
            decision.raw_response = raw
            if refreshed_gap <= allowed:
                decision.feature_snapshot = fresh
                return None

        return (
            "Current adverse price movement exceeds the authoritative fee-after return "
            "budget."
        )

    async def _fresh_valid_snapshot(self, symbol: str) -> dict[str, Any]:
        snapshot = _feature_snapshot(await self.fresh_feature_provider(symbol))
        if not snapshot:
            return {}
        reason = self.market_data_quality_reason_provider(
            snapshot,
            stage_label="pre-order refreshed market snapshot",
        )
        return {} if reason else snapshot

    @staticmethod
    def _adverse_move(action: Action, move: float) -> float:
        if action == Action.LONG:
            return max(move, 0.0)
        if action == Action.SHORT:
            return max(-move, 0.0)
        return 0.0

    @staticmethod
    def _side(decision: DecisionOutput) -> str:
        return "long" if decision.action == Action.LONG else "short"

    def _side_evidence(self, decision: DecisionOutput) -> dict[str, Any]:
        raw = _safe_dict(decision.raw_response)
        evidence = _safe_dict(raw.get("entry_candidate_evidence"))
        side_evidence = _safe_dict(evidence.get(self._side(decision)))
        if side_evidence:
            return side_evidence
        return _safe_dict(_safe_dict(raw.get("authoritative_return_candidate")).get("side_evidence"))

    def _return_budget_fraction(self, decision: DecisionOutput) -> float:
        evidence = self._side_evidence(decision)
        if evidence.get("production_eligible") is not True:
            return 0.0
        expected_net = _safe_float(evidence.get("expected_net_return_pct"))
        return_lcb = _safe_float(evidence.get("return_lcb_pct"))
        if expected_net <= 0 or return_lcb <= 0:
            return 0.0
        return min(expected_net, return_lcb) / 100.0

    def _return_sample_count(self, decision: DecisionOutput) -> int:
        return max(int(_safe_float(self._side_evidence(decision).get("production_source_count"))), 0)
