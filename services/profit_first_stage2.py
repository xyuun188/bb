"""Profit-First v3 Stage 2 behavior guards.

These guards are small and dependency-free so entry and release paths can share
the same rules without reaching into TradingService internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProfitFirstStage2Decision:
    allowed: bool
    reason: str = ""
    data: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RecentProbePnLBrakePolicy:
    """Block new tiny probes after a recent all-loss probe loop."""

    min_recent_probe_closes: int = 2
    min_upgrade_lane: str = "validated_probe"

    def evaluate(self, plan: dict[str, Any], raw: dict[str, Any]) -> ProfitFirstStage2Decision:
        lane = str(plan.get("decision_lane") or "").lower()
        if lane not in {"tiny_probe", "validated_probe", "meaningful_entry", "high_conviction"}:
            return ProfitFirstStage2Decision(True, data={"lane": lane})
        if lane in {"validated_probe", "meaningful_entry", "high_conviction"}:
            return ProfitFirstStage2Decision(
                True,
                data={"lane": lane, "probe_loss_brake_bypassed_by_upgrade": True},
            )
        health = self._probe_health(raw)
        if not health.get("all_recent_probes_losing"):
            return ProfitFirstStage2Decision(True, data={"lane": lane, "probe_loop_health": health})
        if int(health.get("probe_closed_count") or 0) < self.min_recent_probe_closes:
            return ProfitFirstStage2Decision(True, data={"lane": lane, "probe_loop_health": health})
        return ProfitFirstStage2Decision(
            False,
            reason=(
                "Profit-First v3 probe-loss brake: recent tiny/probe closes are all losing; "
                "this candidate stays shadow unless it upgrades to validated_probe or better."
            ),
            data={
                "lane": lane,
                "probe_loop_health": health,
                "skip_kind": "profit_first_probe_loss_brake",
                "shadow_only": True,
            },
        )

    @staticmethod
    def _probe_health(raw: dict[str, Any]) -> dict[str, Any]:
        candidates = (
            raw.get("probe_loop_health"),
            raw.get("recent_probe_pnl_health"),
            raw.get("profit_first_probe_loop_health"),
        )
        for value in candidates:
            if isinstance(value, dict):
                return value
        return {}


@dataclass(frozen=True, slots=True)
class DefensiveProbeShadowPolicy:
    """Keep low-quality, risk-budget-capped probes out of real execution."""

    min_real_expected_profit_usdt: float = 0.25

    def evaluate(self, raw: dict[str, Any], decision: Any) -> ProfitFirstStage2Decision:
        if not isinstance(raw, dict):
            return ProfitFirstStage2Decision(True)
        plan = raw.get("profit_first_trade_plan")
        plan = plan if isinstance(plan, dict) else {}
        lane = str(plan.get("decision_lane") or "").lower().strip()
        if lane not in {"tiny_probe", "validated_probe"}:
            return ProfitFirstStage2Decision(True, data={"lane": lane})

        sizing = raw.get("profit_risk_sizing")
        sizing = sizing if isinstance(sizing, dict) else {}
        low_payoff = bool(sizing.get("low_payoff_quality"))
        quality_tier = str(sizing.get("quality_tier") or "").lower().strip()
        high_quality = bool(sizing.get("high_quality_entry"))
        expected_profit = _safe_float(sizing.get("expected_profit_usdt"), 0.0)
        dynamic = raw.get("dynamic_leverage_decision")
        if not isinstance(dynamic, dict):
            dynamic = sizing.get("dynamic_leverage_decision")
        dynamic = dynamic if isinstance(dynamic, dict) else {}
        final_leverage = _safe_float(
            dynamic.get("final_integer_leverage"),
            _safe_float(getattr(decision, "suggested_leverage", None), 0.0),
        )
        limiting_factor = str(dynamic.get("limiting_factor") or "").lower().strip()
        dynamic_reasons = {str(item).lower() for item in dynamic.get("reasons") or []}
        risk_budget_capped = limiting_factor == "risk_budget" or "limited_by_risk_budget" in dynamic_reasons

        should_shadow = bool(
            low_payoff
            and not high_quality
            and quality_tier not in {"strong_probe", "quality_override", "high_profit", "elite"}
            and final_leverage <= 1.0
            and risk_budget_capped
            and expected_profit < self.min_real_expected_profit_usdt
        )
        data = {
            "lane": lane,
            "low_payoff_quality": low_payoff,
            "quality_tier": quality_tier,
            "high_quality_entry": high_quality,
            "expected_profit_usdt": expected_profit,
            "final_integer_leverage": final_leverage,
            "dynamic_leverage_limiting_factor": limiting_factor,
            "risk_budget_capped": risk_budget_capped,
        }
        if not should_shadow:
            return ProfitFirstStage2Decision(True, data=data)
        return ProfitFirstStage2Decision(
            False,
            reason=(
                "Profit-First defensive probe guard: low-payoff tiny/probe entry was capped to "
                "1x by risk budget and has too little expected profit; keep it shadow-only until "
                "quality upgrades."
            ),
            data={
                **data,
                "skip_kind": "profit_first_defensive_probe_shadow",
                "shadow_only": True,
                "min_real_expected_profit_usdt": self.min_real_expected_profit_usdt,
            },
        )


@dataclass(frozen=True, slots=True)
class ReleaseNetBenefitPolicy:
    """Protect losing stale probes from release-only churn."""

    min_replacement_expected_net_pct: float = 0.35
    min_replacement_profit_quality: float = 0.45
    min_replacement_lane: str = "validated_probe"

    def evaluate(self, scan: dict[str, Any]) -> ProfitFirstStage2Decision:
        if not isinstance(scan, dict):
            return ProfitFirstStage2Decision(True)
        quality = scan.get("position_quality") if isinstance(scan.get("position_quality"), dict) else {}
        pnl_ratio = _safe_float(quality.get("pnl_ratio"), _safe_float(scan.get("pnl_ratio"), 0.0))
        if pnl_ratio >= 0:
            return ProfitFirstStage2Decision(True, data={"pnl_ratio": pnl_ratio})
        if self._hard_risk(scan, quality):
            return ProfitFirstStage2Decision(
                True,
                data={"pnl_ratio": pnl_ratio, "release_net_benefit_hard_risk": True},
            )
        replacement = self._replacement_opportunity(scan)
        if self._replacement_is_stronger(replacement):
            return ProfitFirstStage2Decision(
                True,
                data={
                    "pnl_ratio": pnl_ratio,
                    "replacement_opportunity": replacement,
                    "release_net_benefit_replacement": True,
                },
            )
        return ProfitFirstStage2Decision(
            False,
            reason=(
                "Profit-First v3 release net-benefit guard: this is a losing release and there "
                "is no hard risk or stronger replacement opportunity; keep it open for the next review."
            ),
            data={
                "pnl_ratio": pnl_ratio,
                "replacement_opportunity": replacement,
                "skip_kind": "profit_first_release_net_benefit_guard",
                "protected_release": True,
            },
        )

    @staticmethod
    def _hard_risk(scan: dict[str, Any], quality: dict[str, Any]) -> bool:
        if bool(scan.get("hard_risk") or scan.get("force_hard_risk_exit")):
            return True
        reasons = {str(item) for item in quality.get("reasons", []) if item is not None}
        if reasons & {"severe_loss_pressure", "hard_loss_pressure", "signal_reversal"}:
            return True
        if _safe_float(quality.get("pnl_ratio"), 0.0) <= -0.025:
            return True
        close_evidence = scan.get("close_evidence") if isinstance(scan.get("close_evidence"), dict) else {}
        return bool(
            close_evidence.get("hard_risk")
            or close_evidence.get("trend_failure")
            or close_evidence.get("predictive_reversal_exit")
        )

    @staticmethod
    def _replacement_opportunity(scan: dict[str, Any]) -> dict[str, Any]:
        replacement = scan.get("replacement_opportunity")
        if isinstance(replacement, dict):
            return replacement
        replacement = scan.get("release_replacement_opportunity")
        return replacement if isinstance(replacement, dict) else {}

    def _replacement_is_stronger(self, replacement: dict[str, Any]) -> bool:
        if not replacement:
            return False
        lane = str(replacement.get("decision_lane") or replacement.get("lane") or "").lower()
        lanes = {"tiny_probe": 1, "validated_probe": 2, "meaningful_entry": 3, "high_conviction": 4}
        required = lanes.get(self.min_replacement_lane, 2)
        expected_net = _safe_float(replacement.get("expected_net_return_pct"), 0.0)
        quality = _safe_float(replacement.get("profit_quality_ratio"), 0.0)
        return bool(
            lanes.get(lane, 0) >= required
            and expected_net >= self.min_replacement_expected_net_pct
            and quality >= self.min_replacement_profit_quality
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
