"""Profit-First v3 Stage 2 behavior guards.

These guards are small and dependency-free so entry and release paths can share
the same rules without reaching into TradingService internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROFIT_FIRST_LANE_ORDER = {
    "shadow_only": 0,
    "tiny_probe": 1,
    "validated_probe": 2,
    "meaningful_entry": 3,
    "high_conviction": 4,
}


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
                "Profit-First 探针亏损刹车：最近的极小/探针仓位平仓全部亏损；"
                "该候选先保留为影子样本，只有升级到已验证探针或更高质量档后才允许真实开仓。"
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

    def evaluate(self, raw: dict[str, Any], decision: Any) -> ProfitFirstStage2Decision:
        if not isinstance(raw, dict):
            return ProfitFirstStage2Decision(True)
        plan = raw.get("profit_first_trade_plan")
        plan = plan if isinstance(plan, dict) else {}
        lane = str(plan.get("decision_lane") or "").lower().strip()
        if lane not in {"tiny_probe", "validated_probe"}:
            return ProfitFirstStage2Decision(True, data={"lane": lane})

        opportunity = raw.get("opportunity_score")
        opportunity = opportunity if isinstance(opportunity, dict) else {}
        evidence_score = opportunity.get("evidence_score")
        evidence_score = evidence_score if isinstance(evidence_score, dict) else {}
        sizing = raw.get("profit_risk_sizing")
        sizing = sizing if isinstance(sizing, dict) else {}
        low_payoff = bool(sizing.get("low_payoff_quality"))
        quality_tier = str(sizing.get("quality_tier") or "").lower().strip()
        high_quality = bool(sizing.get("high_quality_entry"))
        expected_profit = _safe_float(sizing.get("expected_profit_usdt"), 0.0)
        expected_net_return = max(
            _safe_float(opportunity.get("expected_net_return_pct"), 0.0),
            _safe_float(plan.get("expected_net_return_pct"), 0.0),
        )
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

        upgrade = profit_first_real_trade_upgrade_context(raw, decision)
        real_trade_upgrade = bool(upgrade.get("ready"))
        should_shadow = bool(
            low_payoff
            and not high_quality
            and quality_tier not in {"strong_probe", "quality_override", "high_profit", "elite"}
            and final_leverage <= 1.0
            and risk_budget_capped
            and not real_trade_upgrade
            and not bool(evidence_score.get("tradeable_probe"))
        )
        data = {
            "lane": lane,
            "low_payoff_quality": low_payoff,
            "quality_tier": quality_tier,
            "high_quality_entry": high_quality,
            "expected_profit_usdt": expected_profit,
            "expected_net_return_pct": expected_net_return,
            "final_integer_leverage": final_leverage,
            "dynamic_leverage_limiting_factor": limiting_factor,
            "risk_budget_capped": risk_budget_capped,
            "profit_first_real_trade_upgrade": real_trade_upgrade,
            "profit_first_real_trade_upgrade_context": upgrade,
            "tradeable_probe": bool(evidence_score.get("tradeable_probe")),
        }
        if not should_shadow:
            return ProfitFirstStage2Decision(True, data=data)
        return ProfitFirstStage2Decision(
            False,
            reason=(
                "Profit-First 防御探针拦截：该极小/探针开仓属于低收益质量，"
                "又被风险预算限制为 1 倍杠杆，预期实际盈利过低；"
                "本轮只记录影子样本，等收益质量升级后再允许真实开仓。"
            ),
            data={
                **data,
                "skip_kind": "profit_first_defensive_probe_shadow",
                "shadow_only": True,
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
                "Profit-First 释放净收益保护：当前是亏损释放信号，且没有硬风险或更强替代机会；"
                "本轮先保留仓位，等待下一轮复盘。"
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


def _lane_rank(value: Any) -> int:
    return PROFIT_FIRST_LANE_ORDER.get(str(value or "").lower().strip(), 0)


def profit_first_real_trade_upgrade_context(raw: dict[str, Any], decision: Any) -> dict[str, Any]:
    plan = raw.get("profit_first_trade_plan")
    plan = plan if isinstance(plan, dict) else {}
    opportunity = raw.get("opportunity_score")
    opportunity = opportunity if isinstance(opportunity, dict) else {}
    evidence_score = opportunity.get("evidence_score")
    evidence_score = evidence_score if isinstance(evidence_score, dict) else {}
    sizing = raw.get("profit_risk_sizing")
    sizing = sizing if isinstance(sizing, dict) else {}
    ladder = sizing.get("profit_first_position_ladder")
    ladder = ladder if isinstance(ladder, dict) else {}
    dynamic = raw.get("dynamic_leverage_decision")
    if not isinstance(dynamic, dict):
        dynamic = sizing.get("dynamic_leverage_decision")
    dynamic = dynamic if isinstance(dynamic, dict) else {}

    plan_lane = str(plan.get("decision_lane") or "").lower().strip()
    ladder_lane = str(ladder.get("lane") or "").lower().strip()
    quality_tier = str(sizing.get("quality_tier") or "").lower().strip()
    plan_complete = bool(plan.get("is_complete_for_real_trade"))
    tradeable_probe = bool(evidence_score.get("tradeable_probe"))
    evidence_tier = str(evidence_score.get("tier") or "").lower().strip()
    evidence_shadow_only = bool(evidence_score.get("shadow_only"))
    evidence_size_multiplier = _safe_float(evidence_score.get("size_multiplier"), 0.0)
    strategy_quality_override = bool(sizing.get("strategy_quality_override"))
    meaningful_size_reason = str(sizing.get("meaningful_size_reason") or "").strip()
    notional_floor_applied = bool(sizing.get("notional_floor_applied"))
    target_min_notional = _safe_float(sizing.get("target_min_notional_usdt"), 0.0)
    final_notional = _safe_float(sizing.get("final_notional_usdt"), 0.0)
    position_size_pct = max(
        _safe_float(sizing.get("position_size_pct")),
        _safe_float(getattr(decision, "position_size_pct", None)),
    )
    expected_net_return = max(
        _safe_float(opportunity.get("expected_net_return_pct"), 0.0),
        _safe_float(plan.get("expected_net_return_pct"), 0.0),
    )
    expected_profit = max(
        _safe_float(sizing.get("expected_profit_usdt"), 0.0),
        _safe_float(plan.get("expected_profit_usdt"), 0.0),
    )
    if expected_profit <= 0.0 and final_notional > 0.0 and expected_net_return > 0.0:
        expected_profit = final_notional * expected_net_return / 100.0
    final_leverage = _safe_float(
        dynamic.get("final_integer_leverage"),
        _safe_float(getattr(decision, "suggested_leverage", None), 0.0),
    )
    structure_reasons: list[str] = []
    if max(_lane_rank(plan_lane), _lane_rank(ladder_lane)) >= _lane_rank("validated_probe"):
        structure_reasons.append("validated_profit_first_lane")
    if tradeable_probe:
        structure_reasons.append("tradeable_probe")
    if (
        evidence_tier in {"exploration", "small", "medium", "normal"}
        and not evidence_shadow_only
        and evidence_size_multiplier > 0.0
    ):
        structure_reasons.append(f"evidence_tier:{evidence_tier}")
    if strategy_quality_override:
        structure_reasons.append("strategy_quality_override")
    if quality_tier in {
        "good_probe",
        "strong_probe",
        "quality_override",
        "roster_fill",
        "high_profit",
        "elite",
        "winner_add",
    }:
        structure_reasons.append(f"quality_tier:{quality_tier}")
    if meaningful_size_reason:
        structure_reasons.append("meaningful_size_reason")
    if notional_floor_applied or target_min_notional > 0.0:
        structure_reasons.append("notional_floor")

    economically_viable = bool(
        plan_complete
        and expected_net_return > 0.0
        and position_size_pct > 0.0
        and final_leverage > 0.0
        and (
            expected_profit > 0.0
            or final_notional > 0.0
            or _lane_rank(plan_lane) >= _lane_rank("validated_probe")
        )
    )
    ready = bool(economically_viable and structure_reasons)
    return {
        "ready": ready,
        "economically_viable": economically_viable,
        "plan_complete": plan_complete,
        "plan_lane": plan_lane,
        "ladder_lane": ladder_lane,
        "quality_tier": quality_tier,
        "tradeable_probe": tradeable_probe,
        "evidence_tier": evidence_tier,
        "evidence_shadow_only": evidence_shadow_only,
        "evidence_size_multiplier": evidence_size_multiplier,
        "strategy_quality_override": strategy_quality_override,
        "meaningful_size_reason": meaningful_size_reason,
        "notional_floor_applied": notional_floor_applied,
        "target_min_notional_usdt": target_min_notional,
        "final_notional_usdt": final_notional,
        "position_size_pct": position_size_pct,
        "final_integer_leverage": final_leverage,
        "expected_net_return_pct": expected_net_return,
        "expected_profit_usdt": expected_profit,
        "structure_reasons": structure_reasons,
    }
