"""Entry high-risk review gate.

This module owns the business rule that decides when an entry must call the
online high-risk reviewer.  The reviewer service itself still owns model runtime
details such as API calls, non-thinking controls, parsing, redaction, and
circuit breaking.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from core.safe_output import safe_error_text
from core.url_safety import normalize_http_base_url
from services.entry_direction_metrics import selected_entry_metrics
from services.high_risk_review_service import HighRiskReviewService

QUANT_PROFIT_PROBE_MIN_PROFIT_QUALITY_RATIO = 0.12
LOCAL_CONTROLLED_PROBE_MIN_EXPECTED_NET_RETURN_PCT = 0.70
LOCAL_CONTROLLED_PROBE_MIN_PROFIT_QUALITY_RATIO = 0.90
LOCAL_CONTROLLED_PROBE_MAX_LOSS_PROBABILITY = 0.58
LOCAL_CONTROLLED_PROBE_MAX_TAIL_RISK_SCORE = 0.75
LOCAL_CONTROLLED_PROBE_MAX_SIZE_PCT = 0.04
LOCAL_CONTROLLED_LOW_PAYOFF_PROBE_MAX_SIZE_PCT = 0.025
LOCAL_CONTROLLED_PROBE_MAX_LEVERAGE = 5.0
LOCAL_CONTROLLED_PROBE_ML_CONFLICT_MIN_EXPECTED_NET_RETURN_PCT = 0.85
LOCAL_CONTROLLED_PROBE_ML_CONFLICT_MIN_PROFIT_QUALITY_RATIO = 1.0
LOCAL_CONTROLLED_PROBE_EVIDENCE_TIERS = {"exploration", "small", "medium", "normal"}
LOCAL_CONTROLLED_PROBE_QUALITY_TIERS = {
    "probe",
    "good_probe",
    "strong_probe",
    "roster_fill",
    "quality_override",
    "high_profit",
    "elite",
    "winner_add",
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def entry_side_value(decision: DecisionOutput) -> str:
    if decision.action == Action.LONG:
        return "long"
    if decision.action == Action.SHORT:
        return "short"
    return "hold"


def entry_expert_disagreement(decision: DecisionOutput) -> float:
    raw = _safe_dict(decision.raw_response)
    opinions = _safe_list(raw.get("opinions"))
    side = entry_side_value(decision)
    opposite = "short" if side == "long" else "long"
    directional = 0
    opposite_votes = 0
    for opinion in opinions:
        if not isinstance(opinion, dict):
            continue
        action = str(opinion.get("action") or "").lower()
        if action in {"long", "short"}:
            directional += 1
            if action == opposite:
                opposite_votes += 1
    return opposite_votes / directional if directional else 0.0


def ml_ai_direction_conflict(decision: DecisionOutput) -> bool:
    raw = _safe_dict(decision.raw_response)
    ml_signal = _safe_dict(raw.get("ml_signal"))
    predictions = _safe_list(ml_signal.get("predictions"))
    primary = _safe_dict(predictions[0]) if predictions else {}
    ml_side = str(primary.get("best_side") or "").lower()
    return ml_side in {"long", "short"} and ml_side != entry_side_value(decision)


def _local_controlled_probe_sources(
    *,
    raw: dict[str, Any],
    opportunity: dict[str, Any],
    quant_probe: dict[str, Any],
) -> list[str]:
    sources: list[str] = []
    if quant_probe.get("triggered"):
        sources.append("quant_profit_probe")
    evidence_probe = _safe_dict(raw.get("evidence_profit_probe"))
    if evidence_probe.get("triggered"):
        sources.append("evidence_profit_probe")

    profit_sizing = _safe_dict(raw.get("profit_risk_sizing"))
    quality_tier = str(profit_sizing.get("quality_tier") or "").lower()
    if quality_tier in LOCAL_CONTROLLED_PROBE_QUALITY_TIERS:
        sources.append(f"sizing:{quality_tier}")

    evidence_score = _safe_dict(opportunity.get("evidence_score"))
    evidence_tier = str(evidence_score.get("tier") or opportunity.get("evidence_tier") or "")
    evidence_tier = evidence_tier.lower()
    if (
        evidence_tier in LOCAL_CONTROLLED_PROBE_EVIDENCE_TIERS
        and not evidence_score.get("shadow_only")
        and not evidence_score.get("hard_block")
    ):
        sources.append(f"evidence:{evidence_tier}")
    return sources


def _local_controlled_probe_payload(
    *,
    raw: dict[str, Any],
    opportunity: dict[str, Any],
    quant_probe: dict[str, Any],
    expected_net: float,
    profit_quality: float,
    loss_probability: float,
    tail_risk: float,
    size_pct: float,
    leverage: float,
    ml_conflict: bool,
    advisory_reasons: list[str],
) -> dict[str, Any] | None:
    """Return metadata when a small positive probe should stay under local controls."""

    sources = _local_controlled_probe_sources(
        raw=raw,
        opportunity=opportunity,
        quant_probe=quant_probe,
    )
    if not sources:
        return None
    evidence_score = _safe_dict(opportunity.get("evidence_score"))
    if evidence_score.get("shadow_only") or evidence_score.get("hard_block"):
        return None
    profit_sizing = _safe_dict(raw.get("profit_risk_sizing"))
    low_payoff_quality = bool(profit_sizing.get("low_payoff_quality"))
    if expected_net < LOCAL_CONTROLLED_PROBE_MIN_EXPECTED_NET_RETURN_PCT:
        return None
    if profit_quality < LOCAL_CONTROLLED_PROBE_MIN_PROFIT_QUALITY_RATIO:
        return None
    if loss_probability > LOCAL_CONTROLLED_PROBE_MAX_LOSS_PROBABILITY:
        return None
    if tail_risk > LOCAL_CONTROLLED_PROBE_MAX_TAIL_RISK_SCORE:
        return None
    if leverage > LOCAL_CONTROLLED_PROBE_MAX_LEVERAGE:
        return None
    if size_pct > LOCAL_CONTROLLED_PROBE_MAX_SIZE_PCT:
        return None
    if low_payoff_quality and size_pct > LOCAL_CONTROLLED_LOW_PAYOFF_PROBE_MAX_SIZE_PCT:
        return None
    if ml_conflict and (
        expected_net < LOCAL_CONTROLLED_PROBE_ML_CONFLICT_MIN_EXPECTED_NET_RETURN_PCT
        or profit_quality < LOCAL_CONTROLLED_PROBE_ML_CONFLICT_MIN_PROFIT_QUALITY_RATIO
    ):
        return None

    return {
        "triggered": False,
        "approved": True,
        "status": "skipped_local_controlled_probe",
        "rule": "positive small probe stays under local risk controls",
        "probe_sources": sources,
        "advisory_reasons": advisory_reasons,
        "low_payoff_quality": low_payoff_quality,
        "reason": (
            "正期望小仓探针已通过本地收益/风险阈值，今日亏损恢复或专家分歧只作为风险提示；"
            "本次不调用线上高风险复核，继续由本地仓位、杠杆、止损预算和执行检查控制风险。"
        ),
        "expected_net_return_pct": round(expected_net, 6),
        "profit_quality_ratio": round(profit_quality, 6),
        "loss_probability": round(loss_probability, 6),
        "tail_risk_score": round(tail_risk, 6),
        "position_size_pct": round(size_pct, 6),
        "leverage": round(leverage, 4),
    }


def _review_opportunity_summary(opportunity: dict[str, Any]) -> dict[str, Any]:
    breakdown = _safe_dict(opportunity.get("expected_net_breakdown"))
    components = breakdown.get("components") if isinstance(breakdown, dict) else []
    compact_components: list[dict[str, Any]] = []
    if isinstance(components, list):
        for component in components[:8]:
            if not isinstance(component, dict):
                continue
            compact_components.append(
                {
                    "key": component.get("key"),
                    "available": bool(component.get("available")),
                    "side": component.get("side"),
                    "raw_return_pct": round(_safe_float(component.get("raw_return_pct"), 0.0), 6),
                    "weight": round(_safe_float(component.get("weight"), 0.0), 6),
                    "contribution_pct": round(
                        _safe_float(component.get("contribution_pct"), 0.0), 6
                    ),
                }
            )
    return {
        "expected_net_return_pct": round(
            _safe_float(opportunity.get("expected_net_return_pct"), 0.0), 6
        ),
        "profit_quality_ratio": round(_safe_float(opportunity.get("profit_quality_ratio"), 0.0), 6),
        "server_profit_loss_probability": round(
            _safe_float(opportunity.get("server_profit_loss_probability"), 0.5), 6
        ),
        "tail_risk_score": round(_safe_float(opportunity.get("tail_risk_score"), 0.0), 6),
        "reward_risk_ratio": round(_safe_float(opportunity.get("reward_risk_ratio"), 0.0), 6),
        "server_profit_expected_return_pct": round(
            _safe_float(opportunity.get("server_profit_expected_return_pct"), 0.0), 6
        ),
        "ml_expected_return_pct": round(
            _safe_float(opportunity.get("ml_expected_return_pct"), 0.0), 6
        ),
        "timeseries_expected_return_pct": round(
            _safe_float(opportunity.get("timeseries_expected_return_pct"), 0.0), 6
        ),
        "expected_net_breakdown": {
            "formula": breakdown.get("formula"),
            "net_pct": round(_safe_float(breakdown.get("net_pct"), 0.0), 6),
            "model_net_pct": round(_safe_float(breakdown.get("model_net_pct"), 0.0), 6),
            "components": compact_components,
        },
    }


@dataclass(slots=True)
class EntryHighRiskReviewGatePolicy:
    """Apply high-risk review business rules for entry decisions."""

    reviewer: Any | None = None
    allocation_state_provider: Callable[[str], Awaitable[dict[str, Any]]] | None = None
    config: Any = field(default_factory=lambda: settings)
    quant_probe_min_profit_quality_ratio: float = QUANT_PROFIT_PROBE_MIN_PROFIT_QUALITY_RATIO

    async def evaluate(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None,
    ) -> str | None:
        if not decision.is_entry or not self.config.high_risk_review_enabled:
            return None

        raw: dict[str, Any] = _safe_dict(decision.raw_response)
        side = entry_side_value(decision)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        quant_probe = _safe_dict(raw.get("quant_profit_probe"))
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        selected_metrics = selected_entry_metrics(decision)
        if selected_metrics.has_selected_side:
            expected_net = selected_metrics.expected_net_return_pct
            profit_quality = selected_metrics.profit_quality_ratio
        tail_risk = _safe_float(opportunity.get("tail_risk_score"), 0.0)
        if selected_metrics.has_selected_side:
            tail_risk = selected_metrics.tail_risk_score
        loss_probability = _safe_float(
            quant_probe.get(
                "loss_probability",
                opportunity.get("server_profit_loss_probability"),
            ),
            1.0,
        )
        if selected_metrics.has_selected_side:
            loss_probability = selected_metrics.loss_probability
        if (
            quant_probe.get("triggered")
            and expected_net > 0
            and profit_quality >= self.quant_probe_min_profit_quality_ratio
            and loss_probability < 0.58
            and tail_risk <= LOCAL_CONTROLLED_PROBE_MAX_TAIL_RISK_SCORE
            and _safe_float(decision.position_size_pct, 0.0) <= 0.04
            and _safe_float(decision.suggested_leverage, 1.0) <= 5.0
        ):
            raw["high_risk_review"] = {
                "triggered": False,
                "skipped_for_quant_probe": True,
                "reason": "正期望小仓量化探针已通过机会评分，跳过在线高风险复核，交由本地风控和执行检查控制风险。",
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(profit_quality, 6),
                "loss_probability": round(loss_probability, 6),
                "tail_risk_score": round(tail_risk, 6),
            }
            decision.raw_response = raw
            return None

        reasons: list[str] = []
        leverage = _safe_float(decision.suggested_leverage, 1.0)
        size_pct = _safe_float(decision.position_size_pct, 0.0)
        disagreement = entry_expert_disagreement(decision)
        ml_conflict = ml_ai_direction_conflict(decision)
        if leverage >= 8.0:
            reasons.append(f"high_leverage:{leverage:.1f}x")
        if size_pct >= 0.10:
            reasons.append(f"large_position:{size_pct:.1%}")
        if disagreement >= 0.34:
            reasons.append(f"expert_disagreement:{disagreement:.0%}")
        if ml_conflict:
            reasons.append("ml_ai_direction_conflict")
        symbol_profile = opportunity.get("symbol_side_profile")
        if isinstance(symbol_profile, dict) and _safe_float(symbol_profile.get("pnl"), 0.0) < -10:
            reasons.append("recent_symbol_side_loss")
        ml_gate = _safe_dict(raw.get("ml_profit_quality_gate"))
        local_gate = _safe_dict(ml_gate.get("local_ai_tools_gate"))
        if ml_gate.get("local_quant_caution") or str(ml_gate.get("status") or "").startswith(
            "soft_"
        ):
            reasons.append(
                str(local_gate.get("status") or ml_gate.get("status") or "local_quant_caution")
            )

        allocation_state: dict[str, Any] = {}
        if self.allocation_state_provider is not None:
            try:
                allocation_state = await self.allocation_state_provider(model_mode)
                if _safe_float(allocation_state.get("today_risk_pnl"), 0.0) < 0:
                    reasons.append("today_recovery_after_loss")
            except Exception:
                allocation_state = {}

        local_probe_review = _local_controlled_probe_payload(
            raw=raw,
            opportunity=opportunity,
            quant_probe=quant_probe,
            expected_net=expected_net,
            profit_quality=profit_quality,
            loss_probability=loss_probability,
            tail_risk=tail_risk,
            size_pct=size_pct,
            leverage=leverage,
            ml_conflict=ml_conflict,
            advisory_reasons=reasons,
        )
        if local_probe_review is not None:
            raw["high_risk_review"] = local_probe_review
            decision.raw_response = raw
            return None

        hard_review_required = bool(
            leverage >= 10.0
            or size_pct >= 0.12
            or disagreement >= 0.50
            or (ml_conflict and expected_net <= 0.35)
            or loss_probability >= 0.72
        )

        if not reasons:
            raw["high_risk_review"] = {
                "triggered": False,
                "reasons": [],
                "rule": "only high-risk entries call online reviewer",
            }
            decision.raw_response = raw
            return None

        if not hard_review_required:
            raw["high_risk_review"] = {
                "triggered": False,
                "advisory_reasons": reasons,
                "approved": True,
                "status": "skipped_advisory_only",
                "rule": "online reviewer only has veto power for truly high-risk entries",
                "reason": (
                    "存在历史表现、今日亏损恢复或本地模型谨慎等风险提示，但未达到必须线上复核的级别；"
                    "本次不调用线上高风险复核，不再用普通弱证据否决 AI 开仓。"
                ),
            }
            decision.raw_response = raw
            return None

        review = {
            "triggered": True,
            "reasons": reasons,
            "model": self.config.high_risk_review_model,
            "api_base": "pending_validation",
            "approved": None,
            "status": "pending",
            "hard_review_required": hard_review_required,
        }
        raw["high_risk_review"] = review
        decision.raw_response = raw

        reviewer = self.reviewer or HighRiskReviewService(self.config)
        try:
            api_base = normalize_http_base_url(
                self.config.high_risk_review_api_base,
                field_name="High-risk review API base",
            )
        except ValueError as exc:
            api_base = ""
            review.update(
                {
                    "api_base": "invalid",
                    "status": "skipped_blocked",
                    "approved": False,
                    "reason": f"高风险复核地址配置无效：{exc}",
                }
            )
            raw["high_risk_review"] = review
            decision.raw_response = raw
            return str(review["reason"])

        review["api_base"] = api_base
        api_key = reviewer.api_key(api_base)
        model = str(self.config.high_risk_review_model or "").strip()
        if not api_base or not model or not api_key:
            review.update(
                {
                    "status": "skipped_blocked",
                    "approved": False,
                    "reason": "高风险复核未完整配置，必须线上复核的开仓暂不提交。",
                }
            )
            raw["high_risk_review"] = review
            decision.raw_response = raw
            return str(review["reason"])

        circuit_open = reviewer.circuit_payload()
        if circuit_open:
            review.update(circuit_open)
            raw["high_risk_review"] = review
            decision.raw_response = raw
            return str(review["reason"])

        prompt = {
            "symbol": decision.symbol,
            "side": side,
            "confidence": decision.confidence,
            "position_size_pct": decision.position_size_pct,
            "leverage": decision.suggested_leverage,
            "stop_loss_pct": decision.stop_loss_pct,
            "take_profit_pct": decision.take_profit_pct,
            "trigger_reasons": reasons,
            "opportunity_score": _review_opportunity_summary(opportunity),
            "today_pnl": allocation_state.get("today_risk_pnl"),
            "open_position_count": len([p for p in open_positions or [] if p.get("is_open", True)]),
        }
        try:
            result = await reviewer.review_trade(
                prompt,
                api_base=api_base,
                api_key=api_key,
                model=model,
            )
            review.update(
                {
                    "status": "completed",
                    "approved": result.approved,
                    "confidence": result.confidence,
                    "reason": result.reason,
                    "attempts": result.attempts,
                }
            )
            raw["high_risk_review"] = review
            decision.raw_response = raw
            if not result.approved:
                return (
                    f"高风险复核否决："
                    f"{review.get('reason') or '线上复核认为该交易盈亏比或证据质量不足'}"
                )
        except Exception as exc:
            error = safe_error_text(exc, limit=180)
            reviewer.record_failure(error)
            if hard_review_required:
                review.update(
                    {
                        "status": "error_blocked",
                        "approved": False,
                        "error": error,
                        "reason": (
                            f"高风险复核调用失败：{error}。"
                            "本次属于必须线上复核的大仓/高杠杆/严重冲突开仓，未完成复核前不提交订单。"
                        ),
                    }
                )
                raw["high_risk_review"] = review
                decision.raw_response = raw
                return str(review["reason"])
            original_size = _safe_float(decision.position_size_pct, 0.0)
            original_leverage = _safe_float(decision.suggested_leverage, 1.0)
            decision.position_size_pct = min(original_size or 0.02, 0.025)
            decision.suggested_leverage = min(original_leverage or 3.0, 5.0)
            review.update(
                {
                    "status": "error_downgraded",
                    "approved": True,
                    "error": error,
                    "reason": (
                        f"高风险复核调用失败：{error}。"
                        "这笔不是必须线上复核的极端高风险交易，已降仓降杠杆后继续走本地风控和 OKX 提交检查。"
                    ),
                    "original_position_size_pct": round(original_size, 6),
                    "adjusted_position_size_pct": round(
                        float(decision.position_size_pct or 0.0), 6
                    ),
                    "original_leverage": round(original_leverage, 4),
                    "adjusted_leverage": round(float(decision.suggested_leverage or 1.0), 4),
                }
            )
            raw["high_risk_review"] = review
            decision.raw_response = raw
            return None
        return None
