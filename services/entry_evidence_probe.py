"""Convert strong entry evidence into controlled probe candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE
from services.entry_probe_market_quality import EntryProbeMarketQualityPolicy

MaxLeverageProvider = Callable[[], float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True, slots=True)
class EntryEvidenceProbePolicy:
    """Create a small entry candidate from evidence when the LLM stayed in HOLD."""

    model_name: str
    max_leverage_provider: MaxLeverageProvider
    market_quality: EntryProbeMarketQualityPolicy

    def create(
        self,
        original: DecisionOutput,
        feature_vector: Any,
        strategy: dict[str, Any] | None,
        ml_signal_context: dict[str, Any] | None,
        local_ai_tools_context: dict[str, Any] | None,
        direction_competition_context: dict[str, Any] | None,
    ) -> DecisionOutput | None:
        if not original.is_hold:
            return None
        raw = original.raw_response if isinstance(original.raw_response, dict) else {}
        evidence = _safe_dict(raw.get("entry_candidate_evidence"))
        if not evidence:
            return None
        sides = [side for side in ("long", "short") if isinstance(evidence.get(side), dict)]
        if not sides:
            return None

        side = max(sides, key=lambda candidate_side: self._score_side(evidence, candidate_side))
        item = _safe_dict(evidence.get(side))
        expected_net = _safe_float(item.get("expected_net_return_pct"), 0.0)
        quality = _safe_float(item.get("profit_quality_ratio"), 0.0)
        loss_probability = _safe_float(item.get("loss_probability"), 1.0)
        tail_risk = _safe_float(item.get("tail_risk_score"), 1.35)
        score = _safe_float(item.get("score"), -999.0)
        min_ref = _safe_float(item.get("min_score_reference"), MIN_ENTRY_OPPORTUNITY_SCORE)
        recommendation = str(item.get("recommendation") or "")
        high_profit = bool(item.get("high_profit_potential"))
        if expected_net < 0.30 or quality < 0.25 or loss_probability > 0.56 or tail_risk > 0.92:
            return None

        market_quality_block = self.market_quality.block_reason(feature_vector, side)
        if market_quality_block and not high_profit:
            raw["evidence_profit_probe_blocked"] = {
                "blocked": True,
                "reason": market_quality_block,
                "side": side,
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(quality, 6),
            }
            original.raw_response = raw
            return None
        if score < max(min_ref - 0.65, 0.20) and "high_profit" not in recommendation:
            return None
        if recommendation == "hold_or_tiny_probe_only" and not high_profit:
            return None

        sizing = self._sizing(expected_net, quality, loss_probability, tail_risk, high_profit)
        raw_response = dict(raw)
        raw_response.update(
            {
                "analysis_type": "market",
                "ml_signal": ml_signal_context or raw.get("ml_signal") or {},
                "local_ai_tools": local_ai_tools_context or raw.get("local_ai_tools") or {},
                "direction_competition": direction_competition_context
                or raw.get("direction_competition")
                or {},
                "evidence_profit_probe": {
                    "triggered": True,
                    "source": "entry_candidate_evidence",
                    "ai_original_action": original.action.value,
                    "side": side,
                    "expected_net_return_pct": round(expected_net, 6),
                    "profit_quality_ratio": round(quality, 6),
                    "loss_probability": round(loss_probability, 6),
                    "tail_risk_score": round(tail_risk, 6),
                    "score": round(score, 6),
                    "min_score_reference": round(min_ref, 6),
                    "high_profit_potential": high_profit,
                    "position_size_pct": round(sizing["size"], 6),
                    "suggested_leverage": round(sizing["leverage"], 6),
                    "reason": (
                        "AI 原始观望，但入场候选证据包显示该方向为正期望且风险可控，"
                        "生成受控探针候选。"
                    ),
                },
            }
        )
        side_label = "做多" if side == "long" else "做空"
        return DecisionOutput(
            model_name=self.model_name,
            symbol=original.symbol,
            action=Action.LONG if side == "long" else Action.SHORT,
            confidence=sizing["confidence"],
            reasoning=(
                f"AI 原始观望；证据包显示{side_label}正期望 {expected_net:.2f}%，"
                f"盈亏质量 {quality:.2f}，亏损概率 {loss_probability:.0%}，"
                "转为受控开仓候选。"
            ),
            position_size_pct=sizing["size"],
            suggested_leverage=min(sizing["leverage"], float(self.max_leverage_provider() or 1.0)),
            stop_loss_pct=sizing["stop_loss_pct"],
            take_profit_pct=sizing["take_profit_pct"],
            raw_response=raw_response,
            feature_snapshot=(
                feature_vector.to_dict()
                if hasattr(feature_vector, "to_dict")
                else (original.feature_snapshot or {})
            ),
        )

    @staticmethod
    def _score_side(evidence: dict[str, Any], side: str) -> float:
        item = _safe_dict(evidence.get(side))
        expected_net = _safe_float(item.get("expected_net_return_pct"), 0.0)
        quality = _safe_float(item.get("profit_quality_ratio"), 0.0)
        loss_probability = _safe_float(item.get("loss_probability"), 1.0)
        tail_risk = _safe_float(item.get("tail_risk_score"), 1.35)
        score = _safe_float(item.get("score"), -999.0)
        min_ref = _safe_float(item.get("min_score_reference"), MIN_ENTRY_OPPORTUNITY_SCORE)
        return (
            expected_net * 2.2
            + quality * 0.75
            + max(score - min_ref + 0.35, -1.0) * 0.40
            - max(loss_probability - 0.48, 0.0) * 1.8
            - max(tail_risk - 0.78, 0.0) * 1.4
        )

    @staticmethod
    def _sizing(
        expected_net: float,
        quality: float,
        loss_probability: float,
        tail_risk: float,
        high_profit: bool,
    ) -> dict[str, float]:
        if high_profit or expected_net >= 1.20:
            size = 0.075
            leverage = 8.0
            confidence = 0.76
        elif expected_net >= 0.65 and quality >= 0.65 and loss_probability <= 0.48:
            size = 0.055
            leverage = 6.0
            confidence = 0.70
        else:
            size = 0.035
            leverage = 5.0
            confidence = 0.64
        stop_loss_pct = 0.014 if tail_risk <= 0.78 else 0.012
        take_profit_pct = min(max(stop_loss_pct * 3.2, expected_net / 100.0 * 0.85), 0.10)
        return {
            "size": size,
            "leverage": leverage,
            "confidence": confidence,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
