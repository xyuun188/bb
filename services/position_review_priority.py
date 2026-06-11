"""Priority scoring for fast position-review triage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from services.exit_fast_risk import FAST_RISK_NEAR_STOP_PROGRESS
from services.exit_predictive_reversal import (
    PREDICTIVE_REVERSAL_EXIT_SCORE,
    PREDICTIVE_REVERSAL_REVIEW_SCORE,
    ExitPredictiveReversalPolicy,
)
from services.trading_params import ESTIMATED_TAKER_FEE_PCT

PROFIT_PROTECTION_MIN_NET_PNL_RATIO = 0.004
PROFIT_PROTECTION_MIN_NET_USDT = 3.00
PROFIT_PROTECTION_MIN_FEE_MULTIPLE = 4.0
PORTFOLIO_PROFIT_PROTECTION_EXIT_SCORE = 82.0

NormalizeSymbol = Callable[[Any], str]
PositionPeakKeyProvider = Callable[[str, str, str], Any]
PositionPeaksProvider = Callable[[], Mapping[Any, dict[str, Any]]]
AggregatePositionGroup = Callable[[list[dict[str, Any]], str, str, str], dict[str, Any]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class PositionReviewPriorityPolicy:
    """Score position groups before spending slow AI review time."""

    normalize_symbol: NormalizeSymbol
    position_peak_key: PositionPeakKeyProvider
    position_peaks_provider: PositionPeaksProvider
    predictive_reversal: ExitPredictiveReversalPolicy
    urgent_exit_markers: tuple[str, ...] = ()

    def portfolio_profit_protection_score(
        self,
        context: dict[str, Any],
        model_name: str,
        symbol: str,
    ) -> tuple[float, list[str]]:
        if not isinstance(context, dict) or not context.get("active"):
            return 0.0, []
        normalized = self.normalize_symbol(symbol)
        model = str(model_name or "")
        for item in context.get("focus_groups", []):
            if (
                item.get("model_name") == model
                and self.normalize_symbol(item.get("symbol")) == normalized
            ):
                return PORTFOLIO_PROFIT_PROTECTION_EXIT_SCORE, ["portfolio_profit_protection_focus"]
        return 0.0, []

    def fast_position_exit_score(
        self,
        pos: dict[str, Any],
        feature_vector: Any | None,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        try:
            entry = float(pos.get("entry_price") or 0.0)
            current = float(pos.get("current_price") or entry or 0.0)
            qty = abs(float(pos.get("quantity") or 0.0))
            stop = float(pos.get("stop_loss") or pos.get("stop_loss_price") or 0.0)
            unrealized = float(pos.get("unrealized_pnl") or 0.0)
        except (TypeError, ValueError):
            return 0.0, reasons
        if entry <= 0 or current <= 0 or qty <= 0:
            return 0.0, reasons

        side = str(pos.get("side") or "").lower()
        notional = max(entry * qty, 1e-9)
        pnl_ratio = unrealized / notional
        estimated_round_trip_fee = max(notional * ESTIMATED_TAKER_FEE_PCT * 2.0, 1e-9)
        score = 0.0
        if pnl_ratio <= -0.02 or unrealized <= -8.0:
            score = max(score, 95.0)
            reasons.append("loss_expanding")
        elif pnl_ratio <= -0.01 or unrealized <= -3.0:
            score = max(score, 82.0)
            reasons.append("loss_needs_review")
        elif pnl_ratio <= -0.006 or unrealized <= -1.2:
            score = max(score, 70.0)
            reasons.append("loss_watch")

        if stop > 0:
            if side == "short":
                total_stop_distance = max(stop - entry, 0.0)
                used_distance = max(current - entry, 0.0)
            else:
                total_stop_distance = max(entry - stop, 0.0)
                used_distance = max(entry - current, 0.0)
            if total_stop_distance > 0:
                stop_progress = used_distance / total_stop_distance
                if stop_progress >= 0.85:
                    score = max(score, 96.0)
                    reasons.append("near_stop")
                elif stop_progress >= FAST_RISK_NEAR_STOP_PROGRESS:
                    score = max(score, 78.0)
                    reasons.append("stop_risk_rising")

        peak_key = self.position_peak_key(
            str(pos.get("model_name") or ""),
            str(pos.get("symbol") or ""),
            side,
        )
        peak_state = self.position_peaks_provider().get(peak_key, {})
        peak_pnl = _safe_float(
            peak_state.get("peak_unrealized_pnl", peak_state.get("peak_pnl")),
            0.0,
        )
        if unrealized >= max(
            notional * PROFIT_PROTECTION_MIN_NET_PNL_RATIO,
            estimated_round_trip_fee * PROFIT_PROTECTION_MIN_FEE_MULTIPLE,
            PROFIT_PROTECTION_MIN_NET_USDT,
        ):
            score = max(score, 72.0)
            reasons.append("profit_lock_candidate")
        if peak_pnl >= 0.8 and unrealized > 0 and unrealized <= peak_pnl * 0.72:
            score = max(score, 80.0)
            retrace_ratio = (peak_pnl - unrealized) / max(peak_pnl, 1e-9)
            reasons.append(f"profit_retrace:{peak_pnl:.2f}->{unrealized:.2f}U/{retrace_ratio:.0%}")

        if feature_vector is not None:
            self._apply_feature_exit_score(feature_vector, side, reasons, score_ref := [score])
            score = score_ref[0]

        return score, reasons

    def _apply_feature_exit_score(
        self,
        feature_vector: Any,
        side: str,
        reasons: list[str],
        score_ref: list[float],
    ) -> None:
        try:
            returns_1 = float(getattr(feature_vector, "returns_1", 0.0) or 0.0)
            returns_5 = float(getattr(feature_vector, "returns_5", 0.0) or 0.0)
            returns_20 = float(getattr(feature_vector, "returns_20", 0.0) or 0.0)
            volume_ratio = float(getattr(feature_vector, "volume_ratio", 1.0) or 1.0)
            rsi_14 = float(getattr(feature_vector, "rsi_14", 50.0) or 50.0)
            bb_pct = float(getattr(feature_vector, "bb_pct", 0.5) or 0.5)
            macd_diff = float(getattr(feature_vector, "macd_diff", 0.0) or 0.0)
            adx_14 = float(getattr(feature_vector, "adx_14", 0.0) or 0.0)
        except (TypeError, ValueError):
            returns_1 = returns_5 = returns_20 = 0.0
            volume_ratio = 1.0
            rsi_14 = 50.0
            bb_pct = 0.5
            macd_diff = 0.0
            adx_14 = 0.0
        adverse_1 = returns_1 <= -0.012 if side == "long" else returns_1 >= 0.012
        adverse_5 = returns_5 <= -0.025 if side == "long" else returns_5 >= 0.025
        if volume_ratio >= 1.1 and (adverse_1 or adverse_5):
            score_ref[0] = max(score_ref[0], 84.0)
            reasons.append("adverse_momentum")
        reversal = self.predictive_reversal.evidence(
            side=side,
            returns_1=returns_1,
            returns_5=returns_5,
            returns_20=returns_20,
            volume_ratio=volume_ratio,
            rsi_14=rsi_14,
            bb_pct=bb_pct,
            macd_diff=macd_diff,
            adx_14=adx_14,
        )
        reversal_score = _safe_float(reversal.get("score"), 0.0)
        if reversal_score >= PREDICTIVE_REVERSAL_EXIT_SCORE:
            score_ref[0] = max(score_ref[0], 88.0)
            reasons.append(f"predictive_reversal:{reversal_score:.0f}")
        elif reversal_score >= PREDICTIVE_REVERSAL_REVIEW_SCORE:
            score_ref[0] = max(score_ref[0], 76.0)
            reasons.append(f"reversal_watch:{reversal_score:.0f}")

    def fast_position_add_score(
        self,
        positions: list[dict[str, Any]],
        feature_vector: Any | None,
    ) -> tuple[float, str | None]:
        if not positions or feature_vector is None:
            return 0.0, None
        sides = {
            str(pos.get("side") or "").lower()
            for pos in positions
            if str(pos.get("side") or "").lower() in {"long", "short"}
        }
        if len(sides) != 1:
            return 0.0, None
        side = next(iter(sides))
        try:
            returns_1 = float(getattr(feature_vector, "returns_1", 0.0) or 0.0)
            returns_5 = float(getattr(feature_vector, "returns_5", 0.0) or 0.0)
            returns_20 = float(getattr(feature_vector, "returns_20", 0.0) or 0.0)
            volume_ratio = float(getattr(feature_vector, "volume_ratio", 1.0) or 1.0)
            adx_14 = float(getattr(feature_vector, "adx_14", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0, None

        total_unrealized = sum(_safe_float(pos.get("unrealized_pnl"), 0.0) for pos in positions)
        total_notional = sum(
            abs(_safe_float(pos.get("entry_price"), 0.0) * _safe_float(pos.get("quantity"), 0.0))
            for pos in positions
        )
        pnl_ratio = total_unrealized / max(total_notional, 1e-9)
        same_direction = (
            returns_1 > 0.0015 and returns_5 > 0.006 and returns_20 > 0.010
            if side == "long"
            else returns_1 < -0.0015 and returns_5 < -0.006 and returns_20 < -0.010
        )
        winner_direction = (
            total_unrealized >= 1.2
            and pnl_ratio >= 0.0012
            and (
                (returns_5 > 0.002 and returns_20 > 0.003)
                if side == "long"
                else (returns_5 < -0.002 and returns_20 < -0.003)
            )
        )
        if not same_direction and not winner_direction:
            return 0.0, None

        score = 62.0 if winner_direction else 58.0
        if volume_ratio >= 1.2:
            score += 8.0
        if adx_14 >= 24.0:
            score += 8.0
        if total_unrealized >= 3.0:
            score += 10.0
        elif total_unrealized >= 1.2:
            score += 6.0
        return min(score, 88.0), (
            "winner_add_candidate" if winner_direction else "trend_add_candidate"
        )

    def scan_groups(
        self,
        grouped_items: list[tuple[tuple[str, str], list[dict[str, Any]]]],
        feature_vectors: dict[str, Any],
        portfolio_profit_context: dict[str, Any] | None,
        *,
        aggregate_position_group: AggregatePositionGroup,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        scans: dict[tuple[str, str], dict[str, Any]] = {}
        for key, positions in grouped_items:
            symbol = key[1]
            normalized = self.normalize_symbol(symbol)
            fv = feature_vectors.get(symbol) or feature_vectors.get(normalized)
            exit_score = 0.0
            add_score = 0.0
            reasons: list[str] = []

            by_side: dict[str, list[dict[str, Any]]] = {}
            for pos in positions or []:
                side = str(pos.get("side") or "").lower()
                if side in {"long", "short"}:
                    by_side.setdefault(side, []).append(pos)

            for side, side_positions in by_side.items():
                aggregate = aggregate_position_group(
                    side_positions, key[0], normalized or symbol, side
                )
                if not aggregate:
                    continue
                pos_exit_score, pos_reasons = self.fast_position_exit_score(aggregate, fv)
                if pos_exit_score > exit_score:
                    exit_score = pos_exit_score
                reasons.extend(pos_reasons)

            add_score, add_reason = self.fast_position_add_score(positions, fv)
            if add_reason:
                reasons.append(add_reason)

            portfolio_score, portfolio_reasons = self.portfolio_profit_protection_score(
                portfolio_profit_context or {},
                key[0],
                normalized,
            )
            if portfolio_score > exit_score:
                exit_score = portfolio_score
            reasons.extend(portfolio_reasons)

            priority_score = max(exit_score, add_score)
            scans[key] = {
                "priority_score": priority_score,
                "exit_score": exit_score,
                "add_score": add_score,
                "reason": "; ".join(dict.fromkeys(reasons))[:260],
            }
        return scans

    def is_urgent_exit_scan(self, scan: dict[str, Any] | None) -> bool:
        if not isinstance(scan, dict):
            return False
        exit_score = _safe_float(scan.get("exit_score"), 0.0)
        reason = str(scan.get("reason") or "")
        if exit_score >= 90.0:
            return True
        return any(marker in reason for marker in self.urgent_exit_markers)
