"""Dynamic position-review prioritization.

This module only orders reviews. It cannot create an add, reduce, or close
permission independently from the unified dynamic exit policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.dynamic_exit_policy import assess_dynamic_exit

NormalizeSymbol = Callable[[Any], str]
PositionPeakKeyProvider = Callable[[str, str, str], Any]
PositionPeaksProvider = Callable[[], Mapping[Any, dict[str, Any]]]
AggregatePositionGroup = Callable[[list[dict[str, Any]], str, str, str], dict[str, Any]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(slots=True)
class PositionReviewPriorityPolicy:
    """Order groups by the current dynamic exit fraction."""

    normalize_symbol: NormalizeSymbol
    position_peak_key: PositionPeakKeyProvider
    position_peaks_provider: PositionPeaksProvider

    def _assessment(
        self,
        position: dict[str, Any],
        feature_vector: Any | None,
    ) -> dict[str, Any]:
        side = str(position.get("side") or "").lower()
        if side not in {"long", "short"}:
            return {}
        snapshot = dict(position)
        feature_price = _safe_float(
            getattr(feature_vector, "current_price", 0.0) if feature_vector is not None else 0.0,
            0.0,
        )
        if feature_price > 0.0:
            snapshot["current_price"] = feature_price
        peak_key = self.position_peak_key(
            str(position.get("model_name") or ""),
            str(position.get("symbol") or ""),
            side,
        )
        peak_state = _safe_dict(self.position_peaks_provider().get(peak_key, {}))
        snapshot["peak_unrealized_pnl"] = _safe_float(
            peak_state.get("peak_unrealized_pnl"),
            _safe_float(position.get("peak_unrealized_pnl"), 0.0),
        )
        returns = [
            _safe_float(
                getattr(feature_vector, name, 0.0) if feature_vector is not None else 0.0,
                0.0,
            )
            for name in ("returns_1", "returns_5", "returns_20")
        ]
        adverse = any(
            (side == "long" and value < 0.0) or (side == "short" and value > 0.0)
            for value in returns
        )
        decision = DecisionOutput(
            model_name=str(position.get("model_name") or "ensemble_trader"),
            symbol=self.normalize_symbol(position.get("symbol")),
            action=Action.CLOSE_LONG if side == "long" else Action.CLOSE_SHORT,
            confidence=0.0,
            reasoning="dynamic position review priority",
            position_size_pct=0.0,
            suggested_leverage=1.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            raw_response={
                "close_evidence": {
                    "continuation_deteriorated": adverse,
                    "peak_unrealized_pnl_usdt": snapshot["peak_unrealized_pnl"],
                }
            },
            feature_snapshot=(
                feature_vector.to_dict()
                if feature_vector is not None
                and callable(getattr(feature_vector, "to_dict", None))
                else {}
            ),
        )
        return assess_dynamic_exit(decision, [snapshot]).to_dict()

    def scan_groups(
        self,
        grouped_items: list[tuple[tuple[str, str], list[dict[str, Any]]]],
        feature_vectors: dict[str, Any],
        portfolio_profit_context: dict[str, Any] | None,
        strategy_context: dict[str, Any] | None = None,
        *,
        aggregate_position_group: AggregatePositionGroup,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        del portfolio_profit_context, strategy_context
        scans: dict[tuple[str, str], dict[str, Any]] = {}
        for key, positions in grouped_items:
            symbol = key[1]
            normalized = self.normalize_symbol(symbol)
            feature = feature_vectors.get(symbol) or feature_vectors.get(normalized)
            assessments: list[dict[str, Any]] = []
            by_side: dict[str, list[dict[str, Any]]] = {}
            for position in positions or []:
                side = str(position.get("side") or "").lower()
                if side in {"long", "short"}:
                    by_side.setdefault(side, []).append(position)
            for side, side_positions in by_side.items():
                aggregate = aggregate_position_group(side_positions, key[0], normalized or symbol, side)
                if aggregate:
                    assessment = self._assessment(aggregate, feature)
                    if assessment:
                        assessments.append(assessment)

            eligible = [item for item in assessments if item.get("eligible") is True]
            best = max(
                eligible,
                key=lambda item: _safe_float(item.get("close_fraction"), 0.0),
                default={},
            )
            close_fraction = _safe_float(best.get("close_fraction"), 0.0)
            hard_risk = bool(best.get("hard_risk"))
            scans[key] = {
                "priority_score": close_fraction * 100.0,
                "exit_score": close_fraction * 100.0,
                "add_score": 0.0,
                "reason": str(best.get("reason") or "dynamic_exit_pressure_zero"),
                "force_exit_candidate": False,
                "release_action": "",
                "release_fraction": 0.0,
                "release_reason": "",
                "position_quality": {},
                "dynamic_exit_policy": best,
                "dynamic_exit_eligible": bool(best),
                "dynamic_exit_hard_risk": hard_risk,
            }
        return scans

    @staticmethod
    def is_urgent_exit_scan(scan: dict[str, Any] | None) -> bool:
        if not isinstance(scan, dict):
            return False
        return bool(scan.get("dynamic_exit_eligible"))
