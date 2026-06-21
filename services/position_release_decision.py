"""Deterministic release decisions for low-quality open-position groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.trading_params import DEFAULT_TRADING_PARAMS, ExitPositionQualityParams


@dataclass(frozen=True, slots=True)
class PositionReleaseDecisionPolicy:
    """Build explicit close decisions for positions marked by release triage."""

    min_release_exit_score: float = 90.0
    params: ExitPositionQualityParams = DEFAULT_TRADING_PARAMS.exit_position_quality

    def should_release(self, scan: dict[str, Any] | None) -> bool:
        if not isinstance(scan, dict):
            return False
        if self._fresh_low_quality_scan_is_protected(scan):
            return False
        if bool(scan.get("force_exit_candidate")):
            return True
        try:
            exit_score = float(scan.get("exit_score") or 0.0)
        except (TypeError, ValueError):
            exit_score = 0.0
        return bool(scan.get("release_action")) and exit_score >= self.min_release_exit_score

    def build(
        self,
        *,
        model_name: str,
        symbol: str,
        positions: list[dict[str, Any]],
        scan: dict[str, Any],
        feature_vector: Any | None,
    ) -> DecisionOutput | None:
        action = self._release_action(scan, positions)
        if action is None:
            return None
        if self._fresh_low_quality_scan_is_protected(scan):
            return None

        exit_score = self._safe_float(scan.get("exit_score"), 0.0)
        release_fraction = min(max(self._safe_float(scan.get("release_fraction"), 1.0), 0.05), 1.0)
        quality = (
            scan.get("position_quality") if isinstance(scan.get("position_quality"), dict) else {}
        )
        reason = self._reason(scan, quality)
        raw_response = {
            "analysis_type": "position_review",
            "forced_exit": True,
            "position_release_policy": {
                "forced": True,
                "source": "position_quality_capacity_release",
                "exit_score": round(exit_score, 4),
                "release_fraction": round(release_fraction, 6),
                "release_reason": scan.get("release_reason") or scan.get("reason") or "",
                "scan_reason": scan.get("reason") or "",
            },
            "position_quality": quality,
            "close_evidence": {
                "forced_exit": True,
                "hard_risk": False,
                "source": "low_quality_position_release",
                "reason": reason,
            },
        }
        return DecisionOutput(
            model_name=model_name,
            symbol=symbol,
            action=action,
            confidence=min(max(exit_score / 100.0, 0.82), 0.98),
            reasoning=reason,
            position_size_pct=release_fraction,
            suggested_leverage=1.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            raw_response=raw_response,
            feature_snapshot=self._feature_snapshot(feature_vector),
        )

    @staticmethod
    def _release_action(
        scan: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> Action | None:
        text = str(scan.get("release_action") or "").lower().strip()
        if text == "close_long":
            return Action.CLOSE_LONG
        if text == "close_short":
            return Action.CLOSE_SHORT
        sides = {
            str(position.get("side") or "").lower().strip()
            for position in positions or []
            if str(position.get("side") or "").lower().strip() in {"long", "short"}
        }
        if len(sides) != 1:
            return None
        side = next(iter(sides))
        return Action.CLOSE_LONG if side == "long" else Action.CLOSE_SHORT

    @staticmethod
    def _reason(scan: dict[str, Any], quality: dict[str, Any]) -> str:
        reason = str(scan.get("release_reason") or scan.get("reason") or "低质量持仓释放").strip()
        bucket = str(quality.get("bucket") or "").strip()
        score = quality.get("score")
        hold_hours = quality.get("hold_hours")
        details = []
        if bucket:
            details.append(f"质量分层={bucket}")
        if score is not None:
            details.append(f"质量分={score}")
        if hold_hours is not None:
            details.append(f"持仓小时={hold_hours}")
        suffix = "；" + "，".join(details) if details else ""
        return f"策略纪律触发低质量持仓释放：{reason}{suffix}。"

    def _fresh_low_quality_scan_is_protected(self, scan: dict[str, Any]) -> bool:
        quality = (
            scan.get("position_quality") if isinstance(scan.get("position_quality"), dict) else {}
        )
        hold_hours = self._safe_float(quality.get("hold_hours"), 999.0)
        pnl_ratio = self._safe_float(quality.get("pnl_ratio"), 0.0)
        if hold_hours >= self.params.fresh_position_min_release_hold_hours:
            return False
        if pnl_ratio <= self.params.fresh_position_hard_risk_loss_ratio:
            return False
        reasons = {str(item) for item in quality.get("reasons", []) if item is not None}
        return bool(
            reasons
            & {
                "fresh_position_observation",
                "hard_loss_pressure",
                "loss_pressure",
                "signal_reversal",
            }
        )

    @staticmethod
    def _feature_snapshot(feature_vector: Any | None) -> dict[str, Any]:
        if feature_vector is None:
            return {}
        if isinstance(feature_vector, dict):
            return dict(feature_vector)
        to_dict = getattr(feature_vector, "to_dict", None)
        if callable(to_dict):
            snapshot = to_dict()
            return snapshot if isinstance(snapshot, dict) else {}
        return {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
