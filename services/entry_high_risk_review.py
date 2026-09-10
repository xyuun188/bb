"""Fail-closed high-risk entry review boundary."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from core.safe_output import safe_error_text
from services.trading_policies import PolicyGateResult

_DEFAULT_DISAGREEMENT_THRESHOLD = 1 / 3
_DEFAULT_TAIL_RISK_THRESHOLD = 0.65
_DEFAULT_LEVERAGE_THRESHOLD = 8.0
_DEFAULT_POSITION_SIZE_THRESHOLD = 0.12
_DEFAULT_MIN_APPROVAL_CONFIDENCE = 0.5


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _review_provider(api_base: str) -> str:
    parsed = urlsplit(api_base)
    return parsed.netloc or parsed.path.split("/", 1)[0] or "configured_reviewer"


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def entry_side_value(decision: DecisionOutput) -> str:
    if decision.action == Action.LONG:
        return "long"
    if decision.action == Action.SHORT:
        return "short"
    return "hold"


def entry_expert_disagreement(decision: DecisionOutput) -> float:
    opinions = _safe_list(_safe_dict(decision.raw_response).get("opinions"))
    side = entry_side_value(decision)
    directional = [
        _safe_dict(item)
        for item in opinions
        if str(_safe_dict(item).get("action") or "").lower() in {"long", "short"}
    ]
    opposite = "short" if side == "long" else "long"
    opposite_count = sum(
        str(item.get("action") or "").lower() == opposite for item in directional
    )
    return opposite_count / len(directional) if directional else 0.0


def ml_ai_direction_conflict(decision: DecisionOutput) -> bool:
    raw = _safe_dict(decision.raw_response)
    ml_signal = _safe_dict(raw.get("ml_signal"))
    predictions = _safe_list(ml_signal.get("predictions"))
    primary = _safe_dict(predictions[0]) if predictions else {}
    ml_side = str(primary.get("best_side") or "").lower()
    return bool(ml_side in {"long", "short"} and ml_side != entry_side_value(decision))


@dataclass(slots=True)
class EntryHighRiskReviewGatePolicy:
    """Gate only risky new entries; exit actions always bypass this service."""

    reviewer: Any | None = None
    allocation_state_provider: Callable[[str], Awaitable[dict[str, Any]]] | None = None
    config: Any = field(default_factory=lambda: settings)

    def _threshold(self, name: str, default: float) -> float:
        return _safe_float(getattr(self.config, name, default), default)

    def trigger_reasons(self, decision: DecisionOutput) -> list[str]:
        """Return deterministic reasons that require a second risk opinion."""
        if not decision.is_entry:
            return []
        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        reasons: list[str] = []
        if entry_expert_disagreement(decision) >= self._threshold(
            "high_risk_review_disagreement_threshold", _DEFAULT_DISAGREEMENT_THRESHOLD
        ):
            reasons.append("expert_disagreement")
        if ml_ai_direction_conflict(decision):
            reasons.append("ml_ai_direction_conflict")
        tail_risk = max(
            _safe_float(opportunity.get("tail_risk_score")),
            _safe_float(raw.get("tail_risk_score")),
        )
        if tail_risk >= self._threshold(
            "high_risk_review_tail_risk_threshold", _DEFAULT_TAIL_RISK_THRESHOLD
        ):
            reasons.append("tail_risk")
        expected_net = opportunity.get("expected_net_return_pct")
        if expected_net is not None and _safe_float(expected_net, 0.0) <= 0:
            reasons.append("non_positive_expected_net")
        sizing = _safe_dict(raw.get("profit_risk_sizing"))
        leverage = max(
            _safe_float(decision.suggested_leverage),
            _safe_float(sizing.get("final_leverage")),
        )
        if leverage >= self._threshold(
            "high_risk_review_leverage_threshold", _DEFAULT_LEVERAGE_THRESHOLD
        ):
            reasons.append("high_leverage")
        position_size = max(
            _safe_float(decision.position_size_pct),
            _safe_float(sizing.get("final_position_size_pct")),
        )
        if position_size >= self._threshold(
            "high_risk_review_position_size_threshold", _DEFAULT_POSITION_SIZE_THRESHOLD
        ):
            reasons.append("large_position")
        for key in (
            "abnormal_market_state",
            "liquidity_risk",
            "price_anomaly",
            "protection_order_risk",
        ):
            if bool(raw.get(key)) or bool(opportunity.get(key)):
                reasons.append(key)
        return list(dict.fromkeys(reasons))

    def _prompt(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]],
        reasons: list[str],
        allocation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        sizing = _safe_dict(raw.get("profit_risk_sizing"))
        return {
            "symbol": decision.symbol,
            "side": entry_side_value(decision),
            "mode": model_mode,
            "model_name": decision.model_name,
            "confidence": round(_safe_float(decision.confidence), 6),
            "position_size_pct": round(_safe_float(decision.position_size_pct), 8),
            "suggested_leverage": round(_safe_float(decision.suggested_leverage), 4),
            "stop_loss_pct": round(_safe_float(decision.stop_loss_pct), 8),
            "take_profit_pct": round(_safe_float(decision.take_profit_pct), 8),
            "risk_triggers": reasons,
            "expert_disagreement": round(entry_expert_disagreement(decision), 8),
            "ml_ai_direction_conflict": ml_ai_direction_conflict(decision),
            "opportunity_score": {
                "score": opportunity.get("score"),
                "expected_net_return_pct": opportunity.get("expected_net_return_pct"),
                "tail_risk_score": opportunity.get("tail_risk_score"),
                "reward_risk_ratio": opportunity.get("reward_risk_ratio"),
            },
            "sizing": {
                "final_notional_usdt": sizing.get("final_notional_usdt"),
                "final_leverage": sizing.get("final_leverage"),
                "production_eligible": sizing.get("production_eligible"),
            },
            "open_position_count": len(open_positions),
            "allocation": allocation or {},
        }

    @staticmethod
    def _annotate(decision: DecisionOutput, payload: dict[str, Any]) -> None:
        raw = _safe_dict(decision.raw_response)
        raw["high_risk_review"] = payload
        decision.raw_response = raw

    async def evaluate(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None,
    ) -> PolicyGateResult | None:
        # Keep this explicit guard so direct callers can never make exits depend
        # on a remote reviewer.
        if not decision.is_entry:
            return None
        positions = open_positions or []
        reasons = self.trigger_reasons(decision)
        base_review = {
            "read_only": False,
            "production_permission": False,
            "configured": bool(self.config.high_risk_review_enabled),
            "triggered": bool(reasons),
            "hard_review_required": bool(reasons),
            "reasons": reasons,
            "expert_disagreement": round(entry_expert_disagreement(decision), 8),
            "ml_ai_direction_conflict": ml_ai_direction_conflict(decision),
            "provider": "",
            "model": str(getattr(self.config, "high_risk_review_model", "") or ""),
            "revision": str(getattr(self.config, "high_risk_review_model_revision", "") or ""),
        }
        if not reasons:
            payload = {
                **base_review,
                "status": "not_required",
                "approved": None,
                "approval_semantics": "not_required_is_not_reviewer_approval",
            }
            self._annotate(decision, payload)
            return PolicyGateResult.allow({"high_risk_review": payload})

        api_base = str(getattr(self.config, "high_risk_review_api_base", "") or "").strip()
        api_key = str(getattr(self.config, "high_risk_review_api_key", "") or "").strip()
        model = str(getattr(self.config, "high_risk_review_model", "") or "").strip()
        base_review["provider"] = _review_provider(api_base)
        if not bool(getattr(self.config, "high_risk_review_enabled", False)):
            payload = {**base_review, "status": "config_disabled", "approved": False}
            self._annotate(decision, payload)
            return PolicyGateResult.block(
                "high_risk_review_config_missing",
                "高风险开仓复核未启用，系统按失败关闭处理，未提交订单。",
                {"high_risk_review": payload},
            )
        if self.reviewer is None or not api_base or not api_key or not model:
            payload = {
                **base_review,
                "status": "config_missing",
                "approved": False,
                "error_code": "reviewer_configuration_missing",
            }
            self._annotate(decision, payload)
            return PolicyGateResult.block(
                "high_risk_review_config_missing",
                "高风险开仓复核配置不完整，系统按失败关闭处理，未提交订单。",
                {"high_risk_review": payload},
            )

        circuit_payload = getattr(self.reviewer, "circuit_payload", lambda: None)()
        if circuit_payload:
            payload = {
                **base_review,
                **circuit_payload,
                "status": "circuit_open",
                "required": True,
            }
            self._annotate(decision, payload)
            return PolicyGateResult.block(
                "high_risk_review_circuit_open",
                str(payload.get("reason") or "高风险复核熔断中，未提交开仓订单。"),
                {"high_risk_review": payload},
            )

        allocation: dict[str, Any] | None = None
        allocation_error = ""
        if self.allocation_state_provider is not None:
            try:
                allocation = await self.allocation_state_provider(model_mode)
            except Exception as exc:  # allocation is evidence, never a reason to fail open
                allocation_error = safe_error_text(exc, limit=180)
        prompt = self._prompt(decision, model_mode, positions, reasons, allocation)
        started = time.perf_counter()
        fingerprint = _fingerprint(prompt)
        try:
            result = await self.reviewer.review_trade(
                prompt,
                api_base=api_base,
                api_key=api_key,
                model=model,
            )
        except BaseException as exc:
            if exc.__class__.__name__ == "CancelledError":
                payload = {
                    **base_review,
                    "status": "cancelled",
                    "approved": False,
                    "input_fingerprint": fingerprint,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
                self._annotate(decision, payload)
                raise
            error_text = safe_error_text(exc, limit=220)
            record_failure = getattr(self.reviewer, "record_failure", None)
            if callable(record_failure):
                record_failure(error_text)
            payload = {
                **base_review,
                "status": "error_blocked",
                "approved": False,
                "error_code": "reviewer_call_failed",
                "error_category": str(getattr(exc, "category", "reviewer_call_failed") or "reviewer_call_failed"),
                "provider_status_code": getattr(exc, "status_code", None),
                "attempts": getattr(exc, "attempts", []),
                "retry_count": max(len(getattr(exc, "attempts", [])) - 1, 0),
                "error": error_text,
                "input_fingerprint": fingerprint,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "allocation_error": allocation_error or None,
            }
            self._annotate(decision, payload)
            return PolicyGateResult.block(
                "high_risk_review_failed",
                f"高风险开仓复核失败：{error_text}。系统未提交订单。",
                {"high_risk_review": payload},
            )

        approved = bool(getattr(result, "approved", False))
        confidence = _safe_float(getattr(result, "confidence", 0.0))
        payload = {
            **base_review,
            "status": "approved" if approved else "rejected",
            "approved": approved,
            "confidence": round(confidence, 6),
            "reason": str(getattr(result, "reason", "") or "")[:500],
            "attempts": getattr(result, "attempts", []),
            "retry_count": max(len(getattr(result, "attempts", [])) - 1, 0),
            "input_fingerprint": fingerprint,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "allocation_error": allocation_error or None,
        }
        self._annotate(decision, payload)
        if approved and confidence >= self._threshold(
            "high_risk_review_min_confidence", _DEFAULT_MIN_APPROVAL_CONFIDENCE
        ):
            return PolicyGateResult.allow({"high_risk_review": payload})
        if approved:
            payload["status"] = "invalid_response_blocked"
            payload["error_code"] = "approval_confidence_below_floor"
        return PolicyGateResult.block(
            "high_risk_review_rejected" if not approved else "high_risk_review_invalid",
            str(payload.get("reason") or "高风险复核未批准当前开仓，系统未提交订单。"),
            {"high_risk_review": payload},
        )
