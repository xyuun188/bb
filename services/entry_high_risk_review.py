"""Fail-closed high-risk entry review boundary."""

from __future__ import annotations

import hashlib
import ipaddress
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
from services.entry_direction_metrics import selected_entry_metrics
from services.normal_paper_trade import (
    NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION,
    NORMAL_PAPER_TRADE_VERSION,
)
from services.trading_policies import PolicyGateResult

_DEFAULT_DISAGREEMENT_THRESHOLD = 1 / 3
_DEFAULT_TAIL_RISK_THRESHOLD = 0.65
_DEFAULT_LEVERAGE_THRESHOLD = 8.0
_DEFAULT_POSITION_SIZE_THRESHOLD = 0.12
_DEFAULT_MIN_APPROVAL_CONFIDENCE = 0.5
_PAPER_ADVISORY_RISK_FRACTION_CAP = 0.0002
_PAPER_ADVISORY_LEVERAGE_CAP = 1.0
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_RETIRED_REVIEWER_MARKERS = ("deepseek-r1-14b-risk", "14b", "finquant-expert")


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


def validate_cloud_reviewer_route(
    api_base: str,
    model: str,
    revision: str,
    api_key: str,
) -> tuple[bool, str]:
    """Validate the only route allowed for hard high-risk entry reviews.

    The reviewer is an independent cloud service. A missing identity, local
    endpoint, or retired local model must never silently become an approval
    path. This check is deliberately shared by runtime and health reporting.
    """

    base = str(api_base or "").strip().rstrip("/")
    model_id = str(model or "").strip()
    model_revision = str(revision or "").strip()
    key = str(api_key or "").strip()
    # Most OpenAI-compatible cloud gateways expose a stable model ID but do
    # not publish a provider revision.  Requiring an invented revision blocks
    # otherwise verifiable routes and gives the operator no value.  When a
    # provider does publish one we retain and validate it as extra identity.
    if not base or not model_id or not key:
        return False, "cloud_reviewer_identity_incomplete"
    lowered_model = model_id.lower()
    if any(marker in lowered_model for marker in _RETIRED_REVIEWER_MARKERS):
        return False, "retired_local_reviewer_model"
    identity_values = [model_id.lower()]
    if model_revision:
        identity_values.append(model_revision.lower())
    if any(value in {"unknown", "unverified", "pending"} for value in identity_values):
        return False, "cloud_reviewer_identity_placeholder"
    try:
        parsed = urlsplit(base)
        host = str(parsed.hostname or "").lower()
        if parsed.scheme != "https" or not host or parsed.username or parsed.password:
            return False, "cloud_reviewer_requires_https_without_credentials"
        if host in _LOOPBACK_HOSTS:
            return False, "cloud_reviewer_must_not_use_loopback"
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and (address.is_loopback or address.is_private or address.is_link_local):
            return False, "cloud_reviewer_must_use_public_endpoint"
    except ValueError:
        return False, "cloud_reviewer_invalid_endpoint"
    return True, ""


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

    @staticmethod
    def _paper_advisory_contract(
        decision: DecisionOutput,
        model_mode: str,
    ) -> dict[str, Any]:
        raw = _safe_dict(decision.raw_response)
        normal_trade = _safe_dict(raw.get("normal_paper_trade"))
        sizing = _safe_dict(raw.get("profit_risk_sizing"))
        selected_metrics = selected_entry_metrics(decision, model_mode)
        normal_risk_cap = _safe_float(
            normal_trade.get("single_trade_risk_fraction_cap"), -1.0
        )
        sizing_risk_cap = _safe_float(
            sizing.get("single_trade_risk_fraction_cap"), -1.0
        )
        final_leverage = _safe_float(sizing.get("final_leverage"), -1.0)
        violations: list[str] = []
        if str(model_mode or "").lower() != "paper":
            violations.append("execution_mode_not_paper")
        if not (
            selected_metrics.source == "normal_paper_trade_contract"
            and selected_metrics.quality_observation
        ):
            violations.append("paper_quality_observation_contract_invalid")
        if normal_trade.get("execution_scope") != "paper_only":
            violations.append("normal_trade_scope_invalid")
        if normal_trade.get("production_permission") is not False:
            violations.append("normal_trade_production_permission_invalid")
        if normal_trade.get("paper_quality_observation_only") is not True:
            violations.append("paper_quality_observation_contract_missing")
        if sizing.get("execution_scope") != "paper_only":
            violations.append("sizing_scope_invalid")
        if sizing.get("production_permission") is not False:
            violations.append("sizing_production_permission_invalid")
        if sizing.get("production_eligible") is not True:
            violations.append("sizing_not_eligible")
        if sizing.get("paper_quality_observation_mode") is not True:
            violations.append("paper_quality_observation_sizing_missing")
        current_normal_paper = normal_trade.get("version") == NORMAL_PAPER_TRADE_VERSION
        advisory_risk_cap = (
            NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION
            if current_normal_paper
            else _PAPER_ADVISORY_RISK_FRACTION_CAP
        )
        if not 0.0 < normal_risk_cap <= advisory_risk_cap:
            violations.append("normal_trade_risk_cap_exceeded")
        if not 0.0 < sizing_risk_cap <= advisory_risk_cap:
            violations.append("sizing_risk_cap_exceeded")
        decision_leverage = _safe_float(decision.suggested_leverage, -1.0)
        if current_normal_paper:
            if final_leverage < 1.0 or abs(final_leverage - round(final_leverage)) > 1e-8:
                violations.append("final_leverage_invalid")
            if decision_leverage < 1.0 or abs(decision_leverage - final_leverage) > 1e-8:
                violations.append("decision_leverage_invalid")
        else:
            if not 0.0 < final_leverage <= _PAPER_ADVISORY_LEVERAGE_CAP:
                violations.append("final_leverage_exceeded")
            if not 0.0 < decision_leverage <= _PAPER_ADVISORY_LEVERAGE_CAP:
                violations.append("decision_leverage_exceeded")
        return {
            "eligible": not violations,
            "execution_scope": "paper_only",
            "production_permission": False,
            "paper_quality_observation_mode": sizing.get(
                "paper_quality_observation_mode"
            )
            is True,
            "single_trade_risk_fraction_cap": sizing_risk_cap,
            "final_leverage": final_leverage,
            "risk_fraction_limit": advisory_risk_cap,
            "dynamic_leverage_allowed": current_normal_paper,
            "violations": violations,
        }

    def _allow_paper_advisory_only(
        self,
        decision: DecisionOutput,
        model_mode: str,
        base_review: dict[str, Any],
        *,
        status: str,
        unavailable_reason: str,
        extra: dict[str, Any] | None = None,
    ) -> PolicyGateResult | None:
        contract = self._paper_advisory_contract(decision, model_mode)
        if not contract["eligible"]:
            return None
        payload = {
            **base_review,
            **(extra or {}),
            "status": status,
            "approved": None,
            "hard_review_required": False,
            "advisory_review_required": True,
            "review_required_for_live": True,
            "approval_semantics": "advisory_unavailable_is_not_cloud_approval",
            "unavailable_reason": unavailable_reason,
            "paper_advisory_contract": contract,
        }
        self._annotate(decision, payload)
        return PolicyGateResult.allow({"high_risk_review": payload})

    def trigger_reasons(
        self,
        decision: DecisionOutput,
        model_mode: str = "",
    ) -> list[str]:
        """Return deterministic reasons that require a second risk opinion."""
        if not decision.is_entry:
            return []
        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        metrics = selected_entry_metrics(decision, model_mode)
        reasons: list[str] = []
        if entry_expert_disagreement(decision) >= self._threshold(
            "high_risk_review_disagreement_threshold", _DEFAULT_DISAGREEMENT_THRESHOLD
        ):
            reasons.append("expert_disagreement")
        if ml_ai_direction_conflict(decision):
            reasons.append("ml_ai_direction_conflict")
        tail_risk = max(
            metrics.tail_risk_score,
            _safe_float(raw.get("tail_risk_score")),
        )
        if tail_risk >= self._threshold(
            "high_risk_review_tail_risk_threshold", _DEFAULT_TAIL_RISK_THRESHOLD
        ):
            reasons.append("tail_risk")
        if (
            metrics.expected_net_return_available
            and metrics.expected_net_return_pct <= 0
        ):
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
        metrics = selected_entry_metrics(decision, model_mode)
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
                "side": metrics.side,
                "source": metrics.source,
                "expected_net_return_pct": (
                    metrics.expected_net_return_pct
                    if metrics.expected_net_return_available
                    else None
                ),
                "objective_net_return_pct": metrics.objective_net_return_pct,
                "profit_quality_ratio": metrics.profit_quality_ratio,
                "loss_probability": (
                    metrics.loss_probability
                    if metrics.loss_probability_available
                    else None
                ),
                "tail_risk_score": metrics.tail_risk_score,
                "reward_risk_ratio": opportunity.get("reward_risk_ratio"),
                "paper_quality_observation": metrics.quality_observation,
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
        reasons = self.trigger_reasons(decision, model_mode)
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
        revision = str(getattr(self.config, "high_risk_review_model_revision", "") or "").strip()
        base_review["provider"] = _review_provider(api_base)
        if not bool(getattr(self.config, "high_risk_review_enabled", False)):
            advisory = self._allow_paper_advisory_only(
                decision,
                model_mode,
                base_review,
                status="skipped_advisory_only",
                unavailable_reason="reviewer_config_disabled",
            )
            if advisory is not None:
                return advisory
            payload = {**base_review, "status": "config_disabled", "approved": False}
            self._annotate(decision, payload)
            return PolicyGateResult.block(
                "high_risk_review_config_missing",
                "高风险开仓复核未启用，系统按失败关闭处理，未提交订单。",
                {"high_risk_review": payload},
            )
        route_valid, route_error = validate_cloud_reviewer_route(
            api_base,
            model,
            revision,
            api_key,
        )
        if self.reviewer is None or not route_valid:
            advisory = self._allow_paper_advisory_only(
                decision,
                model_mode,
                base_review,
                status="skipped_advisory_only",
                unavailable_reason=route_error or "reviewer_service_missing",
            )
            if advisory is not None:
                return advisory
            payload = {
                **base_review,
                "status": "config_missing",
                "approved": False,
                "error_code": "reviewer_configuration_missing",
                "route_error": route_error or "reviewer_service_missing",
            }
            self._annotate(decision, payload)
            return PolicyGateResult.block(
                "high_risk_review_config_missing",
                "高风险开仓复核配置不完整，系统按失败关闭处理，未提交订单。",
                {"high_risk_review": payload},
            )

        circuit_payload = getattr(self.reviewer, "circuit_payload", lambda: None)()
        if circuit_payload:
            advisory = self._allow_paper_advisory_only(
                decision,
                model_mode,
                base_review,
                status="error_advisory_only",
                unavailable_reason="reviewer_circuit_open",
                extra={**circuit_payload, "required": False},
            )
            if advisory is not None:
                return advisory
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
            advisory = self._allow_paper_advisory_only(
                decision,
                model_mode,
                base_review,
                status="error_advisory_only",
                unavailable_reason="reviewer_call_failed",
                extra=payload,
            )
            if advisory is not None:
                return advisory
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
        advisory = self._allow_paper_advisory_only(
            decision,
            model_mode,
            base_review,
            status="rejected_advisory_only",
            unavailable_reason=(
                "reviewer_rejection_is_advisory_for_bounded_paper_observation"
            ),
            extra={
                **payload,
                "reviewer_approved": approved,
                "reviewer_confidence": round(confidence, 6),
            },
        )
        if advisory is not None:
            return advisory
        if approved:
            payload["status"] = "invalid_response_blocked"
            payload["error_code"] = "approval_confidence_below_floor"
        return PolicyGateResult.block(
            "high_risk_review_rejected" if not approved else "high_risk_review_invalid",
            str(payload.get("reason") or "高风险复核未批准当前开仓，系统未提交订单。"),
            {"high_risk_review": payload},
        )
