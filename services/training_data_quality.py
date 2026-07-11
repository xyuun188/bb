from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from services.profit_first_trade_plan import normalize_losing_exit_attribution
from services.text_integrity import looks_like_mojibake
from services.trading_params import DEFAULT_TRADING_PARAMS

_QUALITY_PARAMS = DEFAULT_TRADING_PARAMS.training_data_quality
DATA_QUALITY_VERSION = "2026-07-03.v4"
PROFIT_LEARNING_VERSION = "profit-first-training-v1"
PHASE3_TRAINING_POLICY = "clean_training_view_only"
HIGH_CONTAMINATION_EXCLUDED_RATIO = 0.05
HIGH_CONTAMINATION_BLOCKED_REASON_RATIO = 0.02
MEDIUM_CONTAMINATION_EXCLUDED_RATIO = 0.005
MIN_PROMOTION_SHADOW_SAMPLES = 30
MIN_DIRECTION_HIT_RATE = 0.48
MIN_AVG_REALIZED_RETURN_PCT = 0.02
MAX_FALSE_SIGNAL_LOSS_PCT = -0.18
MIN_TIMESERIES_SEQUENCE_LENGTH = 30
MAX_WORST_SAMPLE_COUNT = 8
_SHADOW_BENIGN_DOWNWEIGHT_REASONS = {
    "hold_missed_opportunity_downweighted",
    "very_low_decision_confidence",
}
QualityStatus = Literal["included", "downweighted", "excluded"]
SampleKind = Literal["shadow", "trade", "sequence", "text_sentiment"]
_RETRAIN_TARGETS = (
    "local_ml_signal",
    "local_ai_tools",
    "vector_memory_reindex",
)
_TRADE_REPAIR_SOURCE_MARKERS = ("repair", "correction", "backfill")
_TRADE_REPAIR_SOURCES = {
    "missing_closed_position_repair",
    "okx_native_full_close_fill_correction",
    "okx_order_pair_repair",
    "okx_orphan_position_quarantine",
    "okx_position_link_repair",
}
_TRUSTED_TRADE_PNL_SOURCES = {
    "okx_position_history_realized_pnl",
    "okx_linked_order_net_pnl",
    "okx_close_fill_net_pnl_partial",
    "okx_fill_pnl",
    "okx_order_fact_sync",
    "okx_position_history_settlement",
    "okx_authoritative_reconcile",
    "position_settlement_snapshot",
    "position_settlement_snapshot:okx_order_fact_sync",
    "position_settlement_snapshot:okx_position_history_settlement",
    "position_settlement_snapshot:okx_authoritative_reconcile",
}
_UNTRUSTED_TRADE_PNL_SOURCES = {
    "",
    "local_db",
    "position_realized_pnl",
    "reported",
    "derived_from_prices",
    "manual",
}
_TRUSTED_FUNDING_FEE_SOURCES = {
    "okx_account_bills",
    "okx_positions_history.fundingFee",
    "okx_positions_history.funding_fee",
    "position_settlement_snapshot",
}


@dataclass(frozen=True)
class SampleQualityAssessment:
    status: QualityStatus
    score: float
    weight: float
    reasons: tuple[str, ...]
    version: str = DATA_QUALITY_VERSION

    @property
    def exclude_from_training(self) -> bool:
        return self.status == "excluded"

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_quality_version": self.version,
            "data_quality_status": self.status,
            "data_quality_score": round(self.score, 4),
            "sample_weight": round(self.weight, 4),
            "exclude_from_training": self.exclude_from_training,
            "exclude_from_training_reason": (
                ";".join(self.reasons) if self.exclude_from_training else ""
            ),
            "quality_reasons": list(self.reasons),
        }


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_float(*values: Any) -> float | None:
    for value in values:
        number = _safe_float(value, None)
        if number is not None:
            return number
    return None


def _features(sample: dict[str, Any]) -> dict[str, Any]:
    value = sample.get("features")
    return value if isinstance(value, dict) else {}


def _iter_text_values(value: Any) -> Any:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_text_values(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_text_values(item)


def _has_mojibake_text(*values: Any) -> bool:
    return any(looks_like_mojibake(text) for value in values for text in _iter_text_values(value))


def _sample_guard_reasons(sample: dict[str, Any], *text_values: Any) -> list[str]:
    reasons: list[str] = []
    if _has_mojibake_text(sample, *text_values):
        reasons.append("mojibake_text")
    if _is_duplicate_sample(sample):
        reasons.append("duplicate_sample")
    if _has_future_leakage(sample, *text_values):
        reasons.append("future_leakage")
    return reasons


def _is_benign_downweighted_sample(kind: str, sample: dict[str, Any]) -> bool:
    if kind != "shadow":
        return False
    if _safe_str(sample.get("data_quality_status")) != "downweighted":
        return False
    reasons = {
        _safe_str(reason)
        for reason in sample.get("quality_reasons") or []
        if _safe_str(reason)
    }
    return bool(
        "hold_missed_opportunity_downweighted" in reasons
        and reasons.issubset(_SHADOW_BENIGN_DOWNWEIGHT_REASONS)
    )


def _repair_provenance_reason(sample: dict[str, Any]) -> str:
    repair_source = _safe_str(sample.get("trade_fact_repair_source")).lower()
    reflection_source = _safe_str(sample.get("reflection_source")).lower()
    for candidate in (repair_source, reflection_source):
        if candidate in _TRADE_REPAIR_SOURCES:
            return candidate
    for candidate in (repair_source, reflection_source):
        if candidate and any(token in candidate for token in _TRADE_REPAIR_SOURCE_MARKERS):
            return candidate
    return ""


def _trade_pnl_source(sample: dict[str, Any]) -> str:
    for key in ("pnl_source", "settlement_source", "realized_pnl_source"):
        value = _safe_str(sample.get(key))
        if value:
            return value
    close_raw = _safe_dict(sample.get("close_raw"))
    for key in ("pnl_source", "settlement_source", "realized_pnl_source"):
        value = _safe_str(close_raw.get(key))
        if value:
            return value
    return ""


def _trade_pnl_source_trusted(source: str) -> bool:
    normalized = _safe_str(source)
    if normalized in _UNTRUSTED_TRADE_PNL_SOURCES:
        return False
    if normalized in _TRUSTED_TRADE_PNL_SOURCES:
        return True
    return bool(normalized.startswith("position_settlement_snapshot:okx_"))


def _trade_funding_fee_source(sample: dict[str, Any]) -> str:
    for key in ("funding_fee_source", "funding_source"):
        value = _safe_str(sample.get(key))
        if value:
            return value
    close_raw = _safe_dict(sample.get("close_raw"))
    for key in ("funding_fee_source", "funding_source"):
        value = _safe_str(close_raw.get(key))
        if value:
            return value
    return ""


def _trade_funding_source_trusted(sample: dict[str, Any]) -> bool:
    funding_fee = _trade_funding_fee(sample)
    source = _trade_funding_fee_source(sample)
    if funding_fee is None:
        return False
    return bool(source in _TRUSTED_FUNDING_FEE_SOURCES or source.startswith("okx_"))


def _is_duplicate_sample(sample: dict[str, Any]) -> bool:
    duplicate_count = _safe_float(sample.get("duplicate_count"), 0.0) or 0.0
    return bool(sample.get("is_duplicate") or sample.get("duplicate_of") or duplicate_count > 1)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iter_dict_values(value: Any) -> Any:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_dict_values(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_dict_values(item)


def _first_timestamp(values: tuple[Any, ...], keys: tuple[str, ...]) -> datetime | None:
    for value in values:
        for container in _iter_dict_values(value):
            for key in keys:
                parsed = _parse_datetime(container.get(key))
                if parsed is not None:
                    return parsed
    return None


def _has_future_leakage(sample: dict[str, Any], *containers: Any) -> bool:
    values = (sample, *containers)
    feature_at = _first_timestamp(
        values,
        ("feature_timestamp", "feature_at", "observed_at"),
    )
    label_at = _first_timestamp(
        values,
        ("label_timestamp", "label_at", "outcome_at"),
    )
    return bool(feature_at and label_at and feature_at > label_at)


def _market_data_quality_issue(sample: dict[str, Any], features: dict[str, Any]) -> str:
    for container in (sample, features):
        if not isinstance(container, dict):
            continue
        issue = container.get("market_data_quality")
        if isinstance(issue, dict):
            code = _safe_str(issue.get("code"))
            if code:
                return code
        reason = _safe_str(container.get("training_quality_reason"))
        if reason.startswith("market_data_quality:"):
            return reason.split(":", 1)[1]
    return ""


def _final_assessment(
    score: float, reasons: list[str], *, exclude: bool = False
) -> SampleQualityAssessment:
    if exclude:
        status: QualityStatus = "excluded"
        weight = 0.0
        score = min(score, _QUALITY_PARAMS.excluded_score_cap)
    elif score < _QUALITY_PARAMS.include_score_threshold or reasons:
        status = "downweighted"
        weight = max(
            _QUALITY_PARAMS.downweighted_min_weight,
            min(score, _QUALITY_PARAMS.downweighted_max_weight),
        )
    else:
        status = "included"
        weight = 1.0
    return SampleQualityAssessment(
        status=status, score=max(0.0, min(score, 1.0)), weight=weight, reasons=tuple(reasons)
    )


def assess_shadow_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    features = _features(sample)
    reasons: list[str] = []
    score = 1.0
    exclude = False

    guard_reasons = _sample_guard_reasons(sample, features)
    if guard_reasons:
        return _final_assessment(0.0, guard_reasons, exclude=True)

    quality_issue = _market_data_quality_issue(sample, features)
    if quality_issue:
        return _final_assessment(
            0.0,
            [f"market_data_quality:{quality_issue}"],
            exclude=True,
        )

    if not features:
        return _final_assessment(0.0, ["missing_features"], exclude=True)

    price_warning = _safe_str(features.get("price_reconciliation_warning"))
    indicator_gap = abs(_safe_float(features.get("indicator_price_gap_pct"), 0.0) or 0.0)
    if price_warning or indicator_gap >= _QUALITY_PARAMS.abnormal_indicator_price_gap_pct:
        return _final_assessment(
            0.0,
            ["price_reconciliation:" f"{price_warning or 'indicator_close_diverged'}"],
            exclude=True,
        )

    if bool(features.get("stale")) or bool(features.get("ticker_stale")):
        return _final_assessment(0.0, ["stale_ticker_snapshot"], exclude=True)

    current_price = _safe_float(features.get("current_price") or features.get("close"), None)
    low_24h = _safe_float(features.get("low_24h"), None)
    high_24h = _safe_float(features.get("high_24h"), None)
    if current_price is not None and low_24h is not None and high_24h is not None:
        if current_price > 0 and high_24h > 0 and low_24h > 0 and high_24h >= low_24h:
            tolerance = _QUALITY_PARAMS.training_price_24h_range_tolerance_pct
            lower_bound = low_24h * (1.0 - tolerance)
            upper_bound = high_24h * (1.0 + tolerance)
            if current_price < lower_bound or current_price > upper_bound:
                return _final_assessment(
                    0.0,
                    ["price_outside_24h_range"],
                    exclude=True,
                )

    long_return = _safe_float(sample.get("long_return_pct"), None)
    short_return = _safe_float(sample.get("short_return_pct"), None)
    if long_return is None or short_return is None:
        return _final_assessment(0.0, ["missing_outcome_returns"], exclude=True)
    if (
        abs(long_return) > _QUALITY_PARAMS.abnormal_shadow_return_abs_pct
        or abs(short_return) > _QUALITY_PARAMS.abnormal_shadow_return_abs_pct
    ):
        return _final_assessment(0.05, ["abnormal_outcome_return"], exclude=True)

    action = _safe_str(sample.get("decision_action")).lower()
    best_action = _safe_str(sample.get("best_action")).lower()
    if action == "hold":
        if bool(sample.get("missed_opportunity")) and best_action in {"long", "short"}:
            score -= _QUALITY_PARAMS.hold_missed_opportunity_penalty
            reasons.append("hold_missed_opportunity_downweighted")
        else:
            score -= _QUALITY_PARAMS.hold_observation_penalty
            reasons.append("hold_observation_downweighted")

    confidence = _safe_float(sample.get("decision_confidence"), 0.0) or 0.0
    if confidence < _QUALITY_PARAMS.very_low_confidence_threshold:
        score -= _QUALITY_PARAMS.very_low_confidence_penalty
        reasons.append("very_low_decision_confidence")

    horizon = int(_safe_float(sample.get("horizon_minutes"), 0.0) or 0)
    if horizon <= 0 or horizon > _QUALITY_PARAMS.max_horizon_minutes:
        score -= _QUALITY_PARAMS.invalid_horizon_penalty
        reasons.append("invalid_horizon_minutes")

    if current_price is None or current_price <= 0:
        score -= _QUALITY_PARAMS.invalid_price_penalty
        reasons.append("missing_or_invalid_price_feature")

    spread = abs(_safe_float(features.get("spread_pct"), 0.0) or 0.0)
    if spread > _QUALITY_PARAMS.abnormal_spread_pct:
        exclude = True
        reasons.append("abnormal_spread_feature")
    elif spread > _QUALITY_PARAMS.wide_spread_pct:
        score -= _QUALITY_PARAMS.wide_spread_penalty
        reasons.append("wide_spread_feature")

    return _final_assessment(score, reasons, exclude=exclude)


def assess_trade_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    reasons: list[str] = []
    score = 1.0
    exclude = False

    guard_reasons = _sample_guard_reasons(sample)
    if guard_reasons:
        return _final_assessment(0.0, guard_reasons, exclude=True)

    source = _safe_str(sample.get("source")).lower()
    model_name = _safe_str(sample.get("model_name")).lower()
    if source in set(_QUALITY_PARAMS.manual_trade_sources) or "manual" in model_name:
        return _final_assessment(0.0, ["manual_or_test_trade"], exclude=True)

    trust_reason = _safe_str(sample.get("trade_fact_trust_reason"))
    if trust_reason:
        return _final_assessment(0.0, [f"untrusted_trade_fact:{trust_reason}"], exclude=True)
    if sample.get("trade_fact_trusted") is False:
        return _final_assessment(0.0, ["untrusted_trade_fact"], exclude=True)

    repair_source = _repair_provenance_reason(sample)
    if repair_source:
        return _final_assessment(
            0.0,
            [f"historical_reconciliation_repair:{repair_source}"],
            exclude=True,
        )

    execution_mode = _safe_str(sample.get("execution_mode")).lower()
    if execution_mode and execution_mode not in {"paper", "live", "sim", "simulation"}:
        return _final_assessment(0.0, ["execution_mode_mismatch"], exclude=True)

    close_status = _safe_str(sample.get("close_status") or sample.get("status")).lower()
    if close_status in {"failed", "rejected", "cancelled", "canceled", "error"}:
        return _final_assessment(0.0, ["failed_close_status"], exclude=True)

    position_size_pct = _safe_float(sample.get("position_size_pct"), None)
    evidence_text = " ".join(
        _safe_str(sample.get(field)).lower() for field in ("evidence_tier", "quality_tier")
    )
    if (
        position_size_pct is not None
        and position_size_pct <= 0.001
        and any(token in evidence_text for token in ("weak", "probe", "degraded"))
    ):
        return _final_assessment(0.0, ["weak_evidence_micro_probe"], exclude=True)

    fee_source = sample.get("fee_estimate")
    if fee_source is None:
        fee_source = sample.get("fee")
    if fee_source is None:
        return _final_assessment(0.0, ["missing_fee_estimate"], exclude=True)
    if source == "closed_position":
        pnl_source = _trade_pnl_source(sample)
        if not _trade_pnl_source_trusted(pnl_source):
            return _final_assessment(
                0.0,
                [f"untrusted_realized_pnl_source:{pnl_source or 'missing'}"],
                exclude=True,
            )
        if not _trade_funding_source_trusted(sample):
            return _final_assessment(
                0.0,
                ["missing_or_untrusted_funding_fee_source"],
                exclude=True,
            )

    side = _safe_str(sample.get("side")).lower()
    if side not in {"long", "short"}:
        return _final_assessment(0.0, ["invalid_trade_side"], exclude=True)

    quantity = _safe_float(sample.get("quantity"), 0.0) or 0.0
    if quantity <= 0:
        return _final_assessment(0.0, ["non_positive_quantity"], exclude=True)

    entry_price = _safe_float(sample.get("entry_price"), 0.0) or 0.0
    exit_price = _safe_float(sample.get("exit_price"), 0.0) or 0.0
    if entry_price <= 0 or exit_price < 0:
        score -= _QUALITY_PARAMS.invalid_trade_price_penalty
        reasons.append("invalid_trade_price")

    hold_minutes = _safe_float(sample.get("hold_minutes"), 0.0) or 0.0
    pnl = _safe_float(sample.get("realized_pnl"), 0.0) or 0.0
    fee = abs(_safe_float(fee_source, 0.0) or 0.0)
    if hold_minutes < _QUALITY_PARAMS.fast_loss_exit_minutes and pnl < 0:
        score -= _QUALITY_PARAMS.fast_loss_exit_penalty
        reasons.append("fast_loss_exit_requires_review")
    elif hold_minutes <= 0:
        score -= _QUALITY_PARAMS.missing_hold_duration_penalty
        reasons.append("missing_hold_duration")

    if fee > 0 and abs(pnl) <= fee * _QUALITY_PARAMS.fee_dominated_multiple:
        score -= _QUALITY_PARAMS.fee_dominated_penalty
        reasons.append("fee_dominated_trade")

    outcome = _safe_str(sample.get("outcome")).lower()
    if outcome not in {"profit", "loss", "flat", "win"}:
        score -= _QUALITY_PARAMS.unknown_outcome_penalty
        reasons.append("missing_or_unknown_outcome")

    if pnl < 0:
        attribution = _trade_losing_exit_attribution(sample)
        if attribution == "okx_slippage_or_execution":
            return _final_assessment(0.0, ["execution_anomaly_trade"], exclude=True)
        if attribution == "unknown_requires_review":
            return _final_assessment(0.0, ["unknown_losing_exit_attribution"], exclude=True)

    return _final_assessment(score, reasons, exclude=exclude)


def assess_sequence_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    reasons: list[str] = []
    score = 1.0
    guard_reasons = _sample_guard_reasons(sample, sample.get("close_sequence"))
    if guard_reasons:
        return _final_assessment(0.0, guard_reasons, exclude=True)

    closes = sample.get("close_sequence") or []
    if not isinstance(closes, list) or len(closes) < _QUALITY_PARAMS.min_sequence_length:
        return _final_assessment(0.0, ["short_price_sequence"], exclude=True)
    numeric_closes = [_safe_float(value, None) for value in closes]
    if any(value is None or value <= 0 for value in numeric_closes):
        return _final_assessment(0.0, ["invalid_price_sequence"], exclude=True)
    future_return = _safe_float(sample.get("future_return_pct"), None)
    if future_return is None:
        return _final_assessment(0.0, ["missing_future_return"], exclude=True)
    if abs(future_return) > _QUALITY_PARAMS.abnormal_future_return_abs_pct:
        return _final_assessment(0.05, ["abnormal_future_return"], exclude=True)
    timeframe = _safe_str(sample.get("timeframe"))
    if timeframe not in set(_QUALITY_PARAMS.allowed_sequence_timeframes):
        score -= _QUALITY_PARAMS.unknown_timeframe_penalty
        reasons.append("unknown_timeframe")
    return _final_assessment(score, reasons)


def assess_text_sentiment_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    text = _safe_str(sample.get("text"))
    guard_reasons = _sample_guard_reasons(sample, text)
    if guard_reasons:
        return _final_assessment(0.0, guard_reasons, exclude=True)
    if len(text) < _QUALITY_PARAMS.min_text_length:
        return _final_assessment(0.0, ["empty_or_too_short_text"], exclude=True)
    score = 1.0
    reasons: list[str] = []
    platform = _safe_str(sample.get("platform")).lower()
    if platform in {"", "unknown"}:
        score -= _QUALITY_PARAMS.unknown_text_source_penalty
        reasons.append("unknown_text_source")
    sentiment = _safe_float(sample.get("sentiment_score"), None)
    if sentiment is None:
        score -= _QUALITY_PARAMS.missing_sentiment_penalty
        reasons.append("missing_sentiment_score")
    return _final_assessment(score, reasons)


ASSESSORS = {
    "shadow": assess_shadow_sample,
    "trade": assess_trade_sample,
    "sequence": assess_sequence_sample,
    "text_sentiment": assess_text_sentiment_sample,
}


def _trade_entry_raw(sample: dict[str, Any]) -> dict[str, Any]:
    for key in ("entry_raw", "raw_llm_response", "raw_response"):
        value = _safe_dict(sample.get(key))
        if value:
            return value
    return {}


def _trade_close_raw(sample: dict[str, Any]) -> dict[str, Any]:
    return _safe_dict(sample.get("close_raw"))


def _trade_shadow(sample: dict[str, Any]) -> dict[str, Any]:
    return _safe_dict(sample.get("shadow"))


def _trade_plan(sample: dict[str, Any]) -> dict[str, Any]:
    return _safe_dict(_trade_entry_raw(sample).get("profit_first_trade_plan"))


def _trade_sizing(sample: dict[str, Any]) -> dict[str, Any]:
    return _safe_dict(_trade_entry_raw(sample).get("profit_risk_sizing"))


def _trade_close_evidence(sample: dict[str, Any]) -> dict[str, Any]:
    return _safe_dict(_trade_close_raw(sample).get("close_evidence"))


def _trade_fee(sample: dict[str, Any]) -> float | None:
    fee_source = sample.get("fee_estimate")
    if fee_source is None:
        fee_source = sample.get("fee")
    fee = _safe_float(fee_source, None)
    return None if fee is None else abs(fee)


def _trade_funding_fee(sample: dict[str, Any]) -> float | None:
    for value in (
        sample.get("funding_fee"),
        _trade_close_raw(sample).get("funding_fee"),
        _trade_entry_raw(sample).get("funding_fee"),
    ):
        fee = _safe_float(value, None)
        if fee is not None:
            return fee
    return None


def _trade_notional_usdt(sample: dict[str, Any]) -> float | None:
    quantity = _safe_float(sample.get("quantity"), None)
    entry_price = _safe_float(sample.get("entry_price"), None)
    explicit = _first_float(
        sample.get("notional_usdt"),
        _trade_sizing(sample).get("final_notional_usdt"),
        _trade_sizing(sample).get("target_min_notional_usdt"),
    )
    derived = (
        abs(quantity * entry_price)
        if quantity is not None and entry_price is not None
        else None
    )
    # Historical rows can contain contract-count placeholders in notional_usdt.
    # Prefer the largest internally valid candidate so a tiny malformed denominator
    # cannot turn an ordinary perpetual PnL into a million-percent training return.
    candidates = [
        abs(value)
        for value in (explicit, derived)
        if value is not None and math.isfinite(value) and abs(value) > 0
    ]
    return max(candidates) if candidates else None


def _trade_actual_leverage(sample: dict[str, Any]) -> float | None:
    return _first_float(sample.get("leverage"))


def _trade_planned_leverage(sample: dict[str, Any]) -> float | None:
    return _first_float(
        sample.get("decision_suggested_leverage"),
        sample.get("suggested_leverage"),
        _trade_plan(sample).get("leverage"),
    )


def _trade_position_size_pct(sample: dict[str, Any]) -> float | None:
    return _first_float(
        sample.get("position_size_pct"),
        _trade_sizing(sample).get("position_size_pct"),
        _trade_plan(sample).get("position_size_pct"),
    )


def _trade_losing_exit_attribution(sample: dict[str, Any]) -> str:
    pnl = _safe_float(sample.get("realized_pnl"), 0.0) or 0.0
    if pnl >= 0:
        return ""
    return normalize_losing_exit_attribution(
        sample,
        entry_raw=_trade_entry_raw(sample),
        close_raw=_trade_close_raw(sample),
        shadow=_trade_shadow(sample),
    )


def _trade_profit_class(
    sample: dict[str, Any],
    *,
    fee: float | None,
    attribution: str,
) -> str:
    pnl = _safe_float(sample.get("realized_pnl"), 0.0) or 0.0
    if pnl > 0:
        if fee and pnl <= fee * 1.5:
            return "micro_profit_after_cost"
        return "net_profit"
    if pnl < 0:
        if attribution == "position_too_small_fee_drag":
            return "cost_drag_loss"
        if fee and abs(pnl) <= fee * _QUALITY_PARAMS.fee_dominated_multiple:
            return "cost_drag_loss"
        return "net_loss"
    return "flat"


def _trade_exit_timing_label(sample: dict[str, Any], *, attribution: str) -> str:
    if attribution in {"exit_too_early", "hold_too_short"}:
        return "too_early"
    if attribution in {"exit_too_late", "capital_release_forced_loss"}:
        return "too_late"
    close_evidence = _trade_close_evidence(sample)
    if _safe_float(sample.get("realized_pnl"), 0.0) > 0 and (
        close_evidence.get("profit_protection")
        or close_evidence.get("profit_retrace_protection")
        or close_evidence.get("small_position_profit_lock")
    ):
        return "profit_locked"
    return "neutral_or_unknown"


def _trade_size_efficiency_label(
    sample: dict[str, Any],
    *,
    attribution: str,
    actual_leverage: float | None,
    planned_leverage: float | None,
) -> str:
    if attribution == "position_too_small_fee_drag":
        return "too_small_fee_drag"
    if (
        planned_leverage is not None
        and planned_leverage >= 2.0
        and actual_leverage is not None
        and actual_leverage <= max(1.0, planned_leverage * 0.6)
    ):
        return "underleveraged_vs_plan"
    position_size_pct = _trade_position_size_pct(sample)
    if position_size_pct is not None and position_size_pct <= 0.015:
        return "tiny_size"
    notional_usdt = _trade_notional_usdt(sample)
    if notional_usdt is not None and notional_usdt < 15.0:
        return "tiny_notional"
    return "adequate_or_unknown"


def _trade_payoff_label(
    sample: dict[str, Any],
    *,
    attribution: str,
    exit_timing_label: str,
    fee: float | None,
) -> str:
    pnl = _safe_float(sample.get("realized_pnl"), 0.0) or 0.0
    if pnl < 0 and attribution in {"exit_too_late", "capital_release_forced_loss"}:
        return "loss_dragged"
    if pnl < 0 and attribution in {"exit_too_early", "hold_too_short"}:
        return "loss_cut_fast"
    if pnl > 0 and exit_timing_label == "profit_locked":
        return "profit_locked"
    if pnl > 0 and fee and pnl <= fee * 1.5:
        return "micro_profit"
    return "normal_or_unknown"


def _trade_cost_basis_label(sample: dict[str, Any], *, fee: float | None) -> str:
    funding_fee = _trade_funding_fee(sample)
    if fee is None:
        return "missing_fee"
    if funding_fee is None:
        return "fee_only"
    return "fee_plus_funding"


def _trade_strategy_context(sample: dict[str, Any]) -> dict[str, Any]:
    plan = _trade_plan(sample)
    return {
        "decision_lane": _safe_str(plan.get("decision_lane")),
        "exit_plan_id": _safe_str(plan.get("exit_plan_id")),
        "strategy_profile_id": _safe_str(plan.get("strategy_profile_id")),
        "position_size_pct": _trade_position_size_pct(sample),
        "planned_leverage": _trade_planned_leverage(sample),
        "actual_leverage": _trade_actual_leverage(sample),
    }


def _trade_profit_learning_labels(
    sample: dict[str, Any],
    assessment: SampleQualityAssessment,
) -> dict[str, Any]:
    fee = _trade_fee(sample)
    funding_fee = _trade_funding_fee(sample)
    actual_leverage = _trade_actual_leverage(sample)
    planned_leverage = _trade_planned_leverage(sample)
    attribution = _trade_losing_exit_attribution(sample)
    exit_timing_label = _trade_exit_timing_label(sample, attribution=attribution)
    size_efficiency_label = _trade_size_efficiency_label(
        sample,
        attribution=attribution,
        actual_leverage=actual_leverage,
        planned_leverage=planned_leverage,
    )
    trade_profit_class = _trade_profit_class(sample, fee=fee, attribution=attribution)
    cost_basis_label = _trade_cost_basis_label(sample, fee=fee)
    pnl = _safe_float(sample.get("realized_pnl"), 0.0) or 0.0
    fee_dominated = bool(fee and abs(pnl) <= fee * _QUALITY_PARAMS.fee_dominated_multiple)
    notional = _trade_notional_usdt(sample)
    return_after_cost_pct = (
        None if notional is None or notional <= 0 else pnl / max(notional, 1e-9) * 100.0
    )
    return {
        "version": PROFIT_LEARNING_VERSION,
        "sample_kind": "trade",
        "training_supervision_ready": bool(
            not assessment.exclude_from_training
            and attribution not in {"okx_slippage_or_execution", "unknown_requires_review"}
        ),
        "trade_profit_class": trade_profit_class,
        "losing_exit_attribution": attribution,
        "exit_timing_label": exit_timing_label,
        "size_efficiency_label": size_efficiency_label,
        "payoff_profile_label": _trade_payoff_label(
            sample,
            attribution=attribution,
            exit_timing_label=exit_timing_label,
            fee=fee,
        ),
        "cost_basis_label": cost_basis_label,
        "fee_dominated": fee_dominated,
        "realized_net_pnl_usdt": pnl,
        "return_after_cost_pct": return_after_cost_pct,
        "fee_estimate_usdt": fee,
        "funding_fee_usdt": funding_fee,
        "notional_usdt": notional,
        "strategy_context": _trade_strategy_context(sample),
    }


def _shadow_profit_learning_labels(
    sample: dict[str, Any],
    assessment: SampleQualityAssessment,
) -> dict[str, Any]:
    best_action = _sample_best_direction(sample)
    best_return = _actual_return_for_side(sample, best_action) if best_action else None
    missed = bool(sample.get("missed_opportunity")) and best_action in {"long", "short"}
    if missed and (best_return or 0.0) > 0:
        missed_label = "missed_positive_entry"
    elif missed:
        missed_label = "missed_non_positive_entry"
    else:
        missed_label = "no_missed_opportunity"
    if best_return is None:
        outcome_label = "unknown_outcome"
    elif best_return > 0:
        outcome_label = "positive_shadow_edge"
    elif best_return < 0:
        outcome_label = "negative_shadow_edge"
    else:
        outcome_label = "flat_shadow_edge"
    return {
        "version": PROFIT_LEARNING_VERSION,
        "sample_kind": "shadow",
        "training_supervision_ready": bool(not assessment.exclude_from_training),
        "missed_opportunity_label": missed_label,
        "shadow_outcome_label": outcome_label,
        "opportunity_side": best_action,
    }


def _profit_learning_labels(
    sample: dict[str, Any],
    kind: SampleKind,
    assessment: SampleQualityAssessment,
) -> dict[str, Any]:
    if kind == "trade":
        return _trade_profit_learning_labels(sample, assessment)
    if kind == "shadow":
        return _shadow_profit_learning_labels(sample, assessment)
    return {}


def _apply_profit_learning_aliases(
    annotated: dict[str, Any],
    *,
    kind: SampleKind,
    labels: dict[str, Any],
) -> None:
    if kind == "trade":
        for key in (
            "trade_profit_class",
            "losing_exit_attribution",
            "exit_timing_label",
            "size_efficiency_label",
            "payoff_profile_label",
            "cost_basis_label",
        ):
            value = labels.get(key)
            if isinstance(value, str) and value:
                annotated[key] = value
    if kind == "shadow":
        for key in ("missed_opportunity_label", "shadow_outcome_label"):
            value = labels.get(key)
            if isinstance(value, str) and value:
                annotated[key] = value


def annotate_sample(sample: dict[str, Any], kind: SampleKind) -> dict[str, Any]:
    assessment = ASSESSORS[kind](sample)
    annotated = dict(sample)
    annotated.update(assessment.as_dict())
    labels = _profit_learning_labels(annotated, kind, assessment)
    if labels:
        annotated["profit_learning_labels"] = labels
        _apply_profit_learning_aliases(annotated, kind=kind, labels=labels)
    return annotated


def annotate_samples(samples: list[dict[str, Any]], kind: SampleKind) -> list[dict[str, Any]]:
    return [annotate_sample(sample, kind) for sample in samples]


def trainable_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sample for sample in samples if not bool(sample.get("exclude_from_training"))]


def _profit_learning_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = {}
    supervision_ready = 0
    trade_pnls: list[float] = []
    trade_returns: list[float] = []
    gross_profit = 0.0
    gross_loss = 0.0
    win_count = 0
    loss_count = 0
    flat_count = 0
    for sample in samples:
        labels = _safe_dict(sample.get("profit_learning_labels"))
        if not labels:
            continue
        supervision_ready_for_sample = bool(labels.get("training_supervision_ready"))
        if supervision_ready_for_sample:
            supervision_ready += 1
            if _safe_str(labels.get("sample_kind")) == "trade":
                pnl = _safe_float(labels.get("realized_net_pnl_usdt"), None)
                if pnl is None:
                    pnl = _safe_float(sample.get("realized_pnl"), None)
                if pnl is not None:
                    trade_pnls.append(pnl)
                    if pnl > 0:
                        gross_profit += pnl
                        win_count += 1
                    elif pnl < 0:
                        gross_loss += abs(pnl)
                        loss_count += 1
                    else:
                        flat_count += 1
                return_pct = _safe_float(labels.get("return_after_cost_pct"), None)
                if return_pct is not None:
                    trade_returns.append(return_pct)
        for key, value in labels.items():
            if key in {
                "version",
                "sample_kind",
                "training_supervision_ready",
                "strategy_context",
                "realized_net_pnl_usdt",
                "return_after_cost_pct",
                "fee_estimate_usdt",
                "funding_fee_usdt",
                "notional_usdt",
                "fee_dominated",
            }:
                continue
            if isinstance(value, str) and value:
                counters.setdefault(key, Counter())[value] += 1
    avg_win = gross_profit / max(win_count, 1)
    avg_loss = gross_loss / max(loss_count, 1)
    profit_factor = 999.0 if gross_loss <= 0 and gross_profit > 0 else gross_profit / max(gross_loss, 1e-9)
    small_win_big_loss_ratio = avg_loss / max(avg_win, 1e-9) if win_count and loss_count else 0.0
    quality_warnings: list[str] = []
    if supervision_ready == 0:
        quality_warnings.append("no_supervision_ready_trade_samples")
    if trade_pnls and gross_profit <= gross_loss:
        quality_warnings.append("gross_loss_not_covered_by_profit")
    if win_count and loss_count and avg_loss > avg_win:
        quality_warnings.append("avg_loss_larger_than_avg_win")
    return {
        "supervision_ready_count": supervision_ready,
        "after_fee_quality": {
            "trade_count": len(trade_pnls),
            "win_count": win_count,
            "loss_count": loss_count,
            "flat_count": flat_count,
            "win_rate": round(win_count / max(len(trade_pnls), 1), 6),
            "net_realized_pnl_usdt": round(sum(trade_pnls), 6),
            "gross_profit_usdt": round(gross_profit, 6),
            "gross_loss_usdt": round(gross_loss, 6),
            "profit_factor": round(profit_factor, 6),
            "avg_net_pnl_usdt": round(sum(trade_pnls) / max(len(trade_pnls), 1), 6),
            "avg_win_usdt": round(avg_win, 6),
            "avg_loss_usdt": round(avg_loss, 6),
            "avg_return_after_cost_pct": round(
                sum(trade_returns) / max(len(trade_returns), 1),
                6,
            ),
            "small_win_big_loss_ratio": round(small_win_big_loss_ratio, 6),
            "quality_warnings": quality_warnings,
        },
        "label_counts": {
            key: [{"value": value, "count": count} for value, count in counter.most_common(20)]
            for key, counter in counters.items()
        },
    }


def _sample_action(sample: dict[str, Any]) -> str:
    return (
        _safe_str(sample.get("decision_action"))
        or _safe_str(sample.get("best_action"))
        or _safe_str(sample.get("side"))
        or "unknown"
    ).lower()


def _sample_source(sample: dict[str, Any], kind: str) -> str:
    source = _safe_str(sample.get("source"))
    platform = _safe_str(sample.get("platform"))
    if kind == "text_sentiment" and platform and source in {"", "news", "social"}:
        return platform
    return source or platform


def _sample_best_direction(sample: dict[str, Any]) -> str:
    best = _safe_str(sample.get("best_action")).lower()
    if best in {"long", "short"}:
        return best
    long_return = _safe_float(sample.get("long_return_pct"), None)
    short_return = _safe_float(sample.get("short_return_pct"), None)
    if long_return is None or short_return is None:
        return ""
    if long_return == short_return:
        return "flat"
    return "long" if long_return > short_return else "short"


def _sample_symbol(sample: dict[str, Any]) -> str:
    return _safe_str(sample.get("symbol")) or _safe_str(_features(sample).get("symbol"))


def _actual_return_for_side(sample: dict[str, Any], side: str) -> float | None:
    if side == "long":
        return _safe_float(sample.get("long_return_pct"), None)
    if side == "short":
        return _safe_float(sample.get("short_return_pct"), None)
    return None


def _shadow_tool_direction(tool: dict[str, Any]) -> str:
    side = _safe_str(
        tool.get("timesfm_shadow_side")
        or tool.get("chronos_shadow_side")
        or tool.get("best_side")
        or tool.get("side")
    ).lower()
    if side in {"long", "short"}:
        return side
    direction = _safe_str(tool.get("direction")).lower()
    return "long" if direction == "up" else "short" if direction == "down" else ""


def _professional_shadow_actual(tool: dict[str, Any]) -> bool:
    professional = tool.get("professional_model_shadow")
    if not isinstance(professional, dict):
        return False
    if bool(professional.get("actual_inference")):
        return True
    for key in ("primary_shadow_result", "challenger_shadow_result", "shadow_result"):
        result = professional.get(key)
        if isinstance(result, dict) and result.get("actual_inference"):
            return True
    return False


def _baseline_only_shadow(tool: dict[str, Any]) -> bool:
    professional = tool.get("professional_model_shadow")
    if not isinstance(professional, dict):
        return False
    if bool(tool.get("specialist_inference_active")) or _professional_shadow_actual(tool):
        return False
    return bool(professional.get("baseline_response"))


def _shadow_expected_return(tool: dict[str, Any]) -> float | None:
    for key in (
        "timesfm_shadow_expected_return_pct",
        "timesfm_shadow_expected_move_pct",
        "chronos_shadow_expected_return_pct",
        "chronos_shadow_expected_move_pct",
        "expected_return_pct",
        "expected_move_pct",
    ):
        value = _safe_float(tool.get(key), None)
        if value is not None:
            return value
    professional = tool.get("professional_model_shadow")
    if isinstance(professional, dict):
        result = professional.get("shadow_result")
        if isinstance(result, dict):
            for key in ("expected_return_pct", "expected_move_pct"):
                value = _safe_float(result.get(key), None)
                if value is not None:
                    return value
    return None


def _shadow_result_direction(result: dict[str, Any]) -> str:
    side = _safe_str(result.get("best_side") or result.get("side")).lower()
    if side in {"long", "short"}:
        return side
    direction = _safe_str(result.get("direction")).lower()
    return "long" if direction == "up" else "short" if direction == "down" else ""


def _shadow_result_expected_return(result: dict[str, Any]) -> float | None:
    for key in ("expected_return_pct", "expected_move_pct"):
        value = _safe_float(result.get(key), None)
        if value is not None:
            return value
    return None


def _shadow_result_actual(result: Any) -> bool:
    return bool(isinstance(result, dict) and result.get("actual_inference"))


def _shadow_tool_model_name(tool_name: str, tool: dict[str, Any]) -> str:
    professional = tool.get("professional_model_shadow")
    if isinstance(professional, dict):
        result = professional.get("shadow_result")
        if isinstance(result, dict):
            model = _safe_str(result.get("model"))
            if model:
                return model
    if (
        tool_name == "time_series_prediction"
        and tool.get("timesfm_shadow_expected_return_pct") is not None
    ):
        return "timesfm_shadow_challenger"
    if (
        tool_name == "time_series_prediction"
        and tool.get("chronos_shadow_expected_return_pct") is not None
    ):
        return "chronos_shadow_primary"
    return _safe_str(tool.get("model")) or tool_name


def _shadow_model_key(tool_name: str, model_name: str) -> str:
    return tool_name if not model_name or model_name == tool_name else f"{tool_name}:{model_name}"


def _time_series_shadow_candidates(tool: dict[str, Any]) -> list[dict[str, Any]]:
    professional = tool.get("professional_model_shadow")
    if not isinstance(professional, dict):
        return []
    candidates: list[dict[str, Any]] = []
    for key in ("primary_shadow_result", "challenger_shadow_result"):
        result = professional.get(key)
        if not isinstance(result, dict) or not _shadow_result_actual(result):
            continue
        model = _safe_str(result.get("model"))
        if not model:
            continue
        candidates.append(
            {
                "tool": "time_series_prediction",
                "model": model,
                "direction": _shadow_result_direction(result),
                "expected_return_pct": _shadow_result_expected_return(result),
                "actual_inference": True,
                "specialist_inference_active": bool(tool.get("specialist_inference_active")),
                "sequence_length": int(_safe_float(result.get("sequence_length"), 0.0) or 0),
                "legacy_mixed_shadow": False,
            }
        )
    if candidates:
        return candidates

    result = professional.get("shadow_result")
    if not isinstance(result, dict) or not _shadow_result_actual(result):
        return []
    model = _safe_str(result.get("model"))
    if not model:
        return []
    expected_return = _shadow_result_expected_return(result)
    if expected_return is None:
        expected_return = _shadow_expected_return(tool)
    return [
        {
            "tool": "time_series_prediction",
            "model": model,
            "direction": _shadow_result_direction(result) or _shadow_tool_direction(tool),
            "expected_return_pct": expected_return,
            "actual_inference": True,
            "specialist_inference_active": bool(tool.get("specialist_inference_active")),
            "sequence_length": int(_safe_float(result.get("sequence_length"), 0.0) or 0),
            "legacy_mixed_shadow": True,
        }
    ]


def _shadow_tool_candidates(tool_name: str, tool: dict[str, Any]) -> list[dict[str, Any]]:
    if tool_name == "time_series_prediction":
        candidates = _time_series_shadow_candidates(tool)
        if candidates:
            return candidates
    return [
        {
            "tool": tool_name,
            "model": _shadow_tool_model_name(tool_name, tool),
            "direction": _shadow_tool_direction(tool),
            "expected_return_pct": _shadow_expected_return(tool),
            "actual_inference": _professional_shadow_actual(tool)
            or bool(tool.get("specialist_inference_active")),
            "specialist_inference_active": bool(tool.get("specialist_inference_active")),
            "sequence_length": 0,
            "legacy_mixed_shadow": False,
        }
    ]


def _finalize_shadow_model_row(row: dict[str, Any]) -> dict[str, Any]:
    direction_count = int(row.get("direction_count") or 0)
    expected_count = int(row.get("shadow_expected_return_count") or 0)
    realized_sum = float(row.get("realized_return_sum_pct") or 0.0)
    hit_rate = float(row.get("direction_hit_count") or 0) / max(direction_count, 1)
    avg_realized = realized_sum / max(direction_count, 1)
    avg_expected = float(row.get("shadow_expected_return_sum") or 0.0) / max(expected_count, 1)
    blockers: list[str] = []
    if int(row.get("actual_inference_count") or 0) < MIN_PROMOTION_SHADOW_SAMPLES:
        blockers.append("specialist_shadow_sample_floor_not_met")
    if direction_count >= MIN_PROMOTION_SHADOW_SAMPLES and hit_rate < MIN_DIRECTION_HIT_RATE:
        blockers.append("direction_hit_rate_below_floor")
    if direction_count >= MIN_PROMOTION_SHADOW_SAMPLES and avg_realized < MIN_AVG_REALIZED_RETURN_PCT:
        blockers.append("avg_realized_return_below_floor")
    worst = row.get("worst_realized_return_pct")
    if worst is not None and float(worst) <= MAX_FALSE_SIGNAL_LOSS_PCT:
        blockers.append("false_signal_loss_exceeds_floor")
    if int(row.get("sequence_too_short_count") or 0) > 0:
        blockers.append("timeseries_sequence_too_short_for_promotion")
    blocker_counts = dict(Counter(blockers))
    row["direction_hit_rate"] = round(hit_rate, 4)
    row["avg_shadow_expected_return_pct"] = round(avg_expected, 6)
    row["avg_expected_return_pct"] = round(avg_expected, 6)
    row["avg_realized_return_pct"] = round(avg_realized, 6)
    row["false_signal_count"] = int(row.get("false_signal_count") or 0)
    row["tail_loss_count"] = int(row.get("tail_loss_count") or 0)
    row["tail_loss_symbols"] = [
        {"symbol": symbol, "count": count}
        for symbol, count in row["tail_loss_symbols"].most_common(10)
    ]
    row["worst_samples"] = list(row.get("worst_samples") or [])[:MAX_WORST_SAMPLE_COUNT]
    row["sequence_too_short_count"] = int(row.get("sequence_too_short_count") or 0)
    row["legacy_mixed_shadow_count"] = int(row.get("legacy_mixed_shadow_count") or 0)
    row["legacy_quarantined_count"] = int(row.get("legacy_quarantined_count") or 0)
    row["legacy_sequence_too_short_count"] = int(
        row.get("legacy_sequence_too_short_count") or 0
    )
    row["promotion_ready"] = not bool(blockers)
    row["promotion_blockers"] = blockers
    row["blockers"] = blockers
    row["blocked_reasons"] = blockers
    row["blocked_reason_counts"] = blocker_counts
    row["promotion_gate"] = {
        "minimum_actual_inference_samples": MIN_PROMOTION_SHADOW_SAMPLES,
        "minimum_direction_hit_rate": MIN_DIRECTION_HIT_RATE,
        "minimum_avg_realized_return_pct": MIN_AVG_REALIZED_RETURN_PCT,
        "max_false_signal_loss_pct": MAX_FALSE_SIGNAL_LOSS_PCT,
        "minimum_timeseries_sequence_length": MIN_TIMESERIES_SEQUENCE_LENGTH,
        "actual_inference_count": int(row.get("actual_inference_count") or 0),
        "direction_count": direction_count,
        "direction_hit_rate": round(hit_rate, 4),
        "avg_realized_return_pct": round(avg_realized, 6),
        "worst_realized_return_pct": row.get("worst_realized_return_pct"),
        "tail_loss_count": row["tail_loss_count"],
        "sequence_too_short_count": row["sequence_too_short_count"],
        "legacy_mixed_shadow_count": row["legacy_mixed_shadow_count"],
        "legacy_quarantined_count": row["legacy_quarantined_count"],
        "legacy_sequence_too_short_count": row["legacy_sequence_too_short_count"],
    }
    row.pop("realized_return_sum_pct", None)
    row.pop("shadow_expected_return_sum", None)
    return row


def _compact_worst_shadow_sample(
    sample: dict[str, Any],
    *,
    tool_name: str,
    model_name: str,
    predicted_side: str,
    actual_side: str,
    actual_return: float,
    expected_return: float | None,
    sequence_length: int,
    legacy_mixed_shadow: bool,
) -> dict[str, Any]:
    return {
        "shadow_backtest_id": sample.get("id"),
        "symbol": _sample_symbol(sample),
        "tool": tool_name,
        "model": model_name,
        "predicted_side": predicted_side,
        "actual_best_side": actual_side,
        "actual_return_pct": round(float(actual_return), 6),
        "expected_return_pct": None if expected_return is None else round(float(expected_return), 6),
        "long_return_pct": _safe_float(sample.get("long_return_pct"), None),
        "short_return_pct": _safe_float(sample.get("short_return_pct"), None),
        "sequence_length": sequence_length,
        "legacy_mixed_shadow": bool(legacy_mixed_shadow),
    }


def _remember_worst_shadow_sample(row: dict[str, Any], sample: dict[str, Any]) -> None:
    samples = row.setdefault("worst_samples", [])
    samples.append(sample)
    samples.sort(key=lambda item: float(item.get("actual_return_pct") or 0.0))
    del samples[MAX_WORST_SAMPLE_COUNT:]


def _shadow_model_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for sample in samples:
        features = _features(sample)
        local_shadow = features.get("local_ai_tools_shadow")
        if not isinstance(local_shadow, dict):
            continue
        actual_side = _sample_best_direction(sample)
        for tool_name in ("profit_prediction", "time_series_prediction", "sentiment_analysis"):
            tool = local_shadow.get(tool_name)
            if not isinstance(tool, dict):
                continue
            if _baseline_only_shadow(tool):
                continue
            if not (bool(tool.get("specialist_inference_active")) or _professional_shadow_actual(tool)):
                continue
            for candidate in _shadow_tool_candidates(tool_name, tool):
                if not candidate.get("actual_inference"):
                    continue
                model_name = _safe_str(candidate.get("model")) or _shadow_tool_model_name(tool_name, tool)
                row_key = _shadow_model_key(tool_name, model_name)
                row = rows.setdefault(
                    row_key,
                    {
                        "tool": tool_name,
                        "model": model_name,
                        "model_key": row_key,
                        "sample_count": 0,
                        "actual_inference_count": 0,
                        "direction_count": 0,
                        "direction_hit_count": 0,
                        "shadow_expected_return_sum": 0.0,
                        "shadow_expected_return_count": 0,
                        "realized_return_sum_pct": 0.0,
                        "false_signal_count": 0,
                        "worst_realized_return_pct": None,
                        "best_realized_return_pct": None,
                        "tail_loss_count": 0,
                        "tail_loss_symbols": Counter(),
                        "worst_samples": [],
                        "specialist_inference_count": 0,
                        "sequence_too_short_count": 0,
                        "legacy_mixed_shadow_count": 0,
                        "legacy_quarantined_count": 0,
                        "legacy_sequence_too_short_count": 0,
                    },
                )
                row["sample_count"] += 1
                if bool(candidate.get("specialist_inference_active")):
                    row["specialist_inference_count"] += 1
                legacy_mixed_shadow = bool(candidate.get("legacy_mixed_shadow"))
                if legacy_mixed_shadow:
                    row["legacy_mixed_shadow_count"] += 1
                sequence_length = int(candidate.get("sequence_length") or 0)
                sequence_too_short = (
                    tool_name == "time_series_prediction"
                    and sequence_length < MIN_TIMESERIES_SEQUENCE_LENGTH
                )
                if sequence_too_short:
                    row["legacy_sequence_too_short_count"] += 1
                if legacy_mixed_shadow or sequence_too_short:
                    row["legacy_quarantined_count"] += 1
                    continue
                row["actual_inference_count"] += 1
                direction = _safe_str(candidate.get("direction")).lower()
                if direction in {"long", "short"}:
                    actual_return = _actual_return_for_side(sample, direction)
                    if actual_return is not None:
                        row["direction_count"] += 1
                        row["realized_return_sum_pct"] += actual_return
                        if actual_side == direction:
                            row["direction_hit_count"] += 1
                        elif actual_return < 0:
                            row["false_signal_count"] += 1
                        worst = row.get("worst_realized_return_pct")
                        best = row.get("best_realized_return_pct")
                        row["worst_realized_return_pct"] = (
                            round(actual_return, 6)
                            if worst is None
                            else round(min(float(worst), actual_return), 6)
                        )
                        row["best_realized_return_pct"] = (
                            round(actual_return, 6)
                            if best is None
                            else round(max(float(best), actual_return), 6)
                        )
                        symbol = _sample_symbol(sample)
                        if actual_return <= MAX_FALSE_SIGNAL_LOSS_PCT:
                            row["tail_loss_count"] += 1
                            if symbol:
                                row["tail_loss_symbols"][symbol] += 1
                        _remember_worst_shadow_sample(
                            row,
                            _compact_worst_shadow_sample(
                                sample,
                                tool_name=tool_name,
                                model_name=model_name,
                                predicted_side=direction,
                                actual_side=actual_side,
                                actual_return=actual_return,
                                expected_return=_safe_float(
                                    candidate.get("expected_return_pct"),
                                    None,
                                ),
                                sequence_length=sequence_length,
                                legacy_mixed_shadow=legacy_mixed_shadow,
                            ),
                        )
                expected = _safe_float(candidate.get("expected_return_pct"), None)
                if expected is not None:
                    row["shadow_expected_return_sum"] += expected
                    row["shadow_expected_return_count"] += 1
    for row in rows.values():
        _finalize_shadow_model_row(row)
    return rows


def quality_report(samples_by_kind: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_kind: dict[str, Any] = {}
    totals = Counter()
    reason_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    timeframe_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    trainable_weight_total = 0.0
    for kind, samples in samples_by_kind.items():
        status_counts: Counter[str] = Counter()
        weight_total = 0.0
        kind_action_counts: Counter[str] = Counter()
        kind_trainable_action_counts: Counter[str] = Counter()
        kind_timeframe_counts: Counter[str] = Counter()
        kind_trainable_timeframe_counts: Counter[str] = Counter()
        kind_source_counts: Counter[str] = Counter()
        kind_trainable_source_counts: Counter[str] = Counter()
        benign_downweighted = 0
        contamination_downweighted = 0
        for sample in samples:
            status = _safe_str(sample.get("data_quality_status")) or "unknown"
            trainable = status != "excluded"
            status_counts[status] += 1
            totals[status] += 1
            if status == "downweighted":
                if _is_benign_downweighted_sample(kind, sample):
                    benign_downweighted += 1
                    totals["benign_downweighted"] += 1
                else:
                    contamination_downweighted += 1
                    totals["contamination_downweighted"] += 1
            weight = float(_safe_float(sample.get("sample_weight"), 0.0) or 0.0)
            weight_total += weight
            if trainable:
                trainable_weight_total += weight
            action = _sample_action(sample)
            kind_action_counts[action] += 1
            action_counts[f"{kind}:{action}"] += 1
            if trainable:
                kind_trainable_action_counts[action] += 1
            timeframe = _safe_str(sample.get("timeframe") or sample.get("horizon_minutes"))
            if timeframe:
                kind_timeframe_counts[timeframe] += 1
                timeframe_counts[f"{kind}:{timeframe}"] += 1
                if trainable:
                    kind_trainable_timeframe_counts[timeframe] += 1
            source = _sample_source(sample, kind)
            if source:
                kind_source_counts[source] += 1
                source_counts[f"{kind}:{source}"] += 1
                if trainable:
                    kind_trainable_source_counts[source] += 1
            for reason in sample.get("quality_reasons") or []:
                reason_key = f"{kind}:{reason}"
                reason_counts[reason_key] += 1
        total = len(samples)
        by_kind[kind] = {
            "total": total,
            "included": int(status_counts.get("included", 0)),
            "downweighted": int(status_counts.get("downweighted", 0)),
            "benign_downweighted": int(benign_downweighted),
            "contamination_downweighted": int(contamination_downweighted),
            "excluded": int(status_counts.get("excluded", 0)),
            "effective_weight": round(weight_total, 4),
            "effective_weight_ratio": round(weight_total / max(total, 1), 4),
            "actions": dict(kind_action_counts),
            "trainable_actions": dict(kind_trainable_action_counts),
            "timeframes": dict(kind_timeframe_counts),
            "trainable_timeframes": dict(kind_trainable_timeframe_counts),
            "sources": dict(kind_source_counts),
            "trainable_sources": dict(kind_trainable_source_counts),
            "profit_learning": _profit_learning_report(samples),
        }
    total_count = sum(len(samples) for samples in samples_by_kind.values())
    return {
        "data_quality_version": DATA_QUALITY_VERSION,
        "policy": asdict(DEFAULT_TRADING_PARAMS.training_data_quality),
        "by_kind": by_kind,
        "profit_learning_summary": {
            kind: _profit_learning_report(samples) for kind, samples in samples_by_kind.items()
        },
        "specialist_shadow_models": _shadow_model_report(samples_by_kind.get("shadow", [])),
        "totals": {
            "total": total_count,
            "included": int(totals.get("included", 0)),
            "downweighted": int(totals.get("downweighted", 0)),
            "benign_downweighted": int(totals.get("benign_downweighted", 0)),
            "contamination_downweighted": int(totals.get("contamination_downweighted", 0)),
            "excluded": int(totals.get("excluded", 0)),
            "effective_weight": round(trainable_weight_total, 4),
            "effective_weight_ratio": round(trainable_weight_total / max(total_count, 1), 4),
        },
        "top_reasons": [
            {"reason": reason, "count": count} for reason, count in reason_counts.most_common(20)
        ],
        "top_actions": [
            {"action": action, "count": count} for action, count in action_counts.most_common(20)
        ],
        "top_timeframes": [
            {"timeframe": timeframe, "count": count}
            for timeframe, count in timeframe_counts.most_common(20)
        ],
        "top_sources": [
            {"source": source, "count": count} for source, count in source_counts.most_common(20)
        ],
    }


def governance_report(
    quality: dict[str, Any],
    *,
    artifact_quality_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Return an operator-facing cleanup report for historical training samples.

    Historical rows are preserved for audit and PnL traceability.  Cleanup means
    samples are quarantined from training or downweighted before any model sees
    them, then dependent artifacts can be retrained/reindexed from the clean
    view.
    """

    totals = quality.get("totals") if isinstance(quality.get("totals"), dict) else {}
    total = int(totals.get("total") or 0)
    included = int(totals.get("included") or 0)
    downweighted = int(totals.get("downweighted") or 0)
    excluded = int(totals.get("excluded") or 0)
    trainable = included + downweighted
    effective_weight_ratio = float(totals.get("effective_weight_ratio") or 0.0)
    excluded_ratio = excluded / max(total, 1)
    top_reasons = quality.get("top_reasons") if isinstance(quality.get("top_reasons"), list) else []
    has_contamination = bool(excluded or downweighted or top_reasons)
    blocked_reason_count = sum(
        int(item.get("count") or 0)
        for item in top_reasons
        if isinstance(item, dict)
        and any(
            token in str(item.get("reason") or "")
            for token in (
                "manual_or_test_trade",
                "market_data_quality",
                "missing_features",
                "abnormal",
                "invalid",
            )
        )
    )
    blocked_reason_ratio = blocked_reason_count / max(total, 1)
    contamination_risk = _contamination_risk(
        has_contamination=has_contamination,
        excluded_ratio=excluded_ratio,
        blocked_reason_ratio=blocked_reason_ratio,
        effective_weight_ratio=effective_weight_ratio,
    )
    status = "clean"
    if excluded:
        status = "quarantined"
    elif downweighted:
        status = "downweighted"
    quality_fingerprint_payload = {
        "version": quality.get("data_quality_version") or DATA_QUALITY_VERSION,
        "total": total,
        "included": included,
        "downweighted": downweighted,
        "excluded": excluded,
        "effective_weight_ratio": round(effective_weight_ratio, 8),
        "top_reasons": top_reasons,
    }
    quality_fingerprint = hashlib.sha256(
        json.dumps(
            quality_fingerprint_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    artifact_matches_quality = bool(
        artifact_quality_fingerprint
        and artifact_quality_fingerprint == quality_fingerprint
    )
    return {
        "status": status,
        "data_quality_version": quality.get("data_quality_version") or DATA_QUALITY_VERSION,
        "training_policy": PHASE3_TRAINING_POLICY,
        "raw_records_preserved": True,
        "cleanup_mode": "quarantine_not_delete",
        "quarantine_applied": bool(excluded),
        "downweight_applied": bool(downweighted),
        "trainable_sample_count": trainable,
        "excluded_sample_count": excluded,
        "downweighted_sample_count": downweighted,
        "effective_weight_ratio": round(effective_weight_ratio, 4),
        "excluded_ratio": round(excluded_ratio, 6),
        "blocked_reason_ratio": round(blocked_reason_ratio, 6),
        "contamination_risk": contamination_risk,
        "blocked_reason_count": blocked_reason_count,
        "requires_artifact_refresh": bool(has_contamination and not artifact_matches_quality),
        "quality_fingerprint": quality_fingerprint,
        "artifact_quality_fingerprint": artifact_quality_fingerprint,
        "artifact_matches_quality": artifact_matches_quality,
        "refresh_targets": list(_RETRAIN_TARGETS),
        "summary": (
            f"已保留 {total} 条原始样本；{excluded} 条隔离不训练，"
            f"{downweighted} 条降权训练，{trainable} 条进入清洗后的训练视图。"
        ),
        "notes": [
            "原始交易、分析和复盘记录不删除，避免审计链断裂。",
            "模型训练、策略学习和向量记忆只消费清洗后的训练视图。",
            "清洗策略变更后应重训本地 ML、服务器量化工具，并重建向量索引。",
        ],
    }


def _contamination_risk(
    *,
    has_contamination: bool,
    excluded_ratio: float,
    blocked_reason_ratio: float,
    effective_weight_ratio: float,
) -> str:
    if not has_contamination:
        return "low"
    if (
        excluded_ratio >= HIGH_CONTAMINATION_EXCLUDED_RATIO
        or blocked_reason_ratio >= HIGH_CONTAMINATION_BLOCKED_REASON_RATIO
        or (effective_weight_ratio > 0 and effective_weight_ratio < 0.5)
    ):
        return "high"
    if excluded_ratio >= MEDIUM_CONTAMINATION_EXCLUDED_RATIO or blocked_reason_ratio > 0:
        return "medium"
    return "low"


def annotate_training_payload(
    *,
    shadow_samples: list[dict[str, Any]],
    trade_samples: list[dict[str, Any]],
    sequence_samples: list[dict[str, Any]],
    text_sentiment_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    annotated_shadow = annotate_samples(shadow_samples, "shadow")
    annotated_trade = annotate_samples(trade_samples, "trade")
    annotated_sequence = annotate_samples(sequence_samples, "sequence")
    annotated_text = annotate_samples(text_sentiment_samples, "text_sentiment")
    report = quality_report(
        {
            "shadow": annotated_shadow,
            "trade": annotated_trade,
            "sequence": annotated_sequence,
            "text_sentiment": annotated_text,
        }
    )
    return {
        "shadow_samples": trainable_samples(annotated_shadow),
        "trade_samples": trainable_samples(annotated_trade),
        "sequence_samples": trainable_samples(annotated_sequence),
        "text_sentiment_samples": trainable_samples(annotated_text),
        "quality_report": report,
        "governance_report": governance_report(report),
    }
