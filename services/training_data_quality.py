from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from core.market_facts import (
    MARKET_FACT_CONTRACT_VERSION,
    compact_market_fact_contract,
    market_fact_contract_reasons,
)
from core.training_contracts import (
    SHADOW_LABEL_VERSION,
    shadow_label_contract_reasons,
)
from services.dynamic_policy_values import empirical_policy_value
from services.execution_cost_model import execution_cost_estimate
from services.profit_supervision import (
    PROFIT_SUPERVISION_VERSION,
    apply_correlation_group_weights,
    build_profit_supervision_contract,
    profit_supervision_report,
)
from services.return_loss_attribution import normalize_losing_exit_attribution
from services.text_integrity import looks_like_mojibake
from services.trading_params import DEFAULT_TRADING_PARAMS

_QUALITY_PARAMS = DEFAULT_TRADING_PARAMS.training_data_quality
DATA_QUALITY_VERSION = "2026-07-14.separated-profit-supervision.v4"
PROFIT_LEARNING_VERSION = "separated-profit-supervision-v4"
PHASE3_TRAINING_POLICY = "clean_training_view_only"
MAX_WORST_SAMPLE_COUNT = 8
_SHADOW_BENIGN_DOWNWEIGHT_REASONS = {
    "hold_missed_opportunity_downweighted",
    "very_low_decision_confidence",
}
QualityStatus = Literal["included", "downweighted", "excluded"]
SampleKind = Literal["shadow", "trade", "sequence", "text_sentiment"]
COMPACT_SEQUENCE_SERIES_FORMAT = "compact_native_kline_series.v1"
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


def _shadow_market_fact_contract(features: dict[str, Any]) -> dict[str, Any]:
    compact = _safe_dict(features.get("training_market_fact_contract"))
    return compact or _safe_dict(features.get("market_fact_contract"))


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
        reasons.append(
            "duplicate_decision_horizon_label_version"
            if sample.get("duplicate_label_identity")
            else "duplicate_sample"
        )
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
        score = 0.0
    elif reasons:
        status = "downweighted"
        weight = max(0.0, min(score, 1.0))
    else:
        status = "included"
        weight = 1.0
    return SampleQualityAssessment(
        status=status, score=max(0.0, min(score, 1.0)), weight=weight, reasons=tuple(reasons)
    )


def _shadow_cost_completeness_reasons(features: dict[str, Any]) -> list[str]:
    execution_cost = execution_cost_estimate(features)
    reasons: list[str] = []
    if execution_cost.fee_pct <= 0:
        reasons.append("cost_incomplete:fee_rate_missing")
    if execution_cost.spread_source == "missing" or execution_cost.spread_pct <= 0:
        reasons.append("cost_incomplete:live_spread_missing")

    funding_rate = _safe_float(features.get("funding_rate"), None)
    if funding_rate is None:
        reasons.append("cost_incomplete:funding_rate_missing")
    funding_interval_minutes = _safe_float(features.get("funding_interval_minutes"), None)
    if funding_interval_minutes is None:
        funding_interval_hours = _safe_float(features.get("funding_interval_hours"), None)
        funding_interval_minutes = (
            funding_interval_hours * 60.0 if funding_interval_hours is not None else None
        )
    if funding_interval_minutes is None or funding_interval_minutes <= 0:
        reasons.append("cost_incomplete:funding_interval_missing")
    return reasons


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
    if price_warning:
        return _final_assessment(
            0.0,
            ["price_reconciliation:" f"{price_warning or 'indicator_close_diverged'}"],
            exclude=True,
        )

    if bool(features.get("stale")) or bool(features.get("ticker_stale")):
        return _final_assessment(0.0, ["stale_ticker_snapshot"], exclude=True)

    long_return = _safe_float(sample.get("long_return_pct"), None)
    short_return = _safe_float(sample.get("short_return_pct"), None)
    if long_return is None or short_return is None:
        return _final_assessment(0.0, ["missing_outcome_returns"], exclude=True)
    horizon = int(_safe_float(sample.get("horizon_minutes"), 0.0) or 0)
    if horizon <= 0:
        return _final_assessment(0.0, ["invalid_horizon_minutes"], exclude=True)

    label_contract_reasons = shadow_label_contract_reasons(
        features.get("training_label_contract"),
        decision_id=sample.get("decision_id"),
        horizon_minutes=horizon,
        label_version=sample.get("label_version"),
    )
    if label_contract_reasons:
        return _final_assessment(
            0.0,
            [f"shadow_label_contract:{reason}" for reason in label_contract_reasons],
            exclude=True,
        )

    cost_reasons = _shadow_cost_completeness_reasons(features)
    if cost_reasons:
        return _final_assessment(0.0, cost_reasons, exclude=True)

    fact_contract_reasons = market_fact_contract_reasons(
        _shadow_market_fact_contract(features)
    )
    if fact_contract_reasons:
        return _final_assessment(
            0.0,
            [f"market_fact_contract:{reason}" for reason in fact_contract_reasons],
            exclude=True,
        )

    return _final_assessment(score, reasons, exclude=exclude)


def assess_trade_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    reasons: list[str] = []
    score = 1.0
    exclude = False

    guard_reasons = _sample_guard_reasons(sample)
    if guard_reasons:
        return _final_assessment(0.0, guard_reasons, exclude=True)

    source = _safe_str(sample.get("source")).lower()
    if source == "okx_position_history":
        evidence_gaps = [
            _safe_str(reason)
            for reason in sample.get("training_evidence_gaps") or []
            if _safe_str(reason)
        ]
        if evidence_gaps:
            return _final_assessment(
                0.0,
                [f"incomplete_okx_lifecycle:{reason}" for reason in evidence_gaps],
                exclude=True,
            )
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

    fee_source = sample.get("fee_estimate")
    if fee_source is None:
        fee_source = sample.get("fee")
    if fee_source is None:
        return _final_assessment(0.0, ["missing_fee_estimate"], exclude=True)
    if source in {"closed_position", "okx_position_history"}:
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
        return _final_assessment(0.0, ["invalid_trade_price"], exclude=True)

    hold_minutes = _safe_float(sample.get("hold_minutes"), 0.0) or 0.0
    pnl = _safe_float(sample.get("realized_pnl"), 0.0) or 0.0
    if hold_minutes <= 0:
        return _final_assessment(0.0, ["missing_hold_duration"], exclude=True)

    outcome = _safe_str(sample.get("outcome")).lower()
    if outcome not in {"profit", "loss", "flat", "win"}:
        return _final_assessment(0.0, ["missing_or_unknown_outcome"], exclude=True)

    if pnl < 0:
        attribution = _trade_losing_exit_attribution(sample)
        if attribution == "okx_slippage_or_execution":
            return _final_assessment(0.0, ["execution_anomaly_trade"], exclude=True)
        if attribution == "unknown_requires_review":
            return _final_assessment(
                0.0, ["unknown_losing_exit_attribution"], exclude=True
            )

    return _final_assessment(score, reasons, exclude=exclude)


def assess_sequence_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    reasons: list[str] = []
    score = 1.0
    guard_reasons = _sample_guard_reasons(sample, sample.get("close_sequence"))
    if guard_reasons:
        return _final_assessment(0.0, guard_reasons, exclude=True)

    closes = sample.get("close_sequence") or []
    if not isinstance(closes, list) or not closes:
        return _final_assessment(0.0, ["empty_price_sequence"], exclude=True)
    numeric_closes = [_safe_float(value, None) for value in closes]
    if any(value is None or value <= 0 for value in numeric_closes):
        return _final_assessment(0.0, ["invalid_price_sequence"], exclude=True)
    if sample.get("sequence_format") == COMPACT_SEQUENCE_SERIES_FORMAT:
        volumes = sample.get("volume_sequence") or []
        if not isinstance(volumes, list) or len(volumes) != len(closes):
            return _final_assessment(
                0.0,
                ["compact_sequence_volume_alignment_invalid"],
                exclude=True,
            )
        numeric_volumes = [_safe_float(value, None) for value in volumes]
        if any(value is None or value < 0 for value in numeric_volumes):
            return _final_assessment(
                0.0,
                ["compact_sequence_volume_invalid"],
                exclude=True,
            )
        observation_count = int(
            _safe_float(sample.get("observation_count"), 0.0) or 0
        )
        if observation_count != max(len(closes) - 31, 0):
            return _final_assessment(
                0.0,
                ["compact_sequence_observation_count_mismatch"],
                exclude=True,
            )
        if (
            _safe_str(sample.get("label_name")) != "gross_market_move_pct"
            or not _safe_str(sample.get("label_version"))
        ):
            return _final_assessment(
                0.0,
                ["compact_sequence_label_contract_missing"],
                exclude=True,
            )
        timeframe = _safe_str(sample.get("timeframe"))
        if timeframe not in set(_QUALITY_PARAMS.allowed_sequence_timeframes):
            return _final_assessment(0.0, ["unknown_timeframe"], exclude=True)
        return _final_assessment(score, reasons)
    future_return = _safe_float(sample.get("future_return_pct"), None)
    if future_return is None:
        return _final_assessment(0.0, ["missing_future_return"], exclude=True)
    timeframe = _safe_str(sample.get("timeframe"))
    if timeframe not in set(_QUALITY_PARAMS.allowed_sequence_timeframes):
        return _final_assessment(0.0, ["unknown_timeframe"], exclude=True)
    return _final_assessment(score, reasons)


def assess_text_sentiment_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    text = _safe_str(sample.get("text"))
    guard_reasons = _sample_guard_reasons(sample, text)
    if guard_reasons:
        return _final_assessment(0.0, guard_reasons, exclude=True)
    if not text:
        return _final_assessment(0.0, ["empty_text"], exclude=True)
    score = 1.0
    reasons: list[str] = []
    platform = _safe_str(sample.get("platform")).lower()
    if platform in {"", "unknown"}:
        return _final_assessment(0.0, ["unknown_text_source"], exclude=True)
    sentiment = _safe_float(sample.get("sentiment_score"), None)
    if sentiment is None:
        return _final_assessment(0.0, ["missing_sentiment_score"], exclude=True)
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


def _trade_return_policy(sample: dict[str, Any]) -> dict[str, Any]:
    return _safe_dict(_trade_entry_raw(sample).get("production_return_policy"))


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
    quantity_is_contracts = _safe_str(sample.get("quantity_unit")).lower() == "contracts"
    derived = (
        abs(quantity * entry_price)
        if not quantity_is_contracts and quantity is not None and entry_price is not None
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


def _trade_return_after_cost_pct(
    sample: dict[str, Any], *, pnl: float, notional: float | None
) -> float | None:
    authoritative = _safe_float(sample.get("authoritative_pnl_ratio_pct"), None)
    if authoritative is not None:
        return authoritative
    if notional is None or notional <= 0:
        return None
    return pnl / max(notional, 1e-9) * 100.0


def _trade_actual_leverage(sample: dict[str, Any]) -> float | None:
    return _first_float(sample.get("leverage"))


def _trade_planned_leverage(sample: dict[str, Any]) -> float | None:
    return _first_float(
        sample.get("decision_suggested_leverage"),
        sample.get("suggested_leverage"),
    )


def _trade_position_size_pct(sample: dict[str, Any]) -> float | None:
    return _first_float(
        sample.get("position_size_pct"),
        _trade_sizing(sample).get("position_size_pct"),
        _trade_return_policy(sample).get("position_size_pct"),
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
        if fee and pnl <= fee:
            return "micro_profit_after_cost"
        return "net_profit"
    if pnl < 0:
        if attribution == "position_too_small_fee_drag":
            return "cost_drag_loss"
        if fee and abs(pnl) <= fee:
            return "cost_drag_loss"
        return "net_loss"
    return "flat"


def _trade_exit_timing_label(sample: dict[str, Any], *, attribution: str) -> str:
    if attribution in {"exit_too_early", "hold_too_short"}:
        return "too_early"
    if attribution in {"exit_too_late", "capital_release_forced_loss"}:
        return "too_late"
    close_evidence = _trade_close_evidence(sample)
    dynamic_exit = _safe_dict(close_evidence.get("dynamic_exit_policy"))
    if (
        _safe_float(sample.get("realized_pnl"), 0.0) > 0
        and dynamic_exit.get("eligible") is True
        and (_safe_float(dynamic_exit.get("profit_retrace_ratio"), 0.0) or 0.0) > 0.0
    ):
        return "profit_locked"
    return "neutral_or_unknown"


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
    if pnl > 0 and fee and pnl <= fee:
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
    return_policy = _trade_return_policy(sample)
    return {
        "return_policy_version": _safe_str(
            _safe_dict(return_policy.get("policy_provenance")).get("strategy_version")
        ),
        "return_policy_source_count": int(
            _safe_float(return_policy.get("production_source_count"), 0.0) or 0
        ),
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
    attribution = _trade_losing_exit_attribution(sample)
    exit_timing_label = _trade_exit_timing_label(sample, attribution=attribution)
    trade_profit_class = _trade_profit_class(sample, fee=fee, attribution=attribution)
    cost_basis_label = _trade_cost_basis_label(sample, fee=fee)
    pnl = _safe_float(sample.get("realized_pnl"), 0.0) or 0.0
    fee_dominated = bool(fee and abs(pnl) <= fee)
    notional = _trade_notional_usdt(sample)
    net_return_after_cost_pct = _trade_return_after_cost_pct(
        sample,
        pnl=pnl,
        notional=notional,
    )
    gross_pnl = _safe_float(sample.get("gross_pnl"), None)
    gross_return_on_notional_pct = (
        gross_pnl / notional * 100.0
        if gross_pnl is not None and notional is not None and notional > 0
        else None
    )
    fee_return_pct = (
        abs(fee) / notional * 100.0
        if fee is not None and notional is not None and notional > 0
        else None
    )
    funding_return_pct = (
        funding_fee / notional * 100.0
        if funding_fee is not None and notional is not None and notional > 0
        else None
    )
    slippage_cost = _first_float(
        sample.get("slippage_cost_usdt"),
        sample.get("execution_slippage_usdt"),
    )
    slippage_return_pct = (
        abs(slippage_cost) / notional * 100.0
        if slippage_cost is not None and notional is not None and notional > 0
        else None
    )
    return_on_margin_pct = (
        pnl / (notional / actual_leverage) * 100.0
        if notional is not None
        and notional > 0
        and actual_leverage is not None
        and actual_leverage > 0
        else None
    )
    return {
        "version": PROFIT_LEARNING_VERSION,
        "sample_kind": "trade",
        "training_supervision_ready": bool(
            not assessment.exclude_from_training
            and attribution != "okx_slippage_or_execution"
        ),
        "exit_attribution_supervision_ready": bool(
            not assessment.exclude_from_training
            and attribution not in {"okx_slippage_or_execution", "unknown_requires_review"}
        ),
        "trade_profit_class": trade_profit_class,
        "losing_exit_attribution": attribution,
        "exit_timing_label": exit_timing_label,
        "payoff_profile_label": _trade_payoff_label(
            sample,
            attribution=attribution,
            exit_timing_label=exit_timing_label,
            fee=fee,
        ),
        "cost_basis_label": cost_basis_label,
        "fee_dominated": fee_dominated,
        "realized_net_pnl_usdt": pnl,
        "gross_return_on_notional_pct": gross_return_on_notional_pct,
        "fee_return_pct": fee_return_pct,
        "slippage_return_pct": slippage_return_pct,
        "funding_return_pct": funding_return_pct,
        "net_return_after_cost_pct": net_return_after_cost_pct,
        "return_on_margin_pct": return_on_margin_pct,
        "return_after_cost_pct": net_return_after_cost_pct,
        "return_after_cost_pct_deprecated": True,
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
    if kind in {"shadow", "trade"}:
        annotated["profit_supervision"] = build_profit_supervision_contract(
            annotated,
            kind=kind,
        )
    annotated["training_sample_contract"] = _training_sample_contract(
        annotated,
        kind=kind,
    )
    return annotated


def annotate_samples(samples: list[dict[str, Any]], kind: SampleKind) -> list[dict[str, Any]]:
    prepared = [dict(sample) for sample in samples]
    if kind == "shadow":
        seen: dict[tuple[int, int, str], int] = {}
        for sample in prepared:
            features = _features(sample)
            label_contract = _safe_dict(features.get("training_label_contract"))
            identity = _safe_dict(label_contract.get("identity"))
            decision_id = int(
                _safe_float(sample.get("decision_id") or identity.get("decision_id"), 0.0)
                or 0
            )
            horizon = int(_safe_float(sample.get("horizon_minutes"), 0.0) or 0)
            version = _safe_str(
                sample.get("label_version") or label_contract.get("version")
            )
            if decision_id <= 0 or horizon <= 0 or not version:
                continue
            key = (decision_id, horizon, version)
            if key in seen:
                sample["is_duplicate"] = True
                sample["duplicate_of"] = seen[key]
                sample["duplicate_label_identity"] = {
                    "decision_id": decision_id,
                    "horizon_minutes": horizon,
                    "label_version": version,
                }
            else:
                seen[key] = int(_safe_float(sample.get("id"), 0.0) or len(seen) + 1)
    annotated = [annotate_sample(sample, kind) for sample in prepared]
    apply_correlation_group_weights(annotated, kind=kind)
    if kind in {"shadow", "trade"}:
        for sample in annotated:
            sample["profit_supervision"] = build_profit_supervision_contract(
                sample,
                kind=kind,
            )
            sample["training_sample_contract"] = _training_sample_contract(
                sample,
                kind=kind,
            )
    return annotated


def _training_sample_contract(sample: dict[str, Any], *, kind: SampleKind) -> dict[str, Any]:
    features = _features(sample)
    market_contract = compact_market_fact_contract(_shadow_market_fact_contract(features))
    lineage = {
        "sample_kind": kind,
        "source": _sample_source(sample, kind),
        "symbol": _safe_str(sample.get("symbol") or features.get("symbol")),
        "decision_id": sample.get("decision_id"),
        "horizon_minutes": sample.get("horizon_minutes"),
        "label_version": sample.get("label_version")
        or _safe_dict(features.get("training_label_contract")).get("version"),
        "feature_timestamp": sample.get("feature_timestamp") or features.get("timestamp"),
        "label_timestamp": sample.get("label_timestamp"),
    }
    label = {
        "long_return_pct": _safe_float(sample.get("long_return_pct"), None),
        "short_return_pct": _safe_float(sample.get("short_return_pct"), None),
        "realized_pnl": _safe_float(sample.get("realized_pnl"), None),
        "outcome": _safe_str(sample.get("outcome")),
        "best_action": _safe_str(sample.get("best_action")),
        "label_fingerprint": _safe_dict(
            features.get("training_label_contract")
        ).get("label_fingerprint"),
    }
    if kind == "sequence" and sample.get("sequence_format") == COMPACT_SEQUENCE_SERIES_FORMAT:
        sequence_facts = {
            "sequence_format": COMPACT_SEQUENCE_SERIES_FORMAT,
            "symbol": _safe_str(sample.get("symbol")),
            "timeframe": _safe_str(sample.get("timeframe")),
            "first_open_time": sample.get("first_open_time"),
            "last_open_time": sample.get("last_open_time"),
            "observation_count": int(
                _safe_float(sample.get("observation_count"), 0.0) or 0
            ),
            "label_name": _safe_str(sample.get("label_name")),
            "label_version": _safe_str(sample.get("label_version")),
            "close_sequence": sample.get("close_sequence") or [],
            "volume_sequence": sample.get("volume_sequence") or [],
        }
        label.update(
            {
                "label_name": sequence_facts["label_name"],
                "label_version": sequence_facts["label_version"],
                "observation_count": sequence_facts["observation_count"],
                "sequence_series_fingerprint": hashlib.sha256(
                    json.dumps(
                        sequence_facts,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    cost = {
        key: features.get(key)
        for key in (
            "bid",
            "ask",
            "spread_pct",
            "taker_fee_rate",
            "entry_fee_rate",
            "exit_fee_rate",
            "funding_rate",
            "funding_interval_minutes",
        )
        if key in features
    }
    fingerprint_payload = {
        "data_quality_version": DATA_QUALITY_VERSION,
        "profit_supervision_version": (
            PROFIT_SUPERVISION_VERSION if kind in {"shadow", "trade"} else None
        ),
        "profit_supervision_fingerprint": (
            _safe_dict(sample.get("profit_supervision")).get("contract_fingerprint")
            if kind in {"shadow", "trade"}
            else None
        ),
        "lineage": lineage,
        "market_fact_data_fingerprint": _safe_dict(
            market_contract.get("provenance")
        ).get("data_fingerprint"),
        "cost": cost,
        "label": label,
        "quality_status": sample.get("data_quality_status"),
        "quality_reasons": sample.get("quality_reasons") or [],
    }
    return {
        "version": DATA_QUALITY_VERSION,
        "immutable": True,
        "profit_supervision_version": (
            PROFIT_SUPERVISION_VERSION if kind in {"shadow", "trade"} else None
        ),
        "required_label_version": SHADOW_LABEL_VERSION if kind == "shadow" else None,
        "lineage": lineage,
        "market_fact_contract": market_contract,
        "shadow_label_contract": _safe_dict(features.get("training_label_contract")),
        "cost_facts": cost,
        "label": label,
        "profit_supervision": (
            _safe_dict(sample.get("profit_supervision"))
            if kind in {"shadow", "trade"}
            else {}
        ),
        "quality": {
            "status": sample.get("data_quality_status"),
            "score": sample.get("data_quality_score"),
            "weight": sample.get("sample_weight"),
            "reasons": sample.get("quality_reasons") or [],
        },
        "sample_fingerprint": hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _aggregate_market_fact_contract(
    samples_by_kind: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    shadow_samples = samples_by_kind.get("shadow", [])
    trainable = [
        sample
        for sample in shadow_samples
        if _safe_str(sample.get("data_quality_status")) != "excluded"
    ]
    contracts = [
        _shadow_market_fact_contract(_features(sample)) for sample in trainable
    ]
    reasons = [reason for contract in contracts for reason in market_fact_contract_reasons(contract)]
    clean = bool(contracts) and not reasons
    assertions = (
        {
            name: bool(
                contracts
                and all(
                    (
                        _safe_dict(contract.get("assertions")).get(name)
                        if _safe_dict(contract.get("assertions"))
                        else contract.get(name)
                    )
                    is True
                    for contract in contracts
                )
            )
            for name in (
                "native_instrument_identity_verified",
                "same_contract_price_path_verified",
                "executable_market_fact_verified",
            )
        }
    )
    fingerprints = sorted(
        _safe_str(
            _safe_dict(contract.get("provenance")).get("data_fingerprint")
            or contract.get("data_fingerprint")
        )
        for contract in contracts
        if _safe_str(
            _safe_dict(contract.get("provenance")).get("data_fingerprint")
            or contract.get("data_fingerprint")
        )
    )
    effective_sample_size = sum(
        float(_safe_float(sample.get("sample_weight"), 0.0) or 0.0) for sample in trainable
    )
    generated_at = datetime.now(UTC).isoformat()
    data_fingerprint = hashlib.sha256(
        json.dumps(fingerprints, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "version": MARKET_FACT_CONTRACT_VERSION,
        "status": "clean" if clean else "quarantined" if contracts else "missing",
        "violation_count": len(reasons),
        "violation_reasons": list(dict.fromkeys(reasons)),
        "assertions": assertions,
        "quarantined_raw_sample_count": sum(
            1
            for sample in shadow_samples
            if _safe_str(sample.get("data_quality_status")) == "excluded"
        ),
        "provenance": {
            "source": "immutable_shadow_training_view_market_fact_contracts",
            "observation_window": "current_immutable_shadow_training_view",
            "sample_count": len(contracts),
            "effective_sample_size": round(effective_sample_size, 8),
            "generated_at": generated_at,
            "strategy_version": MARKET_FACT_CONTRACT_VERSION,
            "fallback_reason": "" if clean else "clean_market_fact_training_view_missing",
            "data_fingerprint": data_fingerprint,
        },
    }


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
                return_pct = _safe_float(
                    labels.get("net_return_after_cost_pct"),
                    None,
                )
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
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    small_win_big_loss_ratio = avg_loss / max(avg_win, 1e-9) if win_count and loss_count else 0.0
    quality_warnings: list[str] = []
    if supervision_ready == 0:
        quality_warnings.append("no_supervision_ready_trade_samples")
    if trade_pnls and gross_profit <= gross_loss:
        quality_warnings.append("gross_loss_not_covered_by_profit")
    if win_count and loss_count and avg_loss > avg_win:
        quality_warnings.append("avg_loss_larger_than_avg_win")
    if gross_profit > 0 and gross_loss <= 0:
        quality_warnings.append("profit_factor_undefined_without_losses")
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
            "profit_factor": (
                round(profit_factor, 6) if profit_factor is not None else None
            ),
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


def _training_label_consistency(samples: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    ready_count = 0
    pnl_total = 0.0
    return_total = 0.0
    return_count = 0
    positive_pnl_count = 0
    negative_pnl_count = 0
    for sample in samples:
        labels = _safe_dict(sample.get("profit_learning_labels"))
        if not labels.get("training_supervision_ready"):
            continue
        ready_count += 1
        pnl = _safe_float(labels.get("realized_net_pnl_usdt"), None)
        return_pct = _safe_float(
            labels.get("net_return_after_cost_pct"),
            None,
        )
        sample_key = _safe_str(sample.get("lifecycle_key") or sample.get("position_id"))
        if pnl is None:
            errors.append({"sample_key": sample_key, "reason": "missing_realized_net_pnl"})
            continue
        pnl_total += pnl
        positive_pnl_count += int(pnl > 0)
        negative_pnl_count += int(pnl < 0)
        if return_pct is None:
            errors.append(
                {
                    "sample_key": sample_key,
                    "reason": "missing_net_return_after_cost_pct",
                }
            )
            continue
        return_total += return_pct
        return_count += 1
        if (pnl > 0 and return_pct <= 0) or (pnl < 0 and return_pct >= 0):
            errors.append(
                {
                    "sample_key": sample_key,
                    "reason": "pnl_return_sign_mismatch",
                    "pnl": round(pnl, 8),
                    "return_after_cost_pct": round(return_pct, 8),
                }
            )
        components = _safe_float(sample.get("settlement_components_total"), None)
        if components is not None and abs(pnl - components) > max(1e-6, abs(pnl) * 1e-5):
            errors.append(
                {
                    "sample_key": sample_key,
                    "reason": "settlement_component_sum_mismatch",
                    "pnl": round(pnl, 8),
                    "component_sum": round(components, 8),
                }
            )
    avg_return = return_total / max(return_count, 1)
    if (
        pnl_total > 0
        and positive_pnl_count > negative_pnl_count
        and return_count > 0
        and avg_return <= 0
    ):
        errors.append(
            {
                "sample_key": "aggregate",
                "reason": "positive_net_pnl_but_negative_average_return",
                "net_pnl": round(pnl_total, 8),
                "avg_return_after_cost_pct": round(avg_return, 8),
            }
        )
    return {
        "status": "blocked" if errors else "consistent",
        "promotion_blocked": bool(errors),
        "supervision_ready_count": ready_count,
        "checked_return_count": return_count,
        "net_realized_pnl_usdt": round(pnl_total, 8),
        "avg_return_after_cost_pct": round(avg_return, 8),
        "error_count": len(errors),
        "errors": errors[:100],
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
    realized_returns = [float(value) for value in row.get("realized_returns") or []]
    tail_policy = empirical_policy_value(
        "shadow_return_lower_hinge",
        realized_returns,
        selector="lower_hinge",
        observation_window="completed_shadow_fee_after_returns",
    )
    row["direction_hit_rate"] = round(hit_rate, 4)
    row["avg_shadow_expected_return_pct"] = round(avg_expected, 6)
    row["avg_expected_return_pct"] = round(avg_expected, 6)
    row["avg_realized_return_pct"] = round(avg_realized, 6)
    row["false_signal_count"] = int(row.get("false_signal_count") or 0)
    row["return_lower_hinge_pct"] = tail_policy.value
    row["return_distribution_provenance"] = tail_policy.to_dict()
    row["worst_samples"] = list(row.get("worst_samples") or [])[:MAX_WORST_SAMPLE_COUNT]
    row["legacy_mixed_shadow_count"] = int(row.get("legacy_mixed_shadow_count") or 0)
    row["legacy_quarantined_count"] = int(row.get("legacy_quarantined_count") or 0)
    row["observation_policy"] = {
        "observation_only": True,
        "promotion_authority": False,
        "optimization_target": "fee_after_realized_return",
        "actual_inference_count": int(row.get("actual_inference_count") or 0),
        "direction_count": direction_count,
        "direction_hit_rate": round(hit_rate, 4),
        "avg_realized_return_pct": round(avg_realized, 6),
        "worst_realized_return_pct": row.get("worst_realized_return_pct"),
        "return_lower_hinge_pct": tail_policy.value,
        "legacy_mixed_shadow_count": row["legacy_mixed_shadow_count"],
        "legacy_quarantined_count": row["legacy_quarantined_count"],
    }
    row.pop("realized_return_sum_pct", None)
    row.pop("shadow_expected_return_sum", None)
    row.pop("realized_returns", None)
    row.pop("tail_loss_count", None)
    row.pop("tail_loss_symbols", None)
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
                        "realized_returns": [],
                        "false_signal_count": 0,
                        "worst_realized_return_pct": None,
                        "best_realized_return_pct": None,
                        "tail_loss_count": 0,
                        "tail_loss_symbols": Counter(),
                        "worst_samples": [],
                        "specialist_inference_count": 0,
                        "legacy_mixed_shadow_count": 0,
                        "legacy_quarantined_count": 0,
                    },
                )
                row["sample_count"] += 1
                if bool(candidate.get("specialist_inference_active")):
                    row["specialist_inference_count"] += 1
                legacy_mixed_shadow = bool(candidate.get("legacy_mixed_shadow"))
                if legacy_mixed_shadow:
                    row["legacy_mixed_shadow_count"] += 1
                sequence_length = int(candidate.get("sequence_length") or 0)
                if legacy_mixed_shadow:
                    row["legacy_quarantined_count"] += 1
                    continue
                row["actual_inference_count"] += 1
                direction = _safe_str(candidate.get("direction")).lower()
                if direction in {"long", "short"}:
                    actual_return = _actual_return_for_side(sample, direction)
                    if actual_return is not None:
                        row["direction_count"] += 1
                        row["realized_return_sum_pct"] += actual_return
                        row["realized_returns"].append(actual_return)
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


def _weighted_mean(
    samples: list[dict[str, Any]],
    value_key: str,
) -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    for sample in samples:
        value = _safe_float(sample.get(value_key), None)
        weight = _safe_float(sample.get("sample_weight"), 0.0) or 0.0
        if value is None or weight <= 0:
            continue
        weighted_sum += value * weight
        weight_sum += weight
    return weighted_sum / weight_sum if weight_sum > 0 else None


def _shadow_training_view_diagnostics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    trainable = [sample for sample in samples if not sample.get("exclude_from_training")]
    projected: list[dict[str, Any]] = []
    for sample in trainable:
        long_return = _safe_float(sample.get("long_return_pct"), None)
        short_return = _safe_float(sample.get("short_return_pct"), None)
        if long_return is None or short_return is None:
            continue
        projected.append(
            {
                **sample,
                "best_return_pct": max(long_return, short_return),
            }
        )

    overall = {
        "long_return_pct": _weighted_mean(projected, "long_return_pct"),
        "short_return_pct": _weighted_mean(projected, "short_return_pct"),
        "best_return_pct": _weighted_mean(projected, "best_return_pct"),
    }
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for sample in projected:
        symbol = _safe_str(sample.get("symbol") or _features(sample).get("symbol"))
        by_symbol.setdefault(symbol or "unknown", []).append(sample)

    leave_one_symbol_out: list[dict[str, Any]] = []
    for symbol, symbol_samples in by_symbol.items():
        without_symbol = [
            sample
            for sample in projected
            if _safe_str(sample.get("symbol") or _features(sample).get("symbol"))
            != symbol
        ]
        without = {
            "long_return_pct": _weighted_mean(without_symbol, "long_return_pct"),
            "short_return_pct": _weighted_mean(without_symbol, "short_return_pct"),
            "best_return_pct": _weighted_mean(without_symbol, "best_return_pct"),
        }
        best_delta = (
            float(without["best_return_pct"]) - float(overall["best_return_pct"])
            if without["best_return_pct"] is not None
            and overall["best_return_pct"] is not None
            else None
        )
        leave_one_symbol_out.append(
            {
                "symbol": symbol,
                "sample_count": len(symbol_samples),
                "effective_sample_size": round(
                    sum(
                        float(_safe_float(sample.get("sample_weight"), 0.0) or 0.0)
                        for sample in symbol_samples
                    ),
                    8,
                ),
                "sample_share": round(len(symbol_samples) / max(len(projected), 1), 8),
                "symbol_best_return_pct": _weighted_mean(
                    symbol_samples,
                    "best_return_pct",
                ),
                "without_symbol": without,
                "best_return_mean_delta_pct": best_delta,
                "absolute_influence_pct": abs(best_delta) if best_delta is not None else None,
            }
        )
    leave_one_symbol_out.sort(
        key=lambda item: float(item.get("absolute_influence_pct") or 0.0),
        reverse=True,
    )

    time_buckets: dict[str, list[dict[str, Any]]] = {}
    for sample in projected:
        timestamp = _parse_datetime(
            sample.get("label_timestamp")
            or _safe_dict(
                _safe_dict(_features(sample).get("training_label_contract")).get(
                    "labels"
                )
            ).get("label_timestamp")
            or _safe_dict(_features(sample).get("training_label_contract")).get(
                "label_timestamp"
            )
        )
        bucket = timestamp.date().isoformat() if timestamp else "unknown"
        time_buckets.setdefault(bucket, []).append(sample)
    time_influence = [
        {
            "date": date,
            "sample_count": len(bucket_samples),
            "effective_sample_size": round(
                sum(
                    float(_safe_float(sample.get("sample_weight"), 0.0) or 0.0)
                    for sample in bucket_samples
                ),
                8,
            ),
            "long_return_pct": _weighted_mean(bucket_samples, "long_return_pct"),
            "short_return_pct": _weighted_mean(bucket_samples, "short_return_pct"),
            "best_return_pct": _weighted_mean(bucket_samples, "best_return_pct"),
        }
        for date, bucket_samples in sorted(time_buckets.items())
    ]
    fingerprints = sorted(
        _safe_str(
            _safe_dict(sample.get("training_sample_contract")).get(
                "sample_fingerprint"
            )
        )
        for sample in samples
        if _safe_str(
            _safe_dict(sample.get("training_sample_contract")).get(
                "sample_fingerprint"
            )
        )
    )
    return {
        "raw_sample_count": len(samples),
        "trainable_sample_count": len(trainable),
        "quarantined_sample_count": sum(
            1 for sample in samples if sample.get("exclude_from_training")
        ),
        "effective_sample_size": round(
            sum(
                float(_safe_float(sample.get("sample_weight"), 0.0) or 0.0)
                for sample in trainable
            ),
            8,
        ),
        "overall_return_distribution": overall,
        "leave_one_symbol_out": leave_one_symbol_out,
        "max_single_symbol_influence": (
            leave_one_symbol_out[0] if leave_one_symbol_out else None
        ),
        "time_influence": time_influence,
        "provenance": {
            "source": "immutable_clean_shadow_training_view",
            "observation_window": "all_supplied_shadow_samples",
            "sample_count": len(samples),
            "effective_sample_size": round(
                sum(
                    float(_safe_float(sample.get("sample_weight"), 0.0) or 0.0)
                    for sample in trainable
                ),
                8,
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": DATA_QUALITY_VERSION,
            "fallback_reason": "" if trainable else "clean_training_view_empty",
            "data_fingerprint": hashlib.sha256(
                json.dumps(
                    fingerprints,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
    }


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
        "market_fact_contract": _aggregate_market_fact_contract(samples_by_kind),
        "profit_supervision": profit_supervision_report(
            samples_by_kind.get("shadow", []),
            samples_by_kind.get("trade", []),
        ),
        "policy": asdict(DEFAULT_TRADING_PARAMS.training_data_quality),
        "by_kind": by_kind,
        "profit_learning_summary": {
            kind: _profit_learning_report(samples) for kind, samples in samples_by_kind.items()
        },
        "specialist_shadow_models": _shadow_model_report(samples_by_kind.get("shadow", [])),
        "training_view_diagnostics": _shadow_training_view_diagnostics(
            samples_by_kind.get("shadow", [])
        ),
        "training_label_consistency": _training_label_consistency(
            samples_by_kind.get("trade", [])
        ),
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


def artifact_bound_governance_report(
    quality: dict[str, Any],
    *,
    persist_artifact: bool,
) -> dict[str, Any]:
    """Bind a newly persisted artifact to the exact clean-view fingerprint."""

    report = governance_report(quality)
    if not persist_artifact:
        return report
    return governance_report(
        quality,
        artifact_quality_fingerprint=str(report.get("quality_fingerprint") or ""),
    )


def _contamination_risk(
    *,
    has_contamination: bool,
    excluded_ratio: float,
    blocked_reason_ratio: float,
    effective_weight_ratio: float,
) -> str:
    if not has_contamination:
        return "low"
    del excluded_ratio, blocked_reason_ratio, effective_weight_ratio
    return "high"


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
