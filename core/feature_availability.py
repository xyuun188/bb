"""Observed input availability sealed before any model reads a feature vector."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

FEATURE_AVAILABILITY_VERSION = "2026-10-22.observed-feature-availability.v1"
COMPONENT_AVAILABILITY_VERSION = "2026-10-22.observed-component.v1"
# These are model inputs, not subsequently attached labels or settlement facts.
FROZEN_FEATURE_FIELDS = (
    "symbol", "timestamp", "current_price", "close", "bid", "ask",
    "high_24h", "low_24h", "volume_24h", "volume_24h_base", "volume_24h_contracts",
    "volume_24h_quote", "notional_24h_usdt", "volume_24h_source", "volume",
    "close_sequence", "volume_sequence", "sequence_timeframe", "sequence_length",
    "sequence_quality_warning", "change_24h_pct", "spread_pct", "price_source",
    "indicator_close_price", "indicator_price_gap_pct", "price_reconciliation_warning",
    "market_data_observation_only", "market_data_observation_reasons",
    "feature_refresh_fallback_used", "feature_refresh_fallback_reason",
    "market_fact", "mark_price", "index_price", "mark_price_fact", "index_price_fact",
    "orderbook_fact", "indicator_snapshot_available", "indicator_snapshot_stale",
    "indicator_snapshot_refresh_in_background", "indicator_snapshot_quality",
    "indicator_snapshot_reason", "technical_availability",
    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_diff", "stoch_k",
    "ema_12", "ema_26", "adx_14", "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "bb_pct", "atr_14", "volume_ratio", "returns_1", "returns_5", "returns_20",
    "volatility_20", "price_vs_sma20", "price_vs_sma50",
    "technical_indicator_timeframe", "short_returns_timeframe", "volume_ratio_timeframe",
    "entry_activity_volume_ratio", "entry_activity_volume_timeframe",
    "abnormal_wick_count_72h", "abnormal_wick_max_pct", "abnormal_wick_recent_hours",
    "funding_rate", "funding_data_available", "funding_interval_minutes", "funding_time",
    "next_funding_time", "funding_rate_observed_at", "funding_rate_received_at",
    "taker_fee_rate", "entry_fee_rate", "exit_fee_rate", "round_trip_fee_pct",
    "fee_rate_source", "fee_rate_observed_at", "account_fee_received_at",
    "account_fee_source_observed_at", "fee_policy_provenance",
    "open_interest_contracts", "open_interest_value", "open_interest_observed_at",
    "open_interest_received_at", "orderbook_bid_depth",
    "orderbook_ask_depth", "orderbook_imbalance", "liquidation_risk_score",
    "derivatives_snapshot_stale", "derivatives_refresh_in_background",
    "sector_relative_strength", "news_sentiment_avg", "social_sentiment_avg",
    "social_mention_count", "news_article_count", "headline_count",
    "sentiment_data_available", "direct_sentiment_data_available",
    "direct_news_item_count", "market_news_item_count", "news_sources",
    "sentiment_availability", "model_text_fingerprint", "exchange_inflow", "whale_txn_count",
)


def availability_timestamp(value: Any) -> datetime | None:
    """Parse observed clocks without granting a timezone to ambiguous values."""

    if isinstance(value, datetime):
        result = value
    elif isinstance(value, (int, float)) or (
        isinstance(value, str) and value.replace(".", "", 1).isdigit()
    ):
        try:
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                return None
            result = datetime.fromtimestamp(
                number / 1000.0 if number > 10_000_000_000 else number, UTC
            )
        except (ValueError, OverflowError, OSError):
            return None
    elif isinstance(value, str) and value:
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                result = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
    else:
        return None
    return result.astimezone(UTC) if result.tzinfo is not None else None


def availability_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                   default=str, allow_nan=False).encode("utf-8")
    ).hexdigest()


class ObservedKlines(list):
    """An actual response's rows and receipt travel together in memory."""

    def __init__(self, rows: list, *, received_at: Any = None) -> None:
        super().__init__(rows)
        self.received_at = received_at


def seal_feature_availability(target: Any, *, frozen_at: datetime | None = None) -> Any:
    """Seal once before model consumption; cache reads never refresh this seal."""

    snapshot = (
        target if isinstance(target, dict)
        else getattr(target, "to_dict", lambda: deepcopy(vars(target)))()
    )
    if snapshot.get("feature_availability_contract"):
        return target
    boundary = frozen_at or datetime.now(UTC)
    boundary = availability_timestamp(boundary)
    if boundary is None:
        raise ValueError("feature freeze requires an aware timestamp")
    contract: dict[str, Any] = {
        "version": FEATURE_AVAILABILITY_VERSION,
        "frozen_at": boundary.isoformat(),
    }
    try:
        contract["input_fingerprint"] = availability_fingerprint(
            {key: snapshot[key] for key in FROZEN_FEATURE_FIELDS if key in snapshot}
        )
    except (TypeError, ValueError):
        contract["seal_error"] = "nonfinite_or_invalid_model_input"
    # Import lazily: the shared compactor also imports the frozen field contract.
    from core.training_feature_snapshot import (
        FROZEN_TRAINING_FEATURE_MAX_BYTES,
        frozen_feature_payload_size,
    )

    payload = {
        **snapshot,
        "feature_snapshot_frozen_at": boundary.isoformat(),
        "feature_availability_contract": contract,
    }
    if frozen_feature_payload_size(payload) > FROZEN_TRAINING_FEATURE_MAX_BYTES:
        contract["seal_error"] = "frozen_input_budget_exceeded"
    if isinstance(target, dict):
        target["feature_snapshot_frozen_at"] = boundary.isoformat()
        target["feature_availability_contract"] = contract
    else:
        target.feature_snapshot_frozen_at = boundary.isoformat()
        target.feature_availability_contract = contract
    return target


def _clock_reasons(
    name: str, *, received: Any, available: Any, source: Any, frozen: datetime,
    source_required: bool = False,
) -> list[str]:
    receipt = availability_timestamp(received)
    ready = availability_timestamp(available)
    event = availability_timestamp(source) if source not in (None, "") else None
    reasons = []
    if receipt is None:
        reasons.append(f"{name}:receipt_missing_or_ambiguous")
    if ready is None:
        reasons.append(f"{name}:availability_missing_or_ambiguous")
    if source not in (None, "") and event is None:
        reasons.append(f"{name}:source_clock_ambiguous")
    elif source_required and event is None:
        reasons.append(f"{name}:source_clock_missing")
    if receipt and ready and receipt > ready:
        reasons.append(f"{name}:receipt_after_availability")
    if event and receipt and event > receipt:
        reasons.append(f"{name}:source_after_receipt")
    if ready and ready > frozen:
        reasons.append(f"{name}:available_after_freeze")
    if receipt and receipt > frozen:
        reasons.append(f"{name}:received_after_freeze")
    return reasons


def feature_availability_reasons(features: dict[str, Any], *, decision_at: Any = None) -> list[str]:
    """Fail closed on unobserved or rewritten inputs, including historical rows."""

    contract = features.get("feature_availability_contract")
    if not isinstance(contract, dict):
        return ["feature_availability_contract_missing"]
    if contract.get("version") != FEATURE_AVAILABILITY_VERSION:
        return ["feature_availability_contract_stale"]
    if contract.get("seal_error"):
        return [f"feature_availability_seal_failed:{contract['seal_error']}"]
    if features.get("training_feature_compaction_error"):
        return ["feature_availability_compaction_failed"]
    frozen = availability_timestamp(features.get("feature_snapshot_frozen_at"))
    if frozen is None or frozen != availability_timestamp(contract.get("frozen_at")):
        return ["feature_availability_boundary_missing_or_conflicting"]
    completed = availability_timestamp(
        decision_at if decision_at is not None else features.get("original_decision_timestamp")
    )
    reasons: list[str] = []
    if completed is None:
        reasons.append("original_decision_timestamp_missing_or_ambiguous")
    elif frozen > completed:
        reasons.append("feature_freeze_after_original_decision")
    try:
        fingerprint = availability_fingerprint(
            {key: features[key] for key in FROZEN_FEATURE_FIELDS if key in features}
        )
    except (TypeError, ValueError):
        fingerprint = None
    if not fingerprint or contract.get("input_fingerprint") != fingerprint:
        reasons.append("feature_inputs_changed_after_freeze")

    technical = features.get("technical_availability")
    if not isinstance(technical, dict) or technical.get("version") != COMPONENT_AVAILABILITY_VERSION:
        reasons.append("technical:availability_contract_missing_or_stale")
    else:
        if technical.get("source_delivery_observed") is not True:
            reasons.append("technical:source_delivery_unobserved")
        for timeframe in str(technical.get("timeframes") or "").split(","):
            if not timeframe:
                continue
            reasons.extend(_clock_reasons(
                f"technical:{timeframe}",
                received=technical.get(f"{timeframe}_received_at"),
                available=technical.get("available_at"),
                source=technical.get(f"{timeframe}_closed_at"), frozen=frozen,
                source_required=True,
            ))
        if not technical.get("timeframes") or not technical.get("input_fingerprint"):
            reasons.append("technical:input_lineage_missing")

    sentiment = features.get("sentiment_availability")
    if not isinstance(sentiment, dict) or sentiment.get("version") != COMPONENT_AVAILABILITY_VERSION:
        reasons.append("sentiment:availability_contract_missing_or_stale")
    else:
        ready = availability_timestamp(sentiment.get("available_at"))
        if ready is None or ready > frozen:
            reasons.append("sentiment:availability_missing_or_after_freeze")
        for name in ("news", "social"):
            count = sentiment.get(f"{name}_input_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                reasons.append(f"{name}:input_count_invalid")
                continue
            if sentiment.get(f"{name}_clock_errors") != 0:
                reasons.append(f"{name}:input_clocks_missing_or_conflicting")
            if count > 0:
                reasons.extend(_clock_reasons(
                    name, received=sentiment.get(f"{name}_received_at"),
                    available=sentiment.get("available_at"), source=None, frozen=frozen,
                ))
                if not sentiment.get(f"{name}_input_fingerprint"):
                    reasons.append(f"{name}:input_lineage_missing")

    for name, key in (("ticker", "market_fact"), ("orderbook", "orderbook_fact"),
                      ("mark_price", "mark_price_fact"),
                      ("index_price", "index_price_fact")):
        fact = features.get(key)
        if not isinstance(fact, dict) or not fact:
            reasons.append(f"{name}:fact_missing")
            continue
        reasons.extend(_clock_reasons(
            name, received=fact.get("received_at"), available=fact.get("received_at"),
            source=fact.get("timestamp") or fact.get("source_timestamp_ms"), frozen=frozen,
            source_required=True,
        ))
    for name, receipt, source in (
        ("funding", "funding_rate_received_at", "funding_rate_observed_at"),
        ("account_fee", "account_fee_received_at", "account_fee_source_observed_at"),
    ):
        reasons.extend(_clock_reasons(
            name, received=features.get(receipt), available=features.get(receipt),
            source=features.get(source), frozen=frozen,
        ))
    if features.get("open_interest_contracts") or features.get("open_interest_value"):
        reasons.extend(_clock_reasons(
            "open_interest", received=features.get("open_interest_received_at"),
            available=features.get("open_interest_received_at"),
            source=features.get("open_interest_observed_at"), frozen=frozen,
        ))
    return list(dict.fromkeys(reasons))


def sentiment_input_availability(
    news: list[dict], social: list[dict], *, available_at: datetime,
) -> dict[str, Any]:
    """Bounded lineage for every score/headline contributor, without storing text."""

    evidence: dict[str, Any] = {
        "version": COMPONENT_AVAILABILITY_VERSION, "available_at": available_at.isoformat(),
    }
    for name, items, publication_key in (
        ("news", news, "published_at"), ("social", social, "posted_at"),
    ):
        errors = 0
        receipts = []
        for item in items:
            receipt = availability_timestamp(item.get("received_at"))
            publication = availability_timestamp(item.get(publication_key))
            if receipt is None or receipt > available_at:
                errors += 1
            if item.get(publication_key) and (
                publication is None or (receipt and publication > receipt)
            ):
                errors += 1
            if receipt:
                receipts.append(receipt)
        evidence[f"{name}_input_count"] = len(items)
        evidence[f"{name}_clock_errors"] = errors
        evidence[f"{name}_received_at"] = max(receipts).isoformat() if receipts else None
        evidence[f"{name}_input_fingerprint"] = availability_fingerprint(items)
    return deepcopy(evidence)
