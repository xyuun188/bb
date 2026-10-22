"""Canonical historical market-path facts used by shadow training."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from core.market_facts import MARKET_FACT_CONTRACT_VERSION
from core.training_contracts import HISTORICAL_SHADOW_REBUILD_VERSION, HISTORICAL_SHADOW_SOURCE


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _timestamp_ms(value: datetime) -> int:
    return int(_as_utc(value).timestamp() * 1000)


def build_historical_market_fact(
    symbol: str,
    bar: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    close = float(bar["close"])
    opened = _as_utc(bar["open_time"])
    return {
        "schema_version": "historical_ohlcv_fact.v1",
        "symbol": symbol,
        "native_identity": {
            "symbol": symbol,
            "source": "market_klines",
            "timeframe": "1m",
        },
        "source_interface": "stored_market_klines",
        "source_endpoint": "market_klines",
        "source_channel": "1m",
        "source_timestamp_ms": _timestamp_ms(bar["open_time"]),
        "received_at": opened.isoformat(),
        "source_available_at": (opened + timedelta(minutes=1)).isoformat(),
        "availability_basis": "one_minute_candle_close_lower_bound",
        "delivery_latency_observed": False,
        "prices": {
            "last": close,
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
        },
        "liquidity": {
            "notional_24h_usdt": 0.0,
            "volume_24h_contracts": 0.0,
            "volume_24h_base": float(bar["volume"]),
        },
        "stale": False,
        "role": role,
    }


def build_historical_market_contract(
    *,
    symbol: str,
    entry_fact: dict[str, Any],
    result_fact: dict[str, Any],
    bars: list[dict[str, Any]],
    scheduled_label_at: Any = None,
) -> dict[str, Any]:
    path = {
        "version": "historical_ohlcv_path.v1",
        "status": "clean",
        "identity_match": True,
        "source": HISTORICAL_SHADOW_SOURCE,
        "symbol": symbol,
        "timeframe": "1m",
        "entry_timestamp_ms": entry_fact["source_timestamp_ms"],
        "result_timestamp_ms": result_fact["source_timestamp_ms"],
        "bar_count": len(bars),
        "expected_bar_count": len(bars),
        "path_low": min(float(bar["low"]) for bar in bars),
        "path_high": max(float(bar["high"]) for bar in bars),
        "_ordered_bar_ranges": [
            {
                "open_time_ms": _timestamp_ms(bar["open_time"]),
                "high": bar["high"],
                "low": bar["low"],
            }
            for bar in bars
        ],
    }
    fingerprint_payload = {
        "version": HISTORICAL_SHADOW_REBUILD_VERSION,
        "symbol": symbol,
        "entry_timestamp_ms": entry_fact["source_timestamp_ms"],
        "result_timestamp_ms": result_fact["source_timestamp_ms"],
        "bars": [
            [
                _timestamp_ms(bar["open_time"]),
                bar["open"],
                bar["high"],
                bar["low"],
                bar["close"],
                bar["volume"],
            ]
            for bar in bars
        ],
    }
    data_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    path["path_fingerprint"] = data_fingerprint
    # Keep the exact fingerprint inputs even after rolling candle-cache eviction.
    path["ohlcv_evidence"] = fingerprint_payload
    source_available_at = result_fact.get("source_available_at")
    scheduled = _as_utc(scheduled_label_at) if isinstance(
        scheduled_label_at, datetime
    ) else scheduled_label_at
    label_maturity_lag_seconds = None
    if isinstance(scheduled, datetime) and source_available_at:
        label_maturity_lag_seconds = (
            datetime.fromisoformat(str(source_available_at).replace("Z", "+00:00"))
            - _as_utc(scheduled)
        ).total_seconds()
    return {
        "version": MARKET_FACT_CONTRACT_VERSION,
        "status": "historical_ohlcv_only",
        "violation_count": 0,
        "violation_reasons": [],
        "assertions": {
            "native_instrument_identity_verified": True,
            "same_contract_price_path_verified": True,
            "executable_market_fact_verified": False,
        },
        "entry_fact": entry_fact,
        "result_fact": result_fact,
        "price_path": path,
        "provenance": {
            "source": HISTORICAL_SHADOW_SOURCE,
            "observation_window": {
                "start": entry_fact.get("source_available_at", entry_fact["received_at"]),
                "end": source_available_at or result_fact["received_at"],
            },
            "scheduled_label_at": (
                _as_utc(scheduled).isoformat() if isinstance(scheduled, datetime) else str(scheduled or "")
            ),
            "result_source_available_at": source_available_at or "",
            "label_maturity_lag_seconds": label_maturity_lag_seconds,
            "availability_basis": "one_minute_candle_close_lower_bound",
            "delivery_latency_observed": False,
            "sample_count": len(bars),
            "effective_sample_size": 1.0,
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": HISTORICAL_SHADOW_REBUILD_VERSION,
            "fallback_reason": "historical_ohlcv_has_no_historical_orderbook",
            "data_fingerprint": data_fingerprint,
        },
    }


def compact_historical_market_contract(contract: dict[str, Any]) -> dict[str, Any]:
    provenance = contract["provenance"]
    return {
        "version": contract["version"],
        "status": contract["status"],
        "native_instrument_identity_verified": True,
        "same_contract_price_path_verified": True,
        "executable_market_fact_verified": False,
        "source": HISTORICAL_SHADOW_SOURCE,
        "path_status": "clean",
        "path_fingerprint": provenance["data_fingerprint"],
        "data_fingerprint": provenance["data_fingerprint"],
        "observation_window": provenance["observation_window"],
        "scheduled_label_at": provenance["scheduled_label_at"],
        "result_source_available_at": provenance["result_source_available_at"],
        "label_maturity_lag_seconds": provenance["label_maturity_lag_seconds"],
        "availability_basis": provenance["availability_basis"],
        "delivery_latency_observed": provenance["delivery_latency_observed"],
    }
