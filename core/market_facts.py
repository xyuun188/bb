"""Canonical OKX-native market facts and immutable shadow path contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from core.symbols import normalize_trading_symbol, okx_inst_id_from_symbol, symbol_from_okx_inst_id

MARKET_FACT_SCHEMA_VERSION = "2026-07-14.okx-native-market-fact.v1"
MARKET_FACT_CONTRACT_VERSION = "2026-07-14.native-market-fact.v1"
MARKET_SOURCE_CONSISTENCY_VERSION = "2026-07-14.okx-source-consistency.v1"


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _iso(value: Any) -> str:
    parsed = _datetime(value)
    return parsed.isoformat() if parsed else ""


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) or _text(value).replace(".", "", 1).isdigit():
        number = _safe_float(value, None)
        if number is None or number <= 0:
            return None
        seconds = number / 1000.0 if number > 10_000_000_000 else number
        try:
            parsed = datetime.fromtimestamp(seconds, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    elif value:
        try:
            parsed = datetime.fromisoformat(_text(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _timestamp_ms(value: Any) -> int | None:
    parsed = _datetime(value)
    return int(parsed.timestamp() * 1000.0) if parsed else None


def _fingerprint(payload: Any, *, length: int = 64) -> str:
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return digest[:length]


def normalize_okx_contract_spec(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = _dict(spec)
    inst_id = _text(raw.get("instId") or raw.get("inst_id")).upper()
    normalized = {
        "inst_id": inst_id,
        "inst_type": _text(raw.get("instType") or raw.get("inst_type") or "SWAP").upper(),
        "uly": _text(raw.get("uly")).upper(),
        "inst_family": _text(raw.get("instFamily") or raw.get("inst_family")).upper(),
        "contract_type": _text(raw.get("ctType") or raw.get("contract_type")).lower(),
        "contract_value": _text(raw.get("ctVal") or raw.get("contract_value")),
        "contract_multiplier": _text(
            raw.get("ctMult") or raw.get("contract_multiplier") or "1"
        ),
        "contract_value_currency": _text(
            raw.get("ctValCcy") or raw.get("contract_value_currency")
        ).upper(),
        "settle_currency": _text(raw.get("settleCcy") or raw.get("settle_currency")).upper(),
        "lot_size": _text(raw.get("lotSz") or raw.get("lot_size")),
        "minimum_size": _text(raw.get("minSz") or raw.get("minimum_size")),
        "tick_size": _text(raw.get("tickSz") or raw.get("tick_size")),
        "state": _text(raw.get("state")).lower(),
        "source": _text(raw.get("source") or "okx_public_instruments"),
    }
    normalized["spec_version"] = f"sha256:{_fingerprint(normalized)}"
    return normalized


def _native_identity(
    symbol: str,
    snapshot: Mapping[str, Any],
    contract_spec: Mapping[str, Any] | None,
) -> dict[str, Any]:
    info = _dict(snapshot.get("info"))
    spec = normalize_okx_contract_spec(contract_spec)
    inst_id = _text(
        snapshot.get("inst_id")
        or snapshot.get("instId")
        or info.get("instId")
        or spec.get("inst_id")
        or okx_inst_id_from_symbol(symbol)
    ).upper()
    if not spec.get("inst_id") and inst_id:
        spec = normalize_okx_contract_spec({**_dict(contract_spec), "instId": inst_id})
    return {
        "inst_id": inst_id,
        "inst_type": _text(
            snapshot.get("inst_type")
            or snapshot.get("instType")
            or info.get("instType")
            or spec.get("inst_type")
        ).upper(),
        "uly": _text(snapshot.get("uly") or info.get("uly") or spec.get("uly")).upper(),
        "contract_spec": spec,
        "contract_spec_version": spec.get("spec_version"),
    }


def market_fact_reasons(fact: Mapping[str, Any] | None) -> list[str]:
    value = _dict(fact)
    identity = _dict(value.get("native_identity"))
    spec = _dict(identity.get("contract_spec"))
    prices = _dict(value.get("prices"))
    liquidity = _dict(value.get("liquidity"))
    reasons: list[str] = []
    required_identity = {
        "inst_id": identity.get("inst_id"),
        "inst_type": identity.get("inst_type"),
        "uly": identity.get("uly"),
        "contract_spec_version": identity.get("contract_spec_version"),
        "contract_value": spec.get("contract_value"),
        "contract_multiplier": spec.get("contract_multiplier"),
    }
    reasons.extend(
        f"native_identity_missing:{key}" for key, item in required_identity.items() if not item
    )
    if value.get("schema_version") != MARKET_FACT_SCHEMA_VERSION:
        reasons.append("market_fact_schema_missing_or_stale")
    if not value.get("source_timestamp_ms"):
        reasons.append("source_timestamp_missing")
    if not value.get("source_endpoint") or not value.get("source_channel"):
        reasons.append("source_lineage_incomplete")
    last = _safe_float(prices.get("last"), 0.0) or 0.0
    bid = _safe_float(prices.get("bid"), 0.0) or 0.0
    ask = _safe_float(prices.get("ask"), 0.0) or 0.0
    if last <= 0:
        reasons.append("last_price_non_positive")
    if bid <= 0:
        reasons.append("executable_bid_missing")
    if ask <= 0:
        reasons.append("executable_ask_missing")
    if bid > 0 and ask > 0 and bid > ask:
        reasons.append("crossed_bid_ask")
    if (_safe_float(liquidity.get("notional_24h_usdt"), 0.0) or 0.0) <= 0:
        reasons.append("zero_notional_turnover")
    if (_safe_float(liquidity.get("bid_depth_usdt"), 0.0) or 0.0) <= 0:
        reasons.append("empty_bid_depth")
    if (_safe_float(liquidity.get("ask_depth_usdt"), 0.0) or 0.0) <= 0:
        reasons.append("empty_ask_depth")
    expected_symbol = normalize_trading_symbol(symbol_from_okx_inst_id(identity.get("inst_id")))
    symbol = normalize_trading_symbol(value.get("symbol"))
    if expected_symbol and symbol and expected_symbol != symbol:
        reasons.append("native_instrument_symbol_mismatch")
    if value.get("stale") is True:
        reasons.append("stale_market_fact")
    quality_issue = _text(value.get("source_quality_issue"))
    if quality_issue:
        reasons.append(f"source_quality:{quality_issue}")
    consistency = _dict(value.get("source_consistency"))
    if consistency:
        reasons.extend(
            f"source_consistency:{reason}"
            for reason in market_source_consistency_reasons(consistency)
        )
    return list(dict.fromkeys(reasons))


def build_market_fact(
    symbol: str,
    snapshot: Mapping[str, Any] | None,
    *,
    contract_spec: Mapping[str, Any] | None = None,
    received_at: Any = None,
) -> dict[str, Any]:
    raw = _dict(snapshot)
    info = _dict(raw.get("info"))
    normalized_symbol = normalize_trading_symbol(symbol or raw.get("symbol"))
    source = _text(raw.get("source") or raw.get("price_source")).lower()
    source_timestamp_ms = _timestamp_ms(
        raw.get("source_timestamp_ms")
        or raw.get("timestamp")
        or info.get("ts")
    )
    identity = _native_identity(normalized_symbol, raw, contract_spec or raw.get("contract_spec"))
    default_endpoint = (
        "okx_ws_public"
        if source in {"websocket", "stale_websocket"}
        else "okx_rest_market"
        if source == "rest"
        else ""
    )
    default_channel = "tickers" if default_endpoint else ""
    fact = {
        "schema_version": MARKET_FACT_SCHEMA_VERSION,
        "symbol": normalized_symbol,
        "native_identity": identity,
        "source_interface": source,
        "source_endpoint": _text(
            raw.get("source_endpoint") or default_endpoint
        ),
        "source_channel": _text(raw.get("source_channel") or default_channel),
        "source_timestamp_ms": source_timestamp_ms,
        "source_sequence": raw.get("source_sequence") or info.get("seqId"),
        "received_at": _iso(received_at or raw.get("received_at") or datetime.now(UTC)),
        "prices": {
            "last": _safe_float(
                raw.get("last_price") or raw.get("last") or raw.get("current_price"), 0.0
            ),
            "bid": _safe_float(raw.get("bid") or info.get("bidPx"), 0.0),
            "ask": _safe_float(raw.get("ask") or info.get("askPx"), 0.0),
            "mark": _safe_float(raw.get("mark_price") or raw.get("markPx"), None),
            "index": _safe_float(raw.get("index_price") or raw.get("idxPx"), None),
            "kind": _text(raw.get("price_kind") or "last_bid_ask"),
        },
        "liquidity": {
            "notional_24h_usdt": _safe_float(raw.get("notional_24h_usdt"), 0.0),
            "volume_24h_contracts": _safe_float(raw.get("volume_24h_contracts"), 0.0),
            "volume_24h_base": _safe_float(
                raw.get("volume_24h_base") or raw.get("volume_24h"), 0.0
            ),
            "bid_depth_usdt": _safe_float(raw.get("orderbook_bid_depth"), 0.0),
            "ask_depth_usdt": _safe_float(raw.get("orderbook_ask_depth"), 0.0),
        },
        "stale": bool(raw.get("stale") or raw.get("ticker_stale")),
        "source_quality_issue": _text(
            raw.get("market_data_quality_issue") or raw.get("price_reconciliation_warning")
        ),
        "source_consistency": _dict(raw.get("market_source_consistency")),
    }
    reasons = market_fact_reasons(fact)
    fact["quality"] = {
        "status": "clean" if not reasons else "quarantined",
        "violation_count": len(reasons),
        "reasons": reasons,
    }
    fact["fact_id"] = f"sha256:{_fingerprint(fact)}"
    return fact


def _bar_payload(row: Any) -> dict[str, Any] | None:
    if isinstance(row, Mapping):
        timestamp = row.get("open_time") or row.get("timestamp") or row.get("ts")
        open_ = row.get("open")
        high = row.get("high")
        low = row.get("low")
        close = row.get("close")
        volume = row.get("volume")
    elif isinstance(row, (list, tuple)) and len(row) >= 6:
        timestamp, open_, high, low, close, volume = row[:6]
    else:
        return None
    timestamp_ms = _timestamp_ms(timestamp)
    prices = [_safe_float(item, None) for item in (open_, high, low, close)]
    if timestamp_ms is None or any(item is None or item <= 0 for item in prices):
        return None
    return {
        "open_time_ms": timestamp_ms,
        "open": prices[0],
        "high": prices[1],
        "low": prices[2],
        "close": prices[3],
        "volume": _safe_float(volume, 0.0),
    }


def _tick_aligned(price: Any, tick_size: Any) -> bool:
    try:
        price_decimal = Decimal(str(price))
        tick_decimal = Decimal(str(tick_size))
    except (InvalidOperation, TypeError, ValueError):
        return False
    if price_decimal <= 0 or tick_decimal <= 0:
        return False
    nearest = (price_decimal / tick_decimal).to_integral_value()
    remainder = abs(price_decimal - nearest * tick_decimal)
    # Float conversion can leave one ULP of residue after arithmetic such as
    # bid = last - tick. Tolerance is derived from representation precision,
    # never from a market or strategy percentage.
    precision = Decimal(str(math.ulp(float(price_decimal)) * 4.0))
    return remainder <= precision


def market_source_consistency_reasons(
    contract: Mapping[str, Any] | None,
) -> list[str]:
    value = _dict(contract)
    if value.get("version") != MARKET_SOURCE_CONSISTENCY_VERSION:
        return ["source_consistency_contract_missing_or_stale"]
    reasons = [_text(item) for item in value.get("reasons") or [] if _text(item)]
    if value.get("status") != "clean":
        return reasons or ["source_consistency_not_clean"]
    assertions = _dict(value.get("assertions"))
    required = (
        "native_identity_verified",
        "executable_quotes_verified",
        "tick_alignment_verified",
        "reference_prices_verified",
        "one_minute_path_verified",
    )
    return [
        f"source_consistency_assertion_failed:{name}"
        for name in required
        if assertions.get(name) is not True
    ]


def build_market_source_consistency(
    primary_fact: Mapping[str, Any] | None,
    comparison_facts: Iterable[Mapping[str, Any]],
    *,
    orderbook_fact: Mapping[str, Any] | None,
    mark_price_fact: Mapping[str, Any] | None,
    index_price_fact: Mapping[str, Any] | None,
    bars: Iterable[Any],
    generated_at: Any = None,
) -> dict[str, Any]:
    primary = _dict(primary_fact)
    facts = [primary]
    seen_ids = {_text(primary.get("fact_id"))}
    for candidate in comparison_facts:
        value = _dict(candidate)
        fact_id = _text(value.get("fact_id"))
        if value and fact_id not in seen_ids:
            facts.append(value)
            seen_ids.add(fact_id)

    identity = _dict(primary.get("native_identity"))
    spec = _dict(identity.get("contract_spec"))
    inst_id = _text(identity.get("inst_id")).upper()
    inst_type = _text(identity.get("inst_type")).upper()
    uly = _text(identity.get("uly")).upper()
    spec_version = _text(identity.get("contract_spec_version"))
    tick_size = _text(spec.get("tick_size"))
    reasons: list[str] = []

    for fact in facts:
        fact_identity = _dict(fact.get("native_identity"))
        if any(
            fact_identity.get(key) != identity.get(key)
            for key in ("inst_id", "inst_type", "uly", "contract_spec_version")
        ):
            reasons.append("native_identity_mismatch_across_ticker_sources")
        reasons.extend(
            f"ticker_fact:{reason}" for reason in market_fact_reasons(fact)
        )

    orderbook = _dict(orderbook_fact)
    mark = _dict(mark_price_fact)
    index = _dict(index_price_fact)
    if not orderbook:
        reasons.append("native_orderbook_fact_missing")
    else:
        if _text(orderbook.get("inst_id")).upper() != inst_id:
            reasons.append("orderbook_native_identity_mismatch")
        if not _timestamp_ms(orderbook.get("source_timestamp_ms")):
            reasons.append("orderbook_source_timestamp_missing")
        if (_safe_float(orderbook.get("bid"), 0.0) or 0.0) <= 0:
            reasons.append("orderbook_executable_bid_missing")
        if (_safe_float(orderbook.get("ask"), 0.0) or 0.0) <= 0:
            reasons.append("orderbook_executable_ask_missing")
        if (_safe_float(orderbook.get("bid_depth_usdt"), 0.0) or 0.0) <= 0:
            reasons.append("orderbook_bid_depth_missing")
        if (_safe_float(orderbook.get("ask_depth_usdt"), 0.0) or 0.0) <= 0:
            reasons.append("orderbook_ask_depth_missing")

    if not mark:
        reasons.append("mark_price_fact_missing")
    else:
        if _text(mark.get("inst_id")).upper() != inst_id:
            reasons.append("mark_price_native_identity_mismatch")
        if _text(mark.get("inst_type")).upper() != inst_type:
            reasons.append("mark_price_instrument_type_mismatch")
        if (_safe_float(mark.get("price"), 0.0) or 0.0) <= 0:
            reasons.append("mark_price_missing")
        if not _timestamp_ms(mark.get("source_timestamp_ms")):
            reasons.append("mark_price_source_timestamp_missing")

    if not index:
        reasons.append("index_price_fact_missing")
    else:
        if _text(index.get("inst_id")).upper() != uly:
            reasons.append("index_price_native_identity_mismatch")
        if (_safe_float(index.get("price"), 0.0) or 0.0) <= 0:
            reasons.append("index_price_missing")
        if not _timestamp_ms(index.get("source_timestamp_ms")):
            reasons.append("index_price_source_timestamp_missing")

    intervals: list[dict[str, Any]] = []
    for fact in facts:
        prices = _dict(fact.get("prices"))
        intervals.append(
            {
                "source": fact.get("source_interface"),
                "bid": _safe_float(prices.get("bid"), 0.0),
                "ask": _safe_float(prices.get("ask"), 0.0),
                "source_timestamp_ms": fact.get("source_timestamp_ms"),
            }
        )
    if orderbook:
        intervals.append(
            {
                "source": "okx_rest_orderbook",
                "bid": _safe_float(orderbook.get("bid"), 0.0),
                "ask": _safe_float(orderbook.get("ask"), 0.0),
                "source_timestamp_ms": orderbook.get("source_timestamp_ms"),
            }
        )
    valid_intervals = [
        interval
        for interval in intervals
        if (interval.get("bid") or 0.0) > 0
        and (interval.get("ask") or 0.0) > 0
        and (interval.get("bid") or 0.0) <= (interval.get("ask") or 0.0)
    ]
    if len(valid_intervals) != len(intervals) or not valid_intervals:
        reasons.append("executable_quote_interval_invalid")

    tick_alignment_verified = bool(tick_size) and all(
        _tick_aligned(interval[side], tick_size)
        for interval in valid_intervals
        for side in ("bid", "ask")
    )
    if not tick_alignment_verified:
        reasons.append("executable_quote_not_tick_aligned")

    normalized_bars = [bar for row in bars if (bar := _bar_payload(row)) is not None]
    normalized_bars.sort(key=lambda item: item["open_time_ms"])
    path_low = min((item["low"] for item in normalized_bars), default=None)
    path_high = max((item["high"] for item in normalized_bars), default=None)
    timestamps = [
        timestamp
        for value in [*facts, orderbook, mark, index]
        if (timestamp := _timestamp_ms(value.get("source_timestamp_ms"))) is not None
    ]
    missing_minutes: list[int] = []
    if timestamps and normalized_bars:
        minute_ms = 60_000
        first_open = min(timestamps) - min(timestamps) % minute_ms
        last_open = max(timestamps) - max(timestamps) % minute_ms
        available_minutes = {
            item["open_time_ms"] - item["open_time_ms"] % minute_ms
            for item in normalized_bars
        }
        missing_minutes = [
            minute
            for minute in range(first_open, last_open + minute_ms, minute_ms)
            if minute not in available_minutes
        ]
    if not timestamps or not normalized_bars:
        reasons.append("recent_one_minute_price_path_missing")
    elif missing_minutes:
        reasons.append("recent_one_minute_price_path_incomplete")

    observed_prices = [
        price
        for fact in facts
        for price in (
            _safe_float(_dict(fact.get("prices")).get("last"), None),
            _safe_float(_dict(fact.get("prices")).get("bid"), None),
            _safe_float(_dict(fact.get("prices")).get("ask"), None),
        )
        if price is not None
    ]
    observed_prices.extend(
        price
        for price in (
            _safe_float(orderbook.get("bid"), None),
            _safe_float(orderbook.get("ask"), None),
            _safe_float(mark.get("price"), None),
            _safe_float(index.get("price"), None),
        )
        if price is not None
    )
    tick = _safe_float(tick_size, 0.0) or 0.0
    prices_within_path = bool(observed_prices and path_low is not None and path_high is not None)
    if prices_within_path:
        prices_within_path = all(
            path_low - tick <= price <= path_high + tick for price in observed_prices
        )
    if not prices_within_path:
        reasons.append("observed_price_outside_recent_native_path")

    quotes_overlap = bool(valid_intervals) and (
        max(float(item["bid"]) for item in valid_intervals)
        <= min(float(item["ask"]) for item in valid_intervals) + tick
    )
    executable_quotes_verified = bool(
        valid_intervals
        and len(valid_intervals) == len(intervals)
        and (quotes_overlap or prices_within_path)
    )
    if not executable_quotes_verified:
        reasons.append("executable_quote_sources_not_reconciled")

    reasons = list(dict.fromkeys(reasons))
    assertions = {
        "native_identity_verified": not any("identity_mismatch" in item for item in reasons),
        "executable_quotes_verified": executable_quotes_verified,
        "tick_alignment_verified": tick_alignment_verified,
        "reference_prices_verified": not any(
            item.startswith(("mark_price_", "index_price_")) for item in reasons
        ),
        "one_minute_path_verified": bool(
            normalized_bars and not missing_minutes and prices_within_path
        ),
    }
    contract = {
        "version": MARKET_SOURCE_CONSISTENCY_VERSION,
        "status": "clean" if not reasons and all(assertions.values()) else "quarantined",
        "reasons": reasons,
        "assertions": assertions,
        "native_identity": {
            "inst_id": inst_id,
            "inst_type": inst_type,
            "uly": uly,
            "contract_spec_version": spec_version,
            "tick_size": tick_size,
        },
        "ticker_sources": [
            {
                "fact_id": fact.get("fact_id"),
                "source_interface": fact.get("source_interface"),
                "source_timestamp_ms": fact.get("source_timestamp_ms"),
            }
            for fact in facts
        ],
        "executable_intervals": valid_intervals,
        "quotes_overlap": quotes_overlap,
        "path": {
            "source": "okx_native_swap_candles_1m",
            "bar_count": len(normalized_bars),
            "path_low": path_low,
            "path_high": path_high,
            "missing_open_times_ms": missing_minutes,
            "observed_prices_within_path": prices_within_path,
        },
        "reference_prices": {"mark": mark, "index": index},
        "orderbook_fact": orderbook,
        "provenance": {
            "source": "okx_native_rest_ws_book_mark_index_1m",
            "observation_window": {
                "start_ms": min(timestamps) if timestamps else None,
                "end_ms": max(timestamps) if timestamps else None,
            },
            "sample_count": len(facts),
            "effective_sample_size": float(len(facts)) if not reasons else 0.0,
            "generated_at": _iso(generated_at or datetime.now(UTC)),
            "strategy_version": MARKET_SOURCE_CONSISTENCY_VERSION,
            "fallback_reason": "" if not reasons else ";".join(reasons),
        },
    }
    contract["provenance"]["data_fingerprint"] = _fingerprint(contract)
    return contract


def verify_market_fact_path(
    entry_fact: Mapping[str, Any] | None,
    result_fact: Mapping[str, Any] | None,
    bars: Iterable[Any],
) -> dict[str, Any]:
    entry = _dict(entry_fact)
    result = _dict(result_fact)
    entry_identity = _dict(entry.get("native_identity"))
    result_identity = _dict(result.get("native_identity"))
    reasons: list[str] = []
    identity_keys = ("inst_id", "inst_type", "uly", "contract_spec_version")
    identity_match = bool(entry_identity) and all(
        entry_identity.get(key) == result_identity.get(key) and entry_identity.get(key)
        for key in identity_keys
    )
    if not identity_match:
        reasons.append("entry_result_native_identity_mismatch")
    entry_ms = _timestamp_ms(entry.get("source_timestamp_ms"))
    result_ms = _timestamp_ms(result.get("source_timestamp_ms"))
    if entry_ms is None or result_ms is None or result_ms < entry_ms:
        reasons.append("invalid_market_fact_time_window")

    normalized_bars = [bar for row in bars if (bar := _bar_payload(row)) is not None]
    normalized_bars.sort(key=lambda item: item["open_time_ms"])
    path_bars: list[dict[str, Any]] = []
    missing_minutes: list[int] = []
    if entry_ms is not None and result_ms is not None and result_ms >= entry_ms:
        minute_ms = 60_000
        first_open = entry_ms - entry_ms % minute_ms
        last_open = result_ms - result_ms % minute_ms
        by_open = {item["open_time_ms"] - item["open_time_ms"] % minute_ms: item for item in normalized_bars}
        expected = list(range(first_open, last_open + minute_ms, minute_ms))
        missing_minutes = [item for item in expected if item not in by_open]
        path_bars = [by_open[item] for item in expected if item in by_open]
        if missing_minutes:
            reasons.append("one_minute_price_path_incomplete")
    if not path_bars:
        reasons.append("one_minute_price_path_missing")

    entry_price = _safe_float(_dict(entry.get("prices")).get("last"), None)
    result_price = _safe_float(_dict(result.get("prices")).get("last"), None)
    path_low = min((item["low"] for item in path_bars), default=None)
    path_high = max((item["high"] for item in path_bars), default=None)
    tick_size = _safe_float(
        _dict(entry_identity.get("contract_spec")).get("tick_size"), 0.0
    ) or 0.0

    def reachable(price: float | None) -> bool:
        return bool(
            price is not None
            and path_low is not None
            and path_high is not None
            and path_low - tick_size <= price <= path_high + tick_size
        )

    entry_reachable = reachable(entry_price)
    result_reachable = reachable(result_price)
    if not entry_reachable:
        reasons.append("entry_price_not_reachable_on_native_path")
    if not result_reachable:
        reasons.append("result_price_not_reachable_on_native_path")
    payload = {
        "source": "okx_native_swap_candles_1m",
        "inst_id": entry_identity.get("inst_id"),
        "entry_source_timestamp_ms": entry_ms,
        "result_source_timestamp_ms": result_ms,
        "bar_count": len(path_bars),
        "missing_open_times_ms": missing_minutes,
        "path_low": path_low,
        "path_high": path_high,
        "entry_price": entry_price,
        "result_price": result_price,
        "entry_reachable": entry_reachable,
        "result_reachable": result_reachable,
        "identity_match": identity_match,
        "reasons": list(dict.fromkeys(reasons)),
    }
    payload["status"] = "clean" if not payload["reasons"] else "quarantined"
    payload["path_fingerprint"] = f"sha256:{_fingerprint({**payload, 'bars': path_bars})}"
    return payload


def build_shadow_market_fact_contract(
    entry_fact: Mapping[str, Any] | None,
    result_fact: Mapping[str, Any] | None,
    price_path: Mapping[str, Any] | None,
    *,
    generated_at: Any = None,
) -> dict[str, Any]:
    entry = _dict(entry_fact)
    result = _dict(result_fact)
    path = _dict(price_path)
    reasons = [f"entry:{reason}" for reason in market_fact_reasons(entry)]
    if not result:
        reasons.append("result_market_fact_missing")
    else:
        reasons.extend(f"result:{reason}" for reason in market_fact_reasons(result))
    if not path:
        reasons.append("native_price_path_missing")
    else:
        reasons.extend(f"path:{reason}" for reason in path.get("reasons") or [])
    reasons = list(dict.fromkeys(reasons))
    entry_clean = not market_fact_reasons(entry)
    result_clean = bool(result) and not market_fact_reasons(result)
    path_clean = bool(path) and path.get("status") == "clean"
    contract = {
        "version": MARKET_FACT_CONTRACT_VERSION,
        "status": "clean" if not reasons else "quarantined",
        "violation_count": len(reasons),
        "violation_reasons": reasons,
        "entry_fact": entry,
        "result_fact": result,
        "price_path": path,
        "assertions": {
            "native_instrument_identity_verified": bool(
                entry_clean and result_clean and path.get("identity_match") is True
            ),
            "same_contract_price_path_verified": path_clean,
            "executable_market_fact_verified": bool(entry_clean and result_clean),
        },
        "provenance": {
            "source": "shadow_entry_result_and_okx_native_1m_path",
            "observation_window": {
                "start": entry.get("received_at") or entry.get("source_timestamp_ms"),
                "end": result.get("received_at") or result.get("source_timestamp_ms"),
            },
            "sample_count": 1,
            "effective_sample_size": 1.0 if not reasons else 0.0,
            "generated_at": _iso(generated_at or datetime.now(UTC)),
            "strategy_version": MARKET_FACT_CONTRACT_VERSION,
            "fallback_reason": "" if not reasons else ";".join(reasons),
            "data_fingerprint": _fingerprint(
                {
                    "entry_fact_id": entry.get("fact_id"),
                    "result_fact_id": result.get("fact_id"),
                    "path_fingerprint": path.get("path_fingerprint"),
                }
            ),
        },
    }
    return contract


def compact_market_fact_contract(contract: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project the full audit contract into the bounded ML training snapshot."""

    value = _dict(contract)
    assertions = _dict(value.get("assertions"))
    provenance = _dict(value.get("provenance"))
    entry = _dict(value.get("entry_fact"))
    result = _dict(value.get("result_fact"))
    entry_identity = _dict(entry.get("native_identity"))
    result_identity = _dict(result.get("native_identity"))
    path = _dict(value.get("price_path"))
    reasons = [_text(item)[:160] for item in value.get("violation_reasons") or [] if _text(item)]
    return {
        "version": value.get("version"),
        "status": value.get("status"),
        "violation_count": value.get("violation_count"),
        "violation_reason_codes": ";".join(reasons),
        "native_instrument_identity_verified": assertions.get(
            "native_instrument_identity_verified"
        ),
        "same_contract_price_path_verified": assertions.get(
            "same_contract_price_path_verified"
        ),
        "executable_market_fact_verified": assertions.get("executable_market_fact_verified"),
        "source": provenance.get("source"),
        "observation_window": json.dumps(
            provenance.get("observation_window"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )[:512],
        "sample_count": provenance.get("sample_count"),
        "effective_sample_size": provenance.get("effective_sample_size"),
        "generated_at": provenance.get("generated_at"),
        "strategy_version": provenance.get("strategy_version"),
        "fallback_reason": _text(provenance.get("fallback_reason"))[:512],
        "data_fingerprint": provenance.get("data_fingerprint"),
        "entry_fact_id": entry.get("fact_id"),
        "result_fact_id": result.get("fact_id"),
        "entry_inst_id": entry_identity.get("inst_id"),
        "result_inst_id": result_identity.get("inst_id"),
        "entry_contract_spec_version": entry_identity.get("contract_spec_version"),
        "result_contract_spec_version": result_identity.get("contract_spec_version"),
        "path_status": path.get("status"),
        "path_fingerprint": path.get("path_fingerprint"),
    }


def market_fact_contract_reasons(contract: Mapping[str, Any] | None) -> list[str]:
    value = _dict(contract)
    if value.get("version") != MARKET_FACT_CONTRACT_VERSION:
        return ["market_fact_contract_missing_or_stale"]
    reasons = [_text(item) for item in value.get("violation_reasons") or [] if _text(item)]
    if not reasons and _text(value.get("violation_reason_codes")):
        reasons = [
            _text(item)
            for item in _text(value.get("violation_reason_codes")).split(";")
            if _text(item)
        ]
    if value.get("status") != "clean" or value.get("violation_count") != 0:
        return reasons or ["market_fact_contract_not_clean"]
    assertions = _dict(value.get("assertions")) or value
    required = (
        "native_instrument_identity_verified",
        "same_contract_price_path_verified",
        "executable_market_fact_verified",
    )
    missing = [item for item in required if assertions.get(item) is not True]
    return [f"market_fact_assertion_failed:{item}" for item in missing]
