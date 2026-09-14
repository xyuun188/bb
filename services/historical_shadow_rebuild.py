"""Rebuild immutable historical market-opportunity shadow labels from facts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import exists, func, select

from core.market_facts import MARKET_FACT_CONTRACT_VERSION
from core.training_contracts import (
    HISTORICAL_SHADOW_REBUILD_VERSION,
    HISTORICAL_SHADOW_SOURCE,
    SHADOW_LABEL_VERSION,
    build_shadow_label_contract,
    compact_shadow_label_contract,
)
from db.session import get_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest
from models.market_data import Kline
from services.shadow_backtest_service import shadow_path_labels

DEFAULT_HISTORICAL_HORIZONS_MINUTES = (5, 15, 60, 240)


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _minute(value: datetime) -> datetime:
    return _as_utc(value).replace(second=0, microsecond=0)


def _timestamp_ms(value: datetime) -> int:
    return int(_as_utc(value).timestamp() * 1000)


def _historical_fact(symbol: str, bar: dict[str, Any], *, role: str) -> dict[str, Any]:
    close = float(bar["close"])
    return {
        "schema_version": "historical_ohlcv_fact.v1",
        "symbol": symbol,
        "native_identity": {"symbol": symbol, "source": "market_klines", "timeframe": "1m"},
        "source_interface": "stored_market_klines",
        "source_endpoint": "market_klines",
        "source_channel": "1m",
        "source_timestamp_ms": _timestamp_ms(bar["open_time"]),
        "received_at": _as_utc(bar["open_time"]).isoformat(),
        "prices": {"last": close, "open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"])},
        "liquidity": {"notional_24h_usdt": 0.0, "volume_24h_contracts": 0.0, "volume_24h_base": float(bar["volume"])},
        "stale": False,
        "role": role,
    }


def _historical_market_contract(
    *,
    symbol: str,
    entry_fact: dict[str, Any],
    result_fact: dict[str, Any],
    bars: list[dict[str, Any]],
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
            {"timestamp_ms": _timestamp_ms(bar["open_time"]), "open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"]}
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
            "observation_window": {"start": entry_fact["received_at"], "end": result_fact["received_at"]},
            "sample_count": len(bars),
            "effective_sample_size": 1.0,
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": HISTORICAL_SHADOW_REBUILD_VERSION,
            "fallback_reason": "historical_ohlcv_has_no_historical_orderbook",
            "data_fingerprint": data_fingerprint,
        },
    }


def build_historical_shadow_sample(
    decision: Any,
    *,
    horizon_minutes: int,
    bars: Iterable[dict[str, Any]],
    shadow_backtest_id: int = 0,
) -> dict[str, Any] | None:
    """Build one matured market-only shadow row without inventing microstructure."""

    created_at = getattr(decision, "created_at", None)
    symbol = str(getattr(decision, "symbol", "") or "").strip()
    if not isinstance(created_at, datetime) or not symbol or horizon_minutes <= 0:
        return None
    ordered = sorted((dict(row) for row in bars), key=lambda row: _as_utc(row["open_time"]))
    entry_at = _minute(created_at)
    result_at = entry_at + timedelta(minutes=int(horizon_minutes))
    window = [row for row in ordered if entry_at <= _as_utc(row["open_time"]) <= result_at]
    if not window or _as_utc(window[0]["open_time"]) != entry_at or _as_utc(window[-1]["open_time"]) != result_at:
        return None
    expected = int(horizon_minutes) + 1
    if len(window) != expected:
        return None
    features = dict(getattr(decision, "feature_snapshot", None) or {})
    entry_price = _finite_float(features.get("current_price"), None) or _finite_float(window[0]["close"], None)
    result_price = _finite_float(window[-1]["close"], None)
    if entry_price is None or result_price is None or entry_price <= 0 or result_price <= 0:
        return None
    entry_fact = _historical_fact(symbol, window[0], role="entry")
    result_fact = _historical_fact(symbol, window[-1], role="result")
    contract = _historical_market_contract(symbol=symbol, entry_fact=entry_fact, result_fact=result_fact, bars=window)
    path_labels = shadow_path_labels(
        entry_price=entry_price,
        price_path=contract["price_path"],
        stop_loss_fraction=_finite_float(getattr(decision, "stop_loss_pct", None), None),
        take_profit_fraction=_finite_float(getattr(decision, "take_profit_pct", None), None),
    )
    long_return = (result_price - entry_price) / entry_price * 100.0
    short_return = (entry_price - result_price) / entry_price * 100.0
    best_action = "long" if long_return > 0 and long_return >= short_return else "short" if short_return > 0 else "hold"
    feature_snapshot = {
        **features,
        "symbol": symbol,
        "current_price": entry_price,
        "historical_market_path_only": True,
        "historical_shadow_rebuild_version": HISTORICAL_SHADOW_REBUILD_VERSION,
        "market_fact": entry_fact,
        "market_fact_contract": contract,
        "training_market_fact_contract": {
            "version": contract["version"],
            "status": contract["status"],
            "native_instrument_identity_verified": True,
            "same_contract_price_path_verified": True,
            "executable_market_fact_verified": False,
            "source": HISTORICAL_SHADOW_SOURCE,
            "path_status": "clean",
            "path_fingerprint": contract["provenance"]["data_fingerprint"],
            "data_fingerprint": contract["provenance"]["data_fingerprint"],
        },
    }
    label_contract = compact_shadow_label_contract(
        build_shadow_label_contract(
            shadow_backtest_id=int(shadow_backtest_id),
            decision_id=int(getattr(decision, "id", 0) or 0),
            horizon_minutes=int(horizon_minutes),
            long_return_pct=long_return,
            short_return_pct=short_return,
            best_action=best_action,
            market_fact_contract=feature_snapshot["training_market_fact_contract"],
            cost_facts={"cost_complete": False},
            label_timestamp=result_at,
            **path_labels,
        )
    )
    feature_snapshot["training_label_contract"] = label_contract
    return {
        "decision_id": int(getattr(decision, "id", 0) or 0),
        "model_name": str(getattr(decision, "model_name", "") or "ensemble_trader"),
        "execution_mode": "paper" if bool(getattr(decision, "is_paper", True)) else "live",
        "symbol": symbol,
        "analysis_type": "market",
        "decision_action": str(getattr(decision, "action", "hold") or "hold"),
        "decision_confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
        "entry_price": entry_price,
        "feature_snapshot": feature_snapshot,
        "raw_llm_response": {},
        "status": "completed",
        "created_at": _as_utc(created_at),
        "updated_at": datetime.now(UTC),
        "due_at": result_at,
        "horizon_minutes": int(horizon_minutes),
        "label_version": SHADOW_LABEL_VERSION,
        "actual_price": result_price,
        "long_return_pct": long_return,
        "short_return_pct": short_return,
        "best_action": best_action,
        "missed_opportunity": str(getattr(decision, "action", "hold") or "hold").lower() == "hold" and best_action != "hold",
        "note": f"{HISTORICAL_SHADOW_SOURCE}; no historical orderbook, market-opportunity task only",
    }


async def rebuild_historical_shadow_samples(
    *,
    since: datetime,
    horizons_minutes: tuple[int, ...] = DEFAULT_HISTORICAL_HORIZONS_MINUTES,
    batch_size: int = 500,
    max_decisions: int | None = None,
) -> dict[str, int | str]:
    """Rebuild missing rows from preserved decisions and 1m K-line facts."""

    horizons = tuple(sorted({int(item) for item in horizons_minutes if int(item) > 0}))
    if not horizons:
        return {"scanned": 0, "created": 0, "skipped": 0, "horizons": ""}
    scanned = created = skipped = 0
    async with get_session_ctx() as session:
        decision_stmt = (
            select(AIDecision)
            .where(
                AIDecision.analysis_type == "market",
                AIDecision.created_at >= _as_utc(since),
                AIDecision.feature_snapshot.is_not(None),
                exists(
                    select(Kline.id).where(
                        Kline.timeframe == "1m",
                        Kline.symbol == AIDecision.symbol,
                        Kline.open_time == func.date_trunc("minute", AIDecision.created_at),
                    )
                ),
            )
            .order_by(AIDecision.created_at.asc(), AIDecision.id.asc())
        )
        if max_decisions is not None:
            decision_stmt = decision_stmt.limit(max(int(max_decisions), 0))
        decisions = list((await session.execute(decision_stmt)).scalars().all())
        if decisions:
            symbols = sorted({str(item.symbol) for item in decisions if str(item.symbol or "").strip()})
            first_at = _minute(decisions[0].created_at)
            last_at = _minute(decisions[-1].created_at) + timedelta(minutes=max(horizons))
            kline_rows = (
                await session.execute(
                    select(Kline)
                    .where(Kline.timeframe == "1m", Kline.symbol.in_(symbols), Kline.open_time >= first_at, Kline.open_time <= last_at)
                    .order_by(Kline.symbol.asc(), Kline.open_time.asc())
                )
            ).scalars().all()
            bars_by_key: dict[tuple[str, datetime], dict[str, Any]] = {}
            for row in kline_rows:
                open_time = _as_utc(row.open_time)
                bars_by_key[(str(row.symbol), open_time)] = {
                    "open_time": open_time,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                }
        else:
            bars_by_key = {}
        decision_ids = [int(item.id) for item in decisions]
        existing_pairs = (
            {
                (int(decision_id), int(horizon))
                for decision_id, horizon in (
                    await session.execute(
                        select(
                            ShadowBacktest.decision_id,
                            ShadowBacktest.horizon_minutes,
                        ).where(
                            ShadowBacktest.decision_id.in_(decision_ids),
                            ShadowBacktest.label_version == SHADOW_LABEL_VERSION,
                        )
                    )
                ).all()
                if decision_id is not None
            }
            if decision_ids
            else set()
        )
        for offset in range(0, len(decisions), max(int(batch_size), 1)):
            batch = decisions[offset : offset + max(int(batch_size), 1)]
            staged: list[tuple[ShadowBacktest, Any, int, list[dict[str, Any]]]] = []
            for decision in batch:
                scanned += 1
                symbol = str(decision.symbol)
                entry_at = _minute(decision.created_at)
                for horizon in horizons:
                    if (int(decision.id), horizon) in existing_pairs:
                        continue
                    window = [
                        bars_by_key.get((symbol, entry_at + timedelta(minutes=minute)))
                        for minute in range(horizon + 1)
                    ]
                    if any(bar is None for bar in window):
                        skipped += 1
                        continue
                    complete_window = [dict(bar) for bar in window if bar is not None]
                    sample = build_historical_shadow_sample(
                        decision,
                        horizon_minutes=horizon,
                        bars=complete_window,
                    )
                    if sample is None:
                        skipped += 1
                        continue
                    row = ShadowBacktest(**sample)
                    session.add(row)
                    staged.append((row, decision, horizon, complete_window))
                    created += 1
            await session.flush()
            for row, decision, horizon, window in staged:
                finalized = build_historical_shadow_sample(
                    decision,
                    horizon_minutes=horizon,
                    bars=window,
                    shadow_backtest_id=int(row.id),
                )
                if finalized is None:
                    raise RuntimeError("historical shadow sample changed during finalization")
                row.feature_snapshot = finalized["feature_snapshot"]
            await session.flush()
    return {"scanned": scanned, "created": created, "skipped": skipped, "horizons": ",".join(map(str, horizons))}
