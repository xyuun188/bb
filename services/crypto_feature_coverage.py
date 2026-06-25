from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.market_data import Kline, Ticker
from models.news import NewsArticle, SocialPost

EXPECTED_KLINE_TIMEFRAMES = ("1m", "5m", "15m", "1h")
KLINE_STALE_LIMIT_SECONDS = {"1m": 180, "5m": 600, "15m": 1800, "1h": 7200}
SNAPSHOT_STALE_LIMIT_SECONDS = 1800
TICKER_STALE_LIMIT_SECONDS = 600
NEWS_STALE_LIMIT_SECONDS = 86400
SOCIAL_STALE_LIMIT_SECONDS = 21600


CRYPTO_FEATURE_LABELS = {
    "kline_1m": "1m Kline",
    "kline_5m": "5m Kline",
    "kline_15m": "15m Kline",
    "kline_1h": "1h Kline",
    "ticker": "Ticker",
    "orderbook_depth": "Orderbook depth",
    "slippage": "Slippage estimate",
    "funding_rate": "Funding rate",
    "open_interest": "Open interest",
    "liquidation_risk": "Liquidation risk",
    "btc_eth_anchor": "BTC/ETH anchor",
    "sector_correlation": "Sector correlation",
    "altcoin_volatility_risk": "Altcoin volatility risk",
    "abnormal_wick": "Abnormal wick",
    "news": "News",
    "social": "Social sentiment",
    "event_calendar": "Event calendar",
}


CORE_MARKET_FEATURES = {"kline_1m", "kline_5m", "kline_15m", "kline_1h", "ticker"}


def summarize_crypto_feature_coverage(
    decisions: Sequence[Any],
    market_coverage: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = _as_utc(now) or datetime.now(UTC)
    coverage = market_coverage if isinstance(market_coverage, dict) else {}
    snapshots = _snapshot_rows(decisions)
    latest_snapshot = snapshots[0]["snapshot"] if snapshots else {}
    latest_snapshot_at = snapshots[0]["timestamp"] if snapshots else None
    evidence = _feature_evidence_from_snapshots(snapshots)
    symbols = sorted(
        {
            str(row.get("symbol") or "").strip()
            for row in snapshots
            if str(row.get("symbol") or "").strip()
        }
    )

    feature_rows: list[dict[str, Any]] = []
    for timeframe in EXPECTED_KLINE_TIMEFRAMES:
        feature_rows.append(_kline_feature(timeframe, coverage, current))
    feature_rows.append(_ticker_feature(coverage, current))
    anchor_evidence = evidence["btc_eth_anchor"]
    if not anchor_evidence["snapshot"]:
        anchor_evidence = _market_anchor_evidence(coverage)
    event_evidence = evidence["event_calendar"]
    if not event_evidence["snapshot"]:
        event_evidence = _market_event_calendar_evidence(coverage)

    feature_rows.extend(
        [
            _orderbook_feature(evidence["orderbook_depth"], symbols, current),
            _slippage_feature(latest_snapshot, latest_snapshot_at, symbols, current),
            _funding_feature(evidence["funding_rate"], symbols, current),
            _open_interest_feature(evidence["open_interest"], symbols, current),
            _liquidation_feature(evidence["liquidation_risk"], symbols, current),
            _anchor_feature(anchor_evidence, symbols, current),
            _sector_feature(evidence["sector_correlation"], symbols, current),
            _altcoin_volatility_feature(evidence["altcoin_volatility_risk"], symbols, current),
            _abnormal_wick_feature(evidence["abnormal_wick"], symbols, current),
            _news_feature(latest_snapshot, latest_snapshot_at, symbols, coverage, current),
            _social_feature(latest_snapshot, latest_snapshot_at, symbols, coverage, current),
            _event_calendar_feature(event_evidence, symbols, current),
        ]
    )

    missing = [row["key"] for row in feature_rows if row["status"] == "missing"]
    stale = [row["key"] for row in feature_rows if row["status"] == "stale"]
    low_confidence = [row["key"] for row in feature_rows if row["status"] == "low_confidence"]
    neutralized = sorted(set(missing + stale + low_confidence))
    core_blocked = any(
        row["key"] in CORE_MARKET_FEATURES for row in feature_rows if row["status"] == "missing"
    )
    status = "ok"
    if core_blocked or not snapshots:
        status = "critical"
    elif missing or stale or low_confidence:
        status = "warning"

    return {
        "audit_only": True,
        "live_signal_mutation": False,
        "feature_defaults_are_neutral": True,
        "can_missing_features_drive_live_entry": False,
        "status": status,
        "generated_at": current.isoformat(),
        "decision_sample_count": len(decisions),
        "feature_snapshot_count": len(snapshots),
        "latest_feature_snapshot_at": _iso(latest_snapshot_at),
        "symbols_observed": symbols,
        "features": feature_rows,
        "missing_features": missing,
        "stale_features": stale,
        "low_confidence_features": low_confidence,
        "neutralized_features": neutralized,
        "feature_contribution_policy": {
            "missing_feature_policy": "neutral_blocked",
            "stale_feature_policy": "neutral_blocked",
            "low_confidence_event_policy": "shadow_only",
            "required_timestamp_policy": "feature_or_source_timestamp_required",
        },
        "safety_rules": [
            "missing_data_source_not_silent_ok",
            "default_zero_never_bullish",
            "low_confidence_events_shadow_only",
            "stale_features_block_live_influence",
            "feature_coverage_report_is_read_only",
        ],
    }


class CryptoFeatureCoverageService:
    def __init__(self, session_context_factory: Any = get_read_session_ctx) -> None:
        self._session_context_factory = session_context_factory

    @staticmethod
    def _decision_projection_columns(model: Any = AIDecision) -> tuple[Any, Any, Any, Any]:
        return (model.id, model.symbol, model.created_at, model.feature_snapshot)

    async def report(self, *, hours: int = 24, limit: int = 1000) -> dict[str, Any]:
        from sqlalchemy import func, or_, select

        capped_hours = max(1, min(int(hours or 24), 168))
        capped_limit = max(50, min(int(limit or 1000), 5000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        async with self._session_context_factory() as session:
            decisions_result = await session.execute(
                select(*self._decision_projection_columns())
                .where(AIDecision.created_at >= since)
                .order_by(AIDecision.created_at.desc())
                .limit(capped_limit)
            )
            kline_rows = list(
                (
                    await session.execute(
                        select(
                            Kline.timeframe,
                            func.count(Kline.id),
                            func.count(func.distinct(Kline.symbol)),
                            func.max(Kline.open_time),
                        )
                        .where(Kline.timeframe.in_(EXPECTED_KLINE_TIMEFRAMES))
                        .group_by(Kline.timeframe)
                    )
                ).all()
            )
            ticker_row = (
                await session.execute(
                    select(
                        func.count(Ticker.id),
                        func.max(func.coalesce(Ticker.updated_at, Ticker.created_at)),
                    )
                )
            ).one()
            news_row = (
                await session.execute(
                    select(
                        func.count(NewsArticle.id),
                        func.max(func.coalesce(NewsArticle.published_at, NewsArticle.fetched_at)),
                    )
                )
            ).one()
            social_row = (
                await session.execute(
                    select(func.count(SocialPost.id), func.max(SocialPost.posted_at))
                )
            ).one()
            event_source_filter = or_(
                NewsArticle.source == "coinmarketcal",
                NewsArticle.source == "okx_announcements",
                NewsArticle.source.like("scrapling:%"),
            )
            event_row = (
                await session.execute(
                    select(
                        func.count(NewsArticle.id),
                        func.max(func.coalesce(NewsArticle.published_at, NewsArticle.fetched_at)),
                    ).where(event_source_filter)
                )
            ).one()
            event_source_rows = list(
                (
                    await session.execute(
                        select(
                            NewsArticle.source,
                            func.count(NewsArticle.id),
                            func.max(
                                func.coalesce(NewsArticle.published_at, NewsArticle.fetched_at)
                            ),
                        )
                        .where(event_source_filter)
                        .group_by(NewsArticle.source)
                        .order_by(func.count(NewsArticle.id).desc())
                        .limit(20)
                    )
                ).all()
            )
            anchor_ticker_rows = list(
                (
                    await session.execute(
                        select(
                            Ticker.symbol,
                            Ticker.change_24h_pct,
                            func.coalesce(Ticker.updated_at, Ticker.created_at),
                        ).where(Ticker.symbol.in_(("BTC/USDT", "ETH/USDT")))
                    )
                ).all()
            )
        decision_rows = [
            SimpleDecisionProjection(
                id=row[0],
                symbol=row[1],
                created_at=row[2],
                feature_snapshot=row[3],
            )
            for row in decisions_result.all()
        ]
        return summarize_crypto_feature_coverage(
            decision_rows,
            _market_coverage_from_rows(
                kline_rows,
                ticker_row,
                news_row,
                social_row,
                anchor_ticker_rows,
                event_row,
                event_source_rows,
            ),
        )


class SimpleDecisionProjection:
    def __init__(
        self,
        *,
        id: Any,
        symbol: Any,
        created_at: Any,
        feature_snapshot: Any,
    ) -> None:
        self.id = id
        self.symbol = symbol
        self.created_at = created_at
        self.feature_snapshot = feature_snapshot


def _market_coverage_from_rows(
    kline_rows: Sequence[Any],
    ticker_row: Any,
    news_row: Any,
    social_row: Any,
    anchor_ticker_rows: Sequence[Any] | None = None,
    event_row: Any | None = None,
    event_source_rows: Sequence[Any] | None = None,
) -> dict[str, Any]:
    klines: dict[str, Any] = {}
    for row in kline_rows:
        timeframe = str(row[0] or "")
        klines[timeframe] = {
            "rows": int(row[1] or 0),
            "symbols": int(row[2] or 0),
            "latest_at": row[3],
        }
    anchor: dict[str, Any] = {}
    for row in anchor_ticker_rows or ():
        symbol = str(row[0] or "").upper()
        if symbol.startswith("BTC/"):
            anchor["btc"] = {"change_24h_pct": row[1], "latest_at": row[2]}
        elif symbol.startswith("ETH/"):
            anchor["eth"] = {"change_24h_pct": row[1], "latest_at": row[2]}
    return {
        "klines": klines,
        "ticker": {"count": int(ticker_row[0] or 0), "latest_at": ticker_row[1]},
        "btc_eth_anchor": anchor,
        "news": {"count": int(news_row[0] or 0), "latest_at": news_row[1]},
        "social": {"count": int(social_row[0] or 0), "latest_at": social_row[1]},
        "event_calendar": {
            "count": int(event_row[0] or 0) if event_row is not None else 0,
            "latest_at": event_row[1] if event_row is not None else None,
            "sources": [
                {"source": str(row[0] or ""), "count": int(row[1] or 0), "latest_at": row[2]}
                for row in event_source_rows or ()
            ],
        },
    }


def _snapshot_rows(decisions: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        snapshot = getattr(decision, "feature_snapshot", None)
        if not isinstance(snapshot, dict) or not snapshot:
            continue
        timestamp = _parse_datetime(snapshot.get("timestamp")) or _as_utc(
            getattr(decision, "created_at", None)
        )
        rows.append(
            {
                "symbol": getattr(decision, "symbol", None),
                "timestamp": timestamp,
                "snapshot": snapshot,
            }
        )
    return sorted(
        rows,
        key=lambda item: item.get("timestamp") or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )


def _feature_evidence_from_snapshots(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = (
        "orderbook_depth",
        "funding_rate",
        "open_interest",
        "liquidation_risk",
        "btc_eth_anchor",
        "sector_correlation",
        "altcoin_volatility_risk",
        "abnormal_wick",
        "event_calendar",
    )
    evidence = {key: {"snapshot": {}, "timestamp": None} for key in keys}
    for row in rows:
        snapshot = _safe_dict(row.get("snapshot"))
        timestamp = _as_utc(row.get("timestamp"))
        if not evidence["orderbook_depth"]["snapshot"] and _has_orderbook_depth(snapshot):
            evidence["orderbook_depth"] = {"snapshot": snapshot, "timestamp": timestamp}
        if not evidence["funding_rate"]["snapshot"] and _has_funding_presence(snapshot):
            evidence["funding_rate"] = {"snapshot": snapshot, "timestamp": timestamp}
        if not evidence["open_interest"]["snapshot"] and _has_open_interest(snapshot):
            evidence["open_interest"] = {"snapshot": snapshot, "timestamp": timestamp}
        if not evidence["liquidation_risk"]["snapshot"] and _first_positive(
            snapshot,
            "liquidation_usd_1h",
            "liquidation_usd_4h",
            "liquidation_risk_score",
        ):
            evidence["liquidation_risk"] = {"snapshot": snapshot, "timestamp": timestamp}
        if (
            not evidence["btc_eth_anchor"]["snapshot"]
            and _first_present(
                snapshot,
                "btc_eth_anchor",
                "btc_change_24h_pct",
                "eth_change_24h_pct",
            )
            is not None
        ):
            evidence["btc_eth_anchor"] = {"snapshot": snapshot, "timestamp": timestamp}
        if (
            not evidence["sector_correlation"]["snapshot"]
            and _first_present(
                snapshot,
                "sector_correlation",
                "sector_relative_strength",
            )
            is not None
        ):
            evidence["sector_correlation"] = {"snapshot": snapshot, "timestamp": timestamp}
        if not evidence["altcoin_volatility_risk"]["snapshot"]:
            value = _first_present(snapshot, "altcoin_volatility_risk", "volatility_20")
            if value is not None and _safe_float(value, 0.0) > 0:
                evidence["altcoin_volatility_risk"] = {
                    "snapshot": snapshot,
                    "timestamp": timestamp,
                }
        if not evidence["abnormal_wick"]["snapshot"] and _has_abnormal_wick_presence(snapshot):
            evidence["abnormal_wick"] = {"snapshot": snapshot, "timestamp": timestamp}
        if not evidence["event_calendar"]["snapshot"] and _has_event_calendar_presence(snapshot):
            evidence["event_calendar"] = {"snapshot": snapshot, "timestamp": timestamp}
    return evidence


def _market_anchor_evidence(coverage: dict[str, Any]) -> dict[str, Any]:
    anchor = _safe_dict(coverage.get("btc_eth_anchor"))
    btc = _safe_dict(anchor.get("btc"))
    eth = _safe_dict(anchor.get("eth"))
    snapshot: dict[str, Any] = {}
    if "change_24h_pct" in btc and btc.get("change_24h_pct") is not None:
        snapshot["btc_change_24h_pct"] = btc.get("change_24h_pct")
    if "change_24h_pct" in eth and eth.get("change_24h_pct") is not None:
        snapshot["eth_change_24h_pct"] = eth.get("change_24h_pct")
    timestamp = _latest_datetime(
        [_parse_datetime(btc.get("latest_at")), _parse_datetime(eth.get("latest_at"))]
    )
    if not snapshot:
        return {"snapshot": {}, "timestamp": None, "source": "market_tickers"}
    return {"snapshot": snapshot, "timestamp": timestamp, "source": "market_tickers"}


def _market_event_calendar_evidence(coverage: dict[str, Any]) -> dict[str, Any]:
    event = _safe_dict(coverage.get("event_calendar"))
    count = int(_safe_float(event.get("count"), 0.0))
    latest = _parse_datetime(event.get("latest_at"))
    source_rows = [
        row
        for row in _safe_list(event.get("sources"))
        if isinstance(row, dict) and row.get("source")
    ]
    if count <= 0 and not source_rows:
        return {"snapshot": {}, "timestamp": None, "source": "event_calendar_sources"}
    items = [
        {
            "source": str(row.get("source") or ""),
            "published_at": _iso(_parse_datetime(row.get("latest_at"))),
            "source_weight": _event_source_weight(str(row.get("source") or "")),
            "event_calendar": True,
            "count": int(_safe_float(row.get("count"), 0.0)),
        }
        for row in source_rows[:12]
    ]
    snapshot = {
        "event_calendar_items": items,
        "event_calendar_source_count": len(items),
        "event_calendar_item_count": count,
    }
    return {"snapshot": snapshot, "timestamp": latest, "source": "event_calendar_sources"}


def _feature_row(
    key: str,
    status: str,
    *,
    source: str = "missing",
    confidence: float = 0.0,
    timestamp: datetime | None = None,
    age_seconds: float | None = None,
    affected_symbols: Sequence[str] | None = None,
    reasons: Sequence[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocked = status in {"missing", "stale", "low_confidence"}
    return {
        "key": key,
        "label": CRYPTO_FEATURE_LABELS.get(key, key),
        "status": status,
        "source": source,
        "confidence": round(max(min(float(confidence or 0.0), 1.0), 0.0), 4),
        "timestamp": _iso(timestamp),
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "affected_symbols": list(affected_symbols or []),
        "live_entry_influence": "blocked" if blocked else "eligible",
        "reasons": list(reasons or []),
        "details": details or {},
    }


def _kline_feature(timeframe: str, coverage: dict[str, Any], now: datetime) -> dict[str, Any]:
    rows = _safe_dict(_safe_dict(coverage.get("klines")).get(timeframe))
    count = int(_safe_float(rows.get("rows"), 0.0))
    symbols = int(_safe_float(rows.get("symbols"), 0.0))
    latest = _parse_datetime(rows.get("latest_at"))
    age = _age_seconds(latest, now)
    key = f"kline_{timeframe}"
    if count <= 0:
        return _feature_row(
            key,
            "missing",
            source="market_klines",
            timestamp=latest,
            age_seconds=age,
            reasons=["timeframe_missing"],
            details={"rows": count, "symbols": symbols, "timeframe": timeframe},
        )
    stale = age is None or age > KLINE_STALE_LIMIT_SECONDS[timeframe]
    return _feature_row(
        key,
        "stale" if stale else "available",
        source="market_klines",
        confidence=0.75 if stale else 1.0,
        timestamp=latest,
        age_seconds=age,
        reasons=["timeframe_stale"] if stale else [],
        details={"rows": count, "symbols": symbols, "timeframe": timeframe},
    )


def _ticker_feature(coverage: dict[str, Any], now: datetime) -> dict[str, Any]:
    ticker = _safe_dict(coverage.get("ticker"))
    count = int(_safe_float(ticker.get("count"), 0.0))
    latest = _parse_datetime(ticker.get("latest_at"))
    age = _age_seconds(latest, now)
    if count <= 0:
        return _feature_row(
            "ticker",
            "missing",
            source="market_tickers",
            timestamp=latest,
            age_seconds=age,
            reasons=["ticker_missing"],
            details={"count": count},
        )
    stale = age is None or age > TICKER_STALE_LIMIT_SECONDS
    return _feature_row(
        "ticker",
        "stale" if stale else "available",
        source="market_tickers",
        confidence=0.7 if stale else 1.0,
        timestamp=latest,
        age_seconds=age,
        reasons=["ticker_stale"] if stale else [],
        details={"count": count},
    )


def _orderbook_feature(
    evidence: dict[str, Any], symbols: Sequence[str], now: datetime
) -> dict[str, Any]:
    snapshot = _safe_dict(evidence.get("snapshot"))
    timestamp = _as_utc(evidence.get("timestamp"))
    bid_depth = _safe_float(snapshot.get("orderbook_bid_depth"), 0.0)
    ask_depth = _safe_float(snapshot.get("orderbook_ask_depth"), 0.0)
    imbalance = _safe_float(snapshot.get("orderbook_imbalance"), 0.0)
    if bid_depth <= 0 or ask_depth <= 0:
        return _feature_row(
            "orderbook_depth",
            "missing",
            source="feature_snapshot",
            timestamp=timestamp,
            age_seconds=_age_seconds(timestamp, now),
            affected_symbols=symbols,
            reasons=["default_zero_without_presence_flag"],
            details={"bid_depth": bid_depth, "ask_depth": ask_depth, "imbalance": imbalance},
        )
    return _snapshot_feature(
        "orderbook_depth",
        timestamp,
        symbols,
        now,
        confidence=0.95,
        details={"bid_depth": bid_depth, "ask_depth": ask_depth, "imbalance": imbalance},
    )


def _slippage_feature(
    snapshot: dict[str, Any], timestamp: datetime | None, symbols: Sequence[str], now: datetime
) -> dict[str, Any]:
    spread = _safe_float(snapshot.get("spread_pct"), 0.0)
    bid = _safe_float(snapshot.get("bid"), 0.0)
    ask = _safe_float(snapshot.get("ask"), 0.0)
    bid_depth = _safe_float(snapshot.get("orderbook_bid_depth"), 0.0)
    ask_depth = _safe_float(snapshot.get("orderbook_ask_depth"), 0.0)
    has_spread = spread > 0 or (bid > 0 and ask >= bid)
    if not has_spread and (bid_depth <= 0 or ask_depth <= 0):
        return _feature_row(
            "slippage",
            "missing",
            source="feature_snapshot",
            timestamp=timestamp,
            age_seconds=_age_seconds(timestamp, now),
            affected_symbols=symbols,
            reasons=["spread_and_depth_missing"],
        )
    return _snapshot_feature(
        "slippage",
        timestamp,
        symbols,
        now,
        confidence=0.85 if has_spread else 0.65,
        details={"spread_pct": spread, "bid": bid, "ask": ask},
    )


def _funding_feature(
    evidence: dict[str, Any], symbols: Sequence[str], now: datetime
) -> dict[str, Any]:
    snapshot = _safe_dict(evidence.get("snapshot"))
    timestamp = _as_utc(evidence.get("timestamp"))
    rate = _safe_float(snapshot.get("funding_rate"), 0.0)
    next_time = snapshot.get("next_funding_time")
    if abs(rate) <= 1e-12 and not next_time:
        return _feature_row(
            "funding_rate",
            "missing",
            source="feature_snapshot",
            timestamp=timestamp,
            age_seconds=_age_seconds(timestamp, now),
            affected_symbols=symbols,
            reasons=["default_zero_without_presence_flag"],
            details={"funding_rate": rate, "next_funding_time": next_time},
        )
    return _snapshot_feature(
        "funding_rate",
        timestamp,
        symbols,
        now,
        confidence=0.9,
        details={"funding_rate": rate, "next_funding_time": next_time},
    )


def _open_interest_feature(
    evidence: dict[str, Any], symbols: Sequence[str], now: datetime
) -> dict[str, Any]:
    snapshot = _safe_dict(evidence.get("snapshot"))
    timestamp = _as_utc(evidence.get("timestamp"))
    contracts = _safe_float(snapshot.get("open_interest_contracts"), 0.0)
    value = _safe_float(snapshot.get("open_interest_value"), 0.0)
    if contracts <= 0 and value <= 0:
        return _feature_row(
            "open_interest",
            "missing",
            source="feature_snapshot",
            timestamp=timestamp,
            age_seconds=_age_seconds(timestamp, now),
            affected_symbols=symbols,
            reasons=["default_zero_without_presence_flag"],
            details={"contracts": contracts, "value": value},
        )
    return _snapshot_feature(
        "open_interest",
        timestamp,
        symbols,
        now,
        confidence=0.9,
        details={"contracts": contracts, "value": value},
    )


def _liquidation_feature(
    evidence: dict[str, Any], symbols: Sequence[str], now: datetime
) -> dict[str, Any]:
    snapshot = _safe_dict(evidence.get("snapshot"))
    timestamp = _as_utc(evidence.get("timestamp"))
    value = _first_positive(
        snapshot, "liquidation_usd_1h", "liquidation_usd_4h", "liquidation_risk_score"
    )
    if value <= 0:
        return _feature_row(
            "liquidation_risk",
            "missing",
            source="feature_snapshot",
            timestamp=timestamp,
            age_seconds=_age_seconds(timestamp, now),
            affected_symbols=symbols,
            reasons=["liquidation_feed_missing"],
        )
    return _snapshot_feature(
        "liquidation_risk", timestamp, symbols, now, confidence=0.8, details={"score": value}
    )


def _anchor_feature(
    evidence: dict[str, Any], symbols: Sequence[str], now: datetime
) -> dict[str, Any]:
    snapshot = _safe_dict(evidence.get("snapshot"))
    timestamp = _as_utc(evidence.get("timestamp"))
    source = str(evidence.get("source") or "feature_snapshot")
    value = _first_present(snapshot, "btc_eth_anchor", "btc_change_24h_pct", "eth_change_24h_pct")
    if value is None:
        return _feature_row(
            "btc_eth_anchor",
            "missing",
            source=source,
            timestamp=timestamp,
            age_seconds=_age_seconds(timestamp, now),
            affected_symbols=symbols,
            reasons=["btc_eth_anchor_missing"],
        )
    age = _age_seconds(timestamp, now)
    stale_limit = (
        TICKER_STALE_LIMIT_SECONDS if source == "market_tickers" else SNAPSHOT_STALE_LIMIT_SECONDS
    )
    stale = age is None or age > stale_limit
    return _feature_row(
        "btc_eth_anchor",
        "stale" if stale else "available",
        source=source,
        confidence=0.6 if stale else 0.8,
        timestamp=timestamp,
        age_seconds=age,
        affected_symbols=symbols,
        reasons=["btc_eth_anchor_stale"] if stale else [],
        details={
            "btc_change_24h_pct": snapshot.get("btc_change_24h_pct"),
            "eth_change_24h_pct": snapshot.get("eth_change_24h_pct"),
        },
    )


def _sector_feature(
    evidence: dict[str, Any], symbols: Sequence[str], now: datetime
) -> dict[str, Any]:
    snapshot = _safe_dict(evidence.get("snapshot"))
    timestamp = _as_utc(evidence.get("timestamp"))
    value = _first_present(snapshot, "sector_correlation", "sector_relative_strength")
    if value is None:
        return _feature_row(
            "sector_correlation",
            "missing",
            source="feature_snapshot",
            timestamp=timestamp,
            age_seconds=_age_seconds(timestamp, now),
            affected_symbols=symbols,
            reasons=["sector_mapping_missing"],
        )
    return _snapshot_feature("sector_correlation", timestamp, symbols, now, confidence=0.7)


def _altcoin_volatility_feature(
    evidence: dict[str, Any], symbols: Sequence[str], now: datetime
) -> dict[str, Any]:
    snapshot = _safe_dict(evidence.get("snapshot"))
    timestamp = _as_utc(evidence.get("timestamp"))
    value = _first_present(snapshot, "altcoin_volatility_risk", "volatility_20")
    if value is None or _safe_float(value, 0.0) <= 0:
        return _feature_row(
            "altcoin_volatility_risk",
            "missing",
            source="feature_snapshot",
            timestamp=timestamp,
            age_seconds=_age_seconds(timestamp, now),
            affected_symbols=symbols,
            reasons=["altcoin_volatility_feature_missing"],
        )
    return _snapshot_feature(
        "altcoin_volatility_risk",
        timestamp,
        symbols,
        now,
        confidence=0.7,
        details={"volatility": _safe_float(value, 0.0)},
    )


def _abnormal_wick_feature(
    evidence: dict[str, Any], symbols: Sequence[str], now: datetime
) -> dict[str, Any]:
    snapshot = _safe_dict(evidence.get("snapshot"))
    timestamp = _as_utc(evidence.get("timestamp"))
    count = int(_safe_float(snapshot.get("abnormal_wick_count_72h"), 0.0))
    max_pct = _safe_float(snapshot.get("abnormal_wick_max_pct"), 0.0)
    recent_hours = _safe_float(snapshot.get("abnormal_wick_recent_hours"), 9999.0)
    if not _has_abnormal_wick_presence(snapshot):
        return _feature_row(
            "abnormal_wick",
            "missing",
            source="feature_snapshot",
            timestamp=timestamp,
            age_seconds=_age_seconds(timestamp, now),
            affected_symbols=symbols,
            reasons=["default_absence_marker_without_presence_flag"],
        )
    return _snapshot_feature(
        "abnormal_wick",
        timestamp,
        symbols,
        now,
        confidence=0.85,
        details={"count_72h": count, "max_pct": max_pct, "recent_hours": recent_hours},
    )


def _news_feature(
    snapshot: dict[str, Any],
    timestamp: datetime | None,
    symbols: Sequence[str],
    coverage: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    article_count = int(_safe_float(snapshot.get("news_article_count"), 0.0))
    direct_count = int(_safe_float(snapshot.get("direct_news_item_count"), 0.0))
    market_count = int(_safe_float(snapshot.get("market_news_item_count"), 0.0))
    items = [
        item for item in _safe_list(snapshot.get("recent_news_items")) if isinstance(item, dict)
    ]
    db_news = _safe_dict(coverage.get("news"))
    db_count = int(_safe_float(db_news.get("count"), 0.0))
    source_timestamp = _latest_datetime([timestamp, _parse_datetime(db_news.get("latest_at"))])
    age = _age_seconds(source_timestamp, now)
    if (
        article_count <= 0
        and direct_count <= 0
        and market_count <= 0
        and not items
        and db_count <= 0
    ):
        return _feature_row(
            "news",
            "missing",
            source="news_articles",
            timestamp=source_timestamp,
            age_seconds=age,
            affected_symbols=symbols,
            reasons=["news_feed_missing"],
        )
    confidence = _news_confidence(items, direct_count, article_count or db_count)
    status = (
        "low_confidence"
        if confidence < 0.35
        else ("stale" if age is None or age > NEWS_STALE_LIMIT_SECONDS else "available")
    )
    reasons: list[str] = []
    if status == "low_confidence":
        reasons.append("low_source_confidence")
    if status == "stale":
        reasons.append("news_stale")
    return _feature_row(
        "news",
        status,
        source="feature_snapshot" if article_count or items else "news_articles",
        confidence=confidence,
        timestamp=source_timestamp,
        age_seconds=age,
        affected_symbols=symbols,
        reasons=reasons,
        details={
            "article_count": article_count,
            "direct_count": direct_count,
            "db_count": db_count,
        },
    )


def _social_feature(
    snapshot: dict[str, Any],
    timestamp: datetime | None,
    symbols: Sequence[str],
    coverage: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    mention_count = int(_safe_float(snapshot.get("social_mention_count"), 0.0))
    db_social = _safe_dict(coverage.get("social"))
    db_count = int(_safe_float(db_social.get("count"), 0.0))
    source_timestamp = _latest_datetime([timestamp, _parse_datetime(db_social.get("latest_at"))])
    age = _age_seconds(source_timestamp, now)
    if mention_count <= 0 and db_count <= 0:
        return _feature_row(
            "social",
            "missing",
            source="social_posts",
            timestamp=source_timestamp,
            age_seconds=age,
            affected_symbols=symbols,
            reasons=["social_feed_missing"],
        )
    confidence = min(max((mention_count or db_count) / 10.0, 0.35), 0.9)
    stale = age is None or age > SOCIAL_STALE_LIMIT_SECONDS
    return _feature_row(
        "social",
        "stale" if stale else "available",
        source="feature_snapshot" if mention_count else "social_posts",
        confidence=confidence,
        timestamp=source_timestamp,
        age_seconds=age,
        affected_symbols=symbols,
        reasons=["social_stale"] if stale else [],
        details={"mention_count": mention_count, "db_count": db_count},
    )


def _event_calendar_feature(
    evidence: dict[str, Any], symbols: Sequence[str], now: datetime
) -> dict[str, Any]:
    snapshot = _safe_dict(evidence.get("snapshot"))
    timestamp = _as_utc(evidence.get("timestamp"))
    items = [
        item for item in _safe_list(snapshot.get("event_calendar_items")) if isinstance(item, dict)
    ]
    news_items = [
        item for item in _safe_list(snapshot.get("recent_news_items")) if isinstance(item, dict)
    ]
    calendar_items = [
        item
        for item in news_items
        if "coinmarketcal" in str(item.get("source") or "").lower()
        or bool(item.get("event_calendar"))
    ]
    all_items = items + calendar_items
    event_time = _latest_datetime(
        _parse_datetime(item.get("published_at") or item.get("event_time") or item.get("timestamp"))
        for item in all_items
    )
    source_timestamp = _latest_datetime([event_time, timestamp]) if all_items else timestamp
    if not all_items:
        return _feature_row(
            "event_calendar",
            "missing",
            source="event_calendar",
            timestamp=source_timestamp,
            age_seconds=_age_seconds(source_timestamp, now),
            affected_symbols=symbols,
            reasons=["dedicated_event_calendar_missing"],
        )
    confidence = _event_confidence(all_items)
    status = "low_confidence" if confidence < 0.45 else "available"
    return _feature_row(
        "event_calendar",
        status,
        source="event_calendar",
        confidence=confidence,
        timestamp=source_timestamp,
        age_seconds=_age_seconds(source_timestamp, now),
        affected_symbols=symbols,
        reasons=["low_source_confidence"] if status == "low_confidence" else [],
        details={"item_count": len(all_items)},
    )


def _snapshot_feature(
    key: str,
    timestamp: datetime | None,
    symbols: Sequence[str],
    now: datetime,
    *,
    confidence: float,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    age = _age_seconds(timestamp, now)
    stale = age is None or age > SNAPSHOT_STALE_LIMIT_SECONDS
    return _feature_row(
        key,
        "stale" if stale else "available",
        source="feature_snapshot",
        confidence=confidence if not stale else min(confidence, 0.6),
        timestamp=timestamp,
        age_seconds=age,
        affected_symbols=symbols,
        reasons=["feature_snapshot_stale_or_timestamp_missing"] if stale else [],
        details=details,
    )


def _news_confidence(
    items: Sequence[dict[str, Any]], direct_count: int, article_count: int
) -> float:
    weights = [_safe_float(item.get("source_weight"), 0.0) for item in items]
    if weights:
        base = max(weights)
    elif direct_count > 0:
        base = 0.65
    elif article_count > 0:
        base = 0.45
    else:
        base = 0.0
    if direct_count > 0:
        base += 0.15
    return min(base, 0.95)


def _event_confidence(items: Sequence[dict[str, Any]]) -> float:
    weights = [_safe_float(item.get("source_weight"), 0.0) for item in items]
    if weights:
        return min(max(weights), 0.95)
    return 0.5


def _event_source_weight(source: str) -> float:
    value = str(source or "").lower()
    if value == "coinmarketcal":
        return 0.8
    if value == "okx_announcements":
        return 0.88
    if value.startswith("scrapling:"):
        return 0.68
    return 0.5


def _latest_datetime(values: Any) -> datetime | None:
    parsed = [_as_utc(value) for value in values if _as_utc(value) is not None]
    return max(parsed) if parsed else None


def _first_positive(snapshot: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = _safe_float(snapshot.get(key), 0.0)
        if value > 0:
            return value
    return 0.0


def _first_present(snapshot: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in snapshot and snapshot.get(key) is not None:
            return snapshot.get(key)
    return None


def _has_orderbook_depth(snapshot: dict[str, Any]) -> bool:
    return (
        _safe_float(snapshot.get("orderbook_bid_depth"), 0.0) > 0
        and _safe_float(snapshot.get("orderbook_ask_depth"), 0.0) > 0
    )


def _has_funding_presence(snapshot: dict[str, Any]) -> bool:
    return abs(_safe_float(snapshot.get("funding_rate"), 0.0)) > 1e-12 or bool(
        snapshot.get("next_funding_time")
    )


def _has_open_interest(snapshot: dict[str, Any]) -> bool:
    return (
        _safe_float(snapshot.get("open_interest_contracts"), 0.0) > 0
        or _safe_float(snapshot.get("open_interest_value"), 0.0) > 0
    )


def _has_abnormal_wick_presence(snapshot: dict[str, Any]) -> bool:
    return any(
        key in snapshot and snapshot.get(key) is not None
        for key in (
            "abnormal_wick_count_72h",
            "abnormal_wick_max_pct",
            "abnormal_wick_recent_hours",
        )
    )


def _has_event_calendar_presence(snapshot: dict[str, Any]) -> bool:
    if _safe_list(snapshot.get("event_calendar_items")):
        return True
    for item in _safe_list(snapshot.get("recent_news_items")):
        if isinstance(item, dict) and (
            "coinmarketcal" in str(item.get("source") or "").lower()
            or bool(item.get("event_calendar"))
        ):
            return True
    return False


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _as_utc(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _age_seconds(value: datetime | None, now: datetime) -> float | None:
    dt = _as_utc(value)
    if dt is None:
        return None
    return max((now - dt).total_seconds(), 0.0)


def _iso(value: datetime | None) -> str | None:
    dt = _as_utc(value)
    return dt.isoformat() if dt else None
