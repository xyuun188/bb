from datetime import UTC, datetime, timedelta

from core.reason_codes import ReasonCode
from services.candidate_pool import CandidatePool, CandidatePoolConfig


def _candidate(symbol: str, **overrides):
    value = {
        "symbol": symbol,
        "exchange_available": True,
        "asset_type": "swap",
        "listed_at": datetime(2025, 1, 1, tzinfo=UTC),
        "history_coverage": 1.0,
        "quote_volume_24h_usdt": 1_000_000,
        "spread_bps": 5,
        "price": 1,
        "min_order_notional_usdt": 1,
        "volatility_pct": 2,
        "strategy_compatible": True,
        "protection_allowed": True,
    }
    value.update(overrides)
    return value


def test_candidate_pool_runs_documented_order_and_keeps_removal_evidence() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    pool = CandidatePool(
        CandidatePoolConfig(
            whitelist=frozenset({"BTC/USDT", "ETH/USDT", "NEW/USDT"}),
            min_listing_age=timedelta(days=7),
        )
    )
    report = pool.build(
        [
            _candidate("BTC/USDT"),
            _candidate("ETH/USDT", spread_bps=100),
            _candidate("NEW/USDT", listed_at=now - timedelta(days=1)),
            _candidate("XRP/USDT"),
        ],
        now=now,
    ).as_dict()

    assert [stage["filter"] for stage in report["funnel"]] == list(CandidatePool.FILTERS)
    assert [item["symbol"] for item in report["accepted"]] == ["BTC/USDT"]
    rejected = {item["symbol"]: item for item in report["rejected"]}
    assert rejected["ETH/USDT"]["reason_code"] == ReasonCode.MARKET_SPREAD
    assert rejected["NEW/USDT"]["reason_code"] == ReasonCode.MARKET_DATA_INCOMPLETE
    assert rejected["XRP/USDT"]["reason_code"] == ReasonCode.MARKET_NOT_WHITELISTED
    assert all(item["reason_evidence"]["blocker"] for item in report["rejected"])


def test_candidate_pool_is_diagnostic_and_preserves_input_records() -> None:
    source = _candidate("BTC/USDT")
    report = CandidatePool().build([source]).as_dict()
    assert report["read_only"] is True
    assert report["is_entry_gate"] is False
    assert "candidate_pool" not in source
