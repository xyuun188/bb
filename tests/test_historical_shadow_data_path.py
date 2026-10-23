import hashlib
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.training_contracts import SHADOW_LABEL_VERSION
from models.learning import ShadowBacktest
from services import data_service as module
from services.data_service import DataService
from services.shadow_backtest_service import ShadowBacktestService


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", [None, "missing", "invalid", "timeout", "storage"])
async def test_old_window_uses_history_facts_without_cache_eviction(monkeypatch, defect):
    start = datetime(2026, 7, 10, tzinfo=UTC)
    stamps = [int((start + timedelta(minutes=i)).timestamp() * 1000) for i in range(6)]
    bars = [[stamp, 100, 102, 99, 101, 10] for stamp in stamps]
    if defect == "missing":
        bars.pop(2)
    if defect == "invalid":
        bars[2][2] = 90

    class Result:
        def scalars(self):
            return self

        def all(self):
            return []

    class Session:
        async def execute(self, _statement):
            return Result()

    @asynccontextmanager
    async def session_ctx():
        yield Session()

    calls = []
    preserved = []

    class Repository:
        def __init__(self, _session):
            pass

        async def upsert_klines_bulk(self, symbol, timeframe, rows):
            if defect == "storage":
                raise RuntimeError("test evidence storage failed")
            preserved.append((symbol, timeframe, rows))

    async def fetch(symbol, *, limit, end_timestamp_ms):
        calls.append((symbol, limit, end_timestamp_ms))
        if defect == "timeout":
            raise TimeoutError("test history timeout")
        return bars

    async def forbidden_persist(*_args, **_kwargs):
        raise AssertionError("a historical outcome must not depend on rolling-cache persistence")

    service = object.__new__(DataService)
    service.rest_client = SimpleNamespace(fetch_historical_ohlcv=fetch)
    service._persist_klines = forbidden_persist
    monkeypatch.setattr(module, "get_session_ctx", session_ctx)
    monkeypatch.setattr(module, "MarketRepository", Repository)
    if defect == "storage":
        with pytest.raises(RuntimeError, match="evidence storage failed"):
            await service.get_historical_shadow_market_path(
                "BTC/USDT", {"source_timestamp_ms": stamps[0]},
                start + timedelta(minutes=5), 5,
            )
        return
    result = await service.get_historical_shadow_market_path(
        "BTC/USDT", {"source_timestamp_ms": stamps[0]},
        start + timedelta(minutes=5), 5,
    )

    assert calls == [("BTC/USDT", 8, stamps[-1])]
    if defect:
        assert result == {}
        assert preserved == []
    else:
        assert len(preserved) == 1
        assert preserved[0][:2] == ("BTC/USDT", "1m")
        assert len(preserved[0][2]) == 6
        assert result["price_path"]["bar_count"] == 6
        assert result["result_fact"]["source_timestamp_ms"] == stamps[-1]
        contract = result["training_market_fact_contract"]
        assert contract["path_status"] == "clean"
        assert len(contract["path_fingerprint"]) == 64
        evidence = result["price_path"]["ohlcv_evidence"]
        assert len(evidence["bars"]) == 6
        assert hashlib.sha256(json.dumps(
            evidence, ensure_ascii=True, separators=(",", ":"), default=str,
        ).encode("utf-8")).hexdigest() == contract["path_fingerprint"]


@pytest.mark.asyncio
async def test_current_unclosed_result_candle_stays_pending():
    async def forbidden_fetch(*_args, **_kwargs):
        raise AssertionError("must wait until the result candle is closed")

    service = object.__new__(DataService)
    service.rest_client = SimpleNamespace(fetch_historical_ohlcv=forbidden_fetch)
    due = datetime.now(UTC).replace(second=0, microsecond=0) + timedelta(minutes=1)
    assert await service.get_historical_shadow_market_path(
        "BTC/USDT", {}, due, 5,
    ) == {}


@pytest.mark.asyncio
async def test_projected_queue_rows_settle_through_real_repository(monkeypatch):
    start = datetime(2026, 7, 10, tzinfo=UTC)
    stamps = [int((start + timedelta(minutes=i)).timestamp()*1000) for i in range(6)]
    bars = [[stamp, 100, 102, 99, 101, 10] for stamp in stamps]
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def session_ctx():
        async with sessions() as session:
            yield session
            await session.commit()

    async def history(*_args, **_kwargs):
        return bars

    async def no_quote(_symbol):
        raise AssertionError("historical settlement must not fetch today's quote")

    data = object.__new__(DataService)
    data.rest_client = SimpleNamespace(fetch_historical_ohlcv=history)
    monkeypatch.setattr(module, "get_session_ctx", session_ctx)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(ShadowBacktest.__table__.create)
            await connection.run_sync(module.Kline.__table__.create)
        async with session_ctx() as session:
            session.add(ShadowBacktest(
                id=1, decision_id=17, model_name="ensemble_trader",
                execution_mode="paper", symbol="BTC/USDT", analysis_type="market",
                entry_price=100, decision_action="hold", decision_confidence=0.7,
                label_version=SHADOW_LABEL_VERSION, horizon_minutes=5,
                due_at=start+timedelta(minutes=5), created_at=start, status="pending",
                feature_snapshot={
                    "rsi_14": 55,
                    "market_fact": {"source_timestamp_ms": stamps[0]},
                    "market_data_quality": {"code": "orderbook_depth_invalid"},
                    "taker_fee_rate": 0.0005, "spread_pct": 0.02,
                    "funding_rate": 0.0, "funding_interval_minutes": 480,
                },
                raw_llm_response={"large": "unneeded-response"*1000},
            ))
        service = ShadowBacktestService(
            latest_price_provider=no_quote, symbol_normalizer=lambda x: x,
            float_parser=lambda x, fallback=0: float(x) if x is not None else fallback,
            session_factory=session_ctx,
            historical_market_path_provider=data.get_historical_shadow_market_path,
        )
        assert await service.update_due(limit=1) == 1
        async with session_ctx() as session:
            row = (await session.execute(select(ShadowBacktest))).scalar_one()
            assert row.status == "quarantined"
            assert "feature_availability_contract_missing" in row.note
            assert row.actual_price == 101
            label = row.training_feature_snapshot["training_label_contract"]
            assert label["scheduled_label_at"] == row.due_at.replace(tzinfo=UTC).isoformat()
            assert label["label_timestamp"] == (start + timedelta(minutes=6)).isoformat()
            assert row.training_feature_snapshot["historical_market_path_only"] is True
            assert len(row.training_feature_snapshot[
                "training_market_fact_contract"
            ]["path_fingerprint"]) == 64
            evidence = row.feature_snapshot["market_fact_contract"]["price_path"][
                "ohlcv_evidence"
            ]
            assert len(evidence["bars"]) == 6
            assert len((await session.execute(select(module.Kline))).scalars().all()) == 6
    finally:
        await engine.dispose()
