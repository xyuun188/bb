from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from core.feature_availability import (
    COMPONENT_AVAILABILITY_VERSION,
    seal_feature_availability,
    sentiment_input_availability,
)
from core.market_facts import MARKET_FACT_CONTRACT_VERSION, build_market_fact
from core.training_contracts import (
    SHADOW_LABEL_VERSION,
    build_shadow_label_contract,
    compact_shadow_label_contract,
)
from core.training_feature_snapshot import (
    FROZEN_TRAINING_FEATURE_KEYS,
    FROZEN_TRAINING_FEATURE_MAX_BYTES,
    TRAINING_FEATURE_SNAPSHOT_MAX_BYTES,
    TRAINING_FEATURE_SNAPSHOT_VERSION,
    frozen_feature_payload_size,
    postgres_training_feature_snapshot_functions,
    training_feature_storage_bytes,
)
from models.decision import AIDecision
from models.learning import (
    ShadowBacktest,
    _compact_training_feature_snapshot,
    _sync_training_feature_snapshot,
)
from services.authoritative_trade_outcome import (
    _COMPACT_TRAINING_FEATURE_KEYS,
    _decision_feature_projection,
)
from services.training_data_quality import assess_shadow_sample, assess_trade_sample

FROZEN_AT = datetime(2026, 10, 22, 1, 0, tzinfo=UTC)
RECEIVED_AT = "2026-10-22T00:59:00+00:00"


def _availability_snapshot() -> dict[str, object]:
    news = [
        {
            "title": f"Bitcoin market update {index}",
            "published_at": "2026-10-22T00:58:00+00:00",
            "received_at": RECEIVED_AT,
        }
        for index in range(20)
    ]
    social = [
        {
            "title": f"Ethereum discussion {index}",
            "posted_at": "2026-10-22T00:57:00+00:00",
            "received_at": RECEIVED_AT,
        }
        for index in range(20)
    ]
    technical = {
        "version": COMPONENT_AVAILABILITY_VERSION,
        "available_at": RECEIVED_AT,
        "timeframes": "1h,4h",
        "source_delivery_observed": True,
        "1h_closed_at": "2026-10-22T00:00:00+00:00",
        "1h_received_at": RECEIVED_AT,
        "4h_closed_at": "2026-10-22T00:00:00+00:00",
        "4h_received_at": RECEIVED_AT,
        "input_fingerprint": "a" * 64,
    }
    sentiment = sentiment_input_availability(
        news,
        social,
        available_at=FROZEN_AT,
    )
    return {
        "feature_snapshot_frozen_at": FROZEN_AT.isoformat(),
        "original_decision_timestamp": "2026-10-22T01:00:10+00:00",
        "feature_availability_contract": {
            "version": "2026-10-22.observed-feature-availability.v1",
            "frozen_at": FROZEN_AT.isoformat(),
            "input_fingerprint": "b" * 64,
        },
        "technical_availability": technical,
        "sentiment_availability": sentiment,
    }


def _conservative_jsonb_size_bound(value: dict[str, object]) -> int:
    """Bound scalar object storage with generous per-key JEntry/alignment room."""

    encoded_bytes = len(
        json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    )
    return encoded_bytes + 32 * len(value) + 16


def test_availability_contracts_survive_training_snapshot_compaction() -> None:
    source = _availability_snapshot()
    compacted = _compact_training_feature_snapshot(source)

    for key in (
        "feature_snapshot_frozen_at",
        "original_decision_timestamp",
        "feature_availability_contract",
        "technical_availability",
        "sentiment_availability",
    ):
        assert compacted[key] == source[key]
        if isinstance(compacted[key], dict):
            assert all(not isinstance(value, (dict, list)) for value in compacted[key].values())

    assert compacted["feature_availability_contract"]["version"] == (
        "2026-10-22.observed-feature-availability.v1"
    )
    assert compacted["feature_availability_contract"]["frozen_at"] == FROZEN_AT.isoformat()
    assert compacted["feature_availability_contract"]["input_fingerprint"] == "b" * 64
    assert compacted["technical_availability"]["input_fingerprint"] == "a" * 64
    assert compacted["sentiment_availability"]["news_input_count"] == 20
    assert compacted["sentiment_availability"]["social_input_count"] == 20
    assert compacted["sentiment_availability"]["news_clock_errors"] == 0
    assert compacted["sentiment_availability"]["social_clock_errors"] == 0


def test_availability_contracts_fit_compaction_strings_and_postgres_jsonb_budget() -> None:
    compacted = _compact_training_feature_snapshot(_availability_snapshot())
    limits = {
        "feature_availability_contract": 1400,
        "technical_availability": 1400,
        "sentiment_availability": 1600,
    }

    for key, encoded_limit in limits.items():
        contract = compacted[key]
        assert all(not isinstance(value, (dict, list)) for value in contract.values())
        assert all(not isinstance(value, str) or len(value) <= 512 for value in contract.values())
        encoded_size = len(
            json.dumps(contract, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        assert encoded_size < encoded_limit
        assert _conservative_jsonb_size_bound(contract) < 2048


def test_sqlalchemy_shadow_compaction_keeps_availability_evidence() -> None:
    source = _availability_snapshot()
    row = ShadowBacktest(
        model_name="test",
        execution_mode="paper",
        symbol="BTC/USDT",
        due_at=FROZEN_AT,
        feature_snapshot=source,
    )

    _sync_training_feature_snapshot(None, None, row)

    assert row.training_feature_snapshot_version == TRAINING_FEATURE_SNAPSHOT_VERSION
    for key in (
        "feature_availability_contract",
        "technical_availability",
        "sentiment_availability",
    ):
        assert row.training_feature_snapshot[key] == source[key]


def _trainable_shadow_sample() -> dict:
    label_at = datetime(2026, 10, 22, 1, 30, tzinfo=UTC)
    market_contract = {
        "version": MARKET_FACT_CONTRACT_VERSION,
        "status": "clean",
        "violation_count": 0,
        "violation_reason_codes": "",
        "native_instrument_identity_verified": True,
        "same_contract_price_path_verified": True,
        "executable_market_fact_verified": True,
        "data_fingerprint": "compact-market-facts",
        "result_source_timestamp_ms": int(label_at.timestamp() * 1000),
        "result_received_at": label_at.isoformat(),
    }
    features = {
        **_availability_snapshot(),
        "symbol": "BTC/USDT",
        "timestamp": "2026-10-22T00:59:00+00:00",
        "current_price": 100.0,
        "spread_pct": 0.03,
        "round_trip_fee_pct": 0.08,
        "funding_rate": 0.0,
        "funding_interval_minutes": 480.0,
        "funding_rate_observed_at": "2026-10-22T00:58:00+00:00",
        "funding_rate_received_at": RECEIVED_AT,
        "account_fee_source_observed_at": "2026-10-22T00:58:00+00:00",
        "account_fee_received_at": RECEIVED_AT,
        "mark_price": None,
        "close_sequence": [100.0 + index / 100 for index in range(60)],
        "volume_sequence": [1000 + index for index in range(60)],
        "news_sources": ["coindesk", "example-source"],
        "training_market_fact_contract": market_contract,
        "recent_headlines": ["raw headline excluded from compact evidence"] * 20,
        "news_articles": [{"title": "raw news", "content": "x" * 100_000}],
        "social_posts": [{"text": "raw social"}],
    }
    for name in ("market_fact", "orderbook_fact", "mark_price_fact", "index_price_fact"):
        features[name] = {
            "timestamp": "2026-10-22T00:58:00+00:00",
            "received_at": RECEIVED_AT,
            "lineage": {"source": "observed"},
        }
    features["training_label_contract"] = compact_shadow_label_contract(
        build_shadow_label_contract(
            shadow_backtest_id=1,
            decision_id=1001,
            horizon_minutes=30,
            long_return_pct=0.42,
            short_return_pct=-0.42,
            best_action="long",
            long_mfe_pct=0.42,
            long_mae_pct=0.0,
            short_mfe_pct=0.0,
            short_mae_pct=0.42,
            long_stop_loss_triggered=False,
            short_stop_loss_triggered=False,
            long_take_profit_triggered=False,
            short_take_profit_triggered=False,
            long_first_touch="none",
            short_first_touch="none",
            market_fact_contract=market_contract,
            cost_facts={"round_trip_fee_pct": 0.08},
            label_timestamp=label_at,
        )
    )
    features.pop("feature_availability_contract")
    seal_feature_availability(features, frozen_at=FROZEN_AT)
    return {
        "id": 1,
        "decision_id": 1001,
        "label_version": SHADOW_LABEL_VERSION,
        "symbol": "BTC/USDT",
        "decision_action": "long",
        "decision_confidence": 0.72,
        "horizon_minutes": 30,
        "features": features,
        "long_return_pct": 0.42,
        "short_return_pct": -0.42,
        "best_action": "long",
        "missed_opportunity": False,
        "label_timestamp": label_at,
    }


def _full_feature_vector_sample() -> dict:
    from data_feed.feature_vector import FeatureVector

    sample = _trainable_shadow_sample()
    vector = FeatureVector(symbol="BTC/USDT", timestamp=FROZEN_AT - timedelta(minutes=1))
    for key, value in sample["features"].items():
        if hasattr(vector, key) and key not in {
            "feature_availability_contract", "feature_snapshot_frozen_at",
        }:
            setattr(vector, key, deepcopy(value))
    vector.close_sequence = [100.0 + index / 100 for index in range(80)]
    vector.volume_sequence = [1000 + index for index in range(80)]
    vector.sequence_length = 80
    vector.funding_data_available = True
    vector.taker_fee_rate = 0.0004
    vector.entry_fee_rate = 0.0004
    vector.exit_fee_rate = 0.0004
    vector.fee_rate_source = "okx_account_fee_snapshot"
    vector.fee_rate_observed_at = RECEIVED_AT
    vector.market_fact = build_market_fact(
        "BTC/USDT",
        {
            "source": "rest", "timestamp": "2026-10-22T00:58:00+00:00",
            "last_price": 100.0, "bid": 99.9, "ask": 100.1,
            "notional_24h_usdt": 10_000_000.0, "volume_24h_base": 100_000.0,
            "orderbook_bid_depth": 100_000.0, "orderbook_ask_depth": 100_000.0,
        },
        contract_spec={
            "instId": "BTC-USDT-SWAP", "instType": "SWAP", "uly": "BTC-USDT",
            "ctVal": "0.01", "ctMult": "1", "ctValCcy": "BTC",
            "settleCcy": "USDT", "lotSz": "1", "minSz": "1", "tickSz": "0.1",
        },
        received_at=RECEIVED_AT,
        require_observed_receipt=True,
    )
    vector.recent_headlines = ["This exact model headline is represented by its digest"]
    vector.recent_news_items = [{"title": "Native model text", "received_at": RECEIVED_AT}]
    features = vector.to_dict()
    features["round_trip_fee_pct"] = sample["features"]["round_trip_fee_pct"]
    features["original_decision_timestamp"] = sample["features"]["original_decision_timestamp"]
    for key in ("training_label_contract", "training_market_fact_contract"):
        features[key] = sample["features"][key]
    features.pop("feature_availability_contract", None)
    seal_feature_availability(features, frozen_at=FROZEN_AT)
    sample["features"] = features
    return sample


def test_full_feature_vector_with_eighty_bar_sequences_fits_and_remains_eligible() -> None:
    from scripts.train_local_ai_tools_models import _compact_local_ai_tools_features

    sample = _full_feature_vector_sample()
    original = deepcopy(sample["features"])
    sample["features"] = _compact_local_ai_tools_features(
        _compact_training_feature_snapshot(sample["features"])
    )
    features = sample["features"]

    assert features["close_sequence"] == original["close_sequence"]
    assert features["volume_sequence"] == original["volume_sequence"]
    assert features["market_fact"] == original["market_fact"]
    assert features["model_text_fingerprint"] == original["model_text_fingerprint"]
    assert frozen_feature_payload_size(features) <= FROZEN_TRAINING_FEATURE_MAX_BYTES
    assert training_feature_storage_bytes(features) <= TRAINING_FEATURE_SNAPSHOT_MAX_BYTES
    assert len(json.dumps(features, ensure_ascii=False).encode()) <= TRAINING_FEATURE_SNAPSHOT_MAX_BYTES
    assert assess_shadow_sample(sample).exclude_from_training is False


def _trade_sample(features: dict) -> dict:
    return {
        "source": "closed_position",
        "symbol": "BTC/USDT",
        "side": "long",
        "entry_price": 100.0,
        "close_price": 101.0,
        "quantity": 0.1,
        "notional": 10.0,
        "realized_pnl": 1.0,
        "entry_fee": 0.02,
        "close_fee": 0.03,
        "entry_fee_source": "okx_fills_history",
        "close_fee_source": "okx_fills_history",
        "funding_fee": 0.0,
        "funding_fee_source": "okx_positions_history.fundingFee",
        "slippage": 0.01,
        "slippage_source": "okx_fills_history_fill_mark_round_trip",
        "net_return_after_all_cost_pct": 10.0,
        "pnl_source": "okx_position_history_realized_pnl",
        "settlement_source": "okx_position_history_settlement",
        "settlement_status": "reconciled",
        "position_size_pct": 0.03,
        "holding_minutes": 35.0,
        "outcome": "profit",
        "features": features,
    }


def test_compacted_sealed_inputs_remain_trainable_without_text_or_resealing() -> None:
    from scripts.train_local_ai_tools_models import (
        _compact_training_shadow_sample,
        _training_transport_features,
    )

    sample = _trainable_shadow_sample()
    original = deepcopy(sample["features"])
    sample["features"] = _compact_training_feature_snapshot(sample["features"])
    compacted_sample = _compact_training_shadow_sample(sample)

    assert compacted_sample is not None
    features = compacted_sample["features"]
    assert assess_shadow_sample(compacted_sample).exclude_from_training is False
    for key in FROZEN_TRAINING_FEATURE_KEYS:
        if key in original:
            assert features[key] == original[key]
    assert "mark_price" in features and features["mark_price"] is None
    assert not {"news_articles", "social_posts", "recent_headlines"} & features.keys()
    transport = _training_transport_features(features)
    assert "feature_availability_contract" not in transport
    assert transport["current_price"] == original["current_price"]
    assert training_feature_storage_bytes(features) <= TRAINING_FEATURE_SNAPSHOT_MAX_BYTES


@pytest.mark.parametrize("mutation", ["sequence", "nested_receipt"])
def test_compaction_keeps_frozen_input_tampering_visible_to_quality(mutation: str) -> None:
    sample = _trainable_shadow_sample()
    if mutation == "sequence":
        sample["features"]["close_sequence"][-1] += 1
    else:
        sample["features"]["orderbook_fact"]["received_at"] = "2026-10-22T01:00:01+00:00"
    sample["features"] = _compact_training_feature_snapshot(sample["features"])

    assessment = assess_shadow_sample(sample)

    assert assessment.exclude_from_training is True
    assert "feature_availability:feature_inputs_changed_after_freeze" in assessment.reasons


def test_oversized_frozen_inputs_fail_closed_without_partial_sealed_payload() -> None:
    sample = _trainable_shadow_sample()
    sample["features"]["close_sequence"] = [100.0] * 1000
    sample["features"].pop("feature_availability_contract")
    seal_feature_availability(sample["features"], frozen_at=FROZEN_AT)
    assert frozen_feature_payload_size(sample["features"]) > FROZEN_TRAINING_FEATURE_MAX_BYTES

    compacted = _compact_training_feature_snapshot(sample["features"])

    assert compacted == {"training_feature_compaction_error": "frozen_input_budget_exceeded"}
    sample["features"] = compacted
    assert assess_shadow_sample(sample).exclude_from_training is True


def test_authoritative_projection_preserves_sealed_inputs_and_nulls_for_quality() -> None:
    from scripts.train_local_ai_tools_models import _compact_local_ai_tools_features

    original = _trainable_shadow_sample()["features"]
    mapping = {f"feature_{key}": value for key, value in original.items()}
    mapping.update({f"feature_present_{key}": True for key in original})

    projected = _decision_feature_projection(SimpleNamespace(_mapping=mapping))
    features = _compact_local_ai_tools_features(projected)

    assert assess_trade_sample(_trade_sample(features)).exclude_from_training is False
    assert features["feature_availability_contract"] == original["feature_availability_contract"]
    assert "mark_price" in features and features["mark_price"] is None
    assert "close" not in features


@pytest.mark.asyncio
async def test_sqlite_snapshot_roundtrip_and_json_projection_remain_eligible(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import settings
    from db.session import close_db, get_session_ctx, init_db
    from scripts.train_local_ai_tools_models import _compact_local_ai_tools_features

    await close_db()
    monkeypatch.setattr(
        settings, "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'availability-compact.db').as_posix()}",
    )
    await init_db()
    sample = _trainable_shadow_sample()
    original = sample["features"]
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="test", symbol="BTC/USDT", action="long",
                confidence=0.72, reasoning="test", feature_snapshot=original,
            )
            shadow = ShadowBacktest(
                decision_id=1001, model_name="test", execution_mode="paper",
                symbol="BTC/USDT", due_at=sample["label_timestamp"], feature_snapshot=original,
            )
            session.add_all([decision, shadow])
            await session.flush()
            decision_id, shadow_id = decision.id, shadow.id
        async with get_session_ctx() as session:
            stored = await session.get(ShadowBacktest, shadow_id)
            assert stored.training_feature_snapshot_version == TRAINING_FEATURE_SNAPSHOT_VERSION
            sample["features"] = stored.training_feature_snapshot
            assert assess_shadow_sample(sample).exclude_from_training is False
            columns = [
                AIDecision.feature_snapshot[key].label(f"feature_{key}")
                for key in _COMPACT_TRAINING_FEATURE_KEYS
            ]
            columns.extend(
                AIDecision.feature_snapshot.op("->")(key).is_not(None).label(
                    f"feature_present_{key}"
                )
                for key in FROZEN_TRAINING_FEATURE_KEYS
            )
            row = (await session.execute(
                select(*columns).where(AIDecision.id == decision_id)
            )).one()
        projected = _compact_local_ai_tools_features(_decision_feature_projection(row))
        assert "mark_price" in projected and projected["mark_price"] is None
        assert "close" not in projected
        assert assess_trade_sample(_trade_sample(projected)).exclude_from_training is False
    finally:
        await close_db()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("BB_TEST_POSTGRES_URL"),
    reason="BB_TEST_POSTGRES_URL must point to a dedicated PostgreSQL test database",
)
async def test_postgres_and_python_snapshot_compaction_match_in_temporary_functions() -> None:
    import asyncpg

    connection = await asyncpg.connect(os.environ["BB_TEST_POSTGRES_URL"])
    transaction = connection.transaction()
    await transaction.start()
    try:
        await connection.execute("CREATE TEMP TABLE bb_compaction_test_scope (id INTEGER)")
        for sql in postgres_training_feature_snapshot_functions():
            await connection.execute(sql.replace("bb_", "pg_temp.bb_"))
        original = _full_feature_vector_sample()["features"]
        oversized = deepcopy(original)
        oversized["close_sequence"] = [100.0] * 1000
        for features in (original, oversized, {"empty": {}, "array": [], "text": "x" * 513}):
            encoded = json.dumps(features, ensure_ascii=False)
            pg_compacted = json.loads(await connection.fetchval(
                "SELECT pg_temp.bb_compact_training_feature_snapshot($1::JSONB)", encoded,
            ))
            assert pg_compacted == _compact_training_feature_snapshot(features)
            assert await connection.fetchval(
                "SELECT pg_temp.bb_training_feature_storage_bytes($1::JSONB)", encoded,
            ) == training_feature_storage_bytes(features)
            assert await connection.fetchval(
                "SELECT pg_column_size($1::JSONB)", json.dumps(pg_compacted),
            ) <= TRAINING_FEATURE_SNAPSHOT_MAX_BYTES
    finally:
        await transaction.rollback()
        await connection.close()
