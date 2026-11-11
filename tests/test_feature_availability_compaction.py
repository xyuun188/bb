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
    FROZEN_FEATURE_FIELDS,
    availability_fingerprint,
    feature_availability_reasons,
    seal_feature_availability,
    sentiment_input_availability,
)
from core.market_facts import (
    MARKET_FACT_CONTRACT_VERSION,
    build_market_fact,
    compact_market_fact,
    compact_market_fact_contract,
    market_fact_contract_reasons,
    market_fact_reasons,
)
from core.training_contracts import (
    HISTORICAL_SHADOW_SOURCE,
    SHADOW_LABEL_VERSION,
    build_shadow_label_contract,
    compact_shadow_label_contract,
)
from core.training_feature_snapshot import (
    FEATURE_STORAGE_COMPACTION_VERSION,
    FROZEN_TRAINING_FEATURE_KEYS,
    FROZEN_TRAINING_FEATURE_MAX_BYTES,
    LEGACY_FEATURE_STORAGE_COMPACTION_VERSION,
    TRAINING_FEATURE_SNAPSHOT_MAX_BYTES,
    TRAINING_FEATURE_SNAPSHOT_VERSION,
    _jsonb_text,
    _storage_fingerprint_value,
    compact_frozen_feature_payload,
    compact_input_fingerprint,
    compact_training_feature_snapshot,
    frozen_feature_payload_size,
    postgres_training_feature_snapshot_functions,
    storage_feature_fingerprint,
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
    result = {
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
    result["feature_availability_contract"]["input_fingerprint"] = availability_fingerprint(
        {key: result[key] for key in FROZEN_FEATURE_FIELDS if key in result}
    )
    return result


def _conservative_jsonb_size_bound(value: dict[str, object]) -> int:
    """Bound scalar object storage with generous per-key JEntry/alignment room."""

    encoded_bytes = len(
        json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    )
    return encoded_bytes + 32 * len(value) + 16


def _json_differences(left: object, right: object, path: str = "$") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}"
            if key not in left or key not in right:
                differences.append(
                    f"{child_path}: key presence differs "
                    f"(left={key in left}, right={key in right})"
                )
            else:
                differences.extend(_json_differences(left[key], right[key], child_path))
            if len(differences) >= 20:
                break
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: array lengths differ ({len(left)} != {len(right)})"]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            differences.extend(_json_differences(left_item, right_item, f"{path}[{index}]"))
            if len(differences) >= 20:
                break
        return differences
    if type(left) is not type(right) or left != right:
        return [f"{path}: {left!r} != {right!r}"]
    return []


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
    assert compacted["feature_availability_contract"]["input_fingerprint"] == (
        source["feature_availability_contract"]["input_fingerprint"]
    )
    assert compacted["technical_availability"]["input_fingerprint"] == "a" * 64
    assert compacted["sentiment_availability"]["news_input_count"] == 20
    assert compacted["sentiment_availability"]["social_input_count"] == 20
    assert compacted["sentiment_availability"]["news_clock_errors"] == 0
    assert compacted["sentiment_availability"]["social_clock_errors"] == 0


def test_storage_fingerprint_normalizes_json_number_scale() -> None:
    assert storage_feature_fingerprint({"value": 1}) == storage_feature_fingerprint(
        {"value": 1.0}
    )
    assert storage_feature_fingerprint({"value": 0.0}) == storage_feature_fingerprint(
        {"value": 0}
    )
    assert storage_feature_fingerprint({"nested": {"value": None}}) == (
        storage_feature_fingerprint({"nested": {}})
    )
    assert storage_feature_fingerprint({"values": [None]}) != storage_feature_fingerprint(
        {"values": []}
    )


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


def _large_native_market_fact_contract() -> dict:
    native_identity = {
        "inst_id": "BTC-USDT-SWAP",
        "inst_type": "SWAP",
        "uly": "BTC-USDT",
        "contract_spec_version": "okx-contract-spec-v1",
    }
    return {
        "version": MARKET_FACT_CONTRACT_VERSION,
        "status": "clean",
        "violation_count": 0,
        "violation_reasons": [],
        "assertions": {
            "native_instrument_identity_verified": True,
            "same_contract_price_path_verified": True,
            "executable_market_fact_verified": True,
        },
        "entry_fact": {
            "fact_id": "entry-fact",
            "native_identity": native_identity,
            "raw_provider_payload": [{"book": [100.0] * 300} for _ in range(8)],
        },
        "result_fact": {
            "fact_id": "result-fact",
            "source_timestamp_ms": 1792632600000,
            "received_at": "2026-10-22T01:30:00+00:00",
            "native_identity": native_identity,
            "raw_provider_payload": [{"book": [101.0] * 300} for _ in range(8)],
        },
        "price_path": {
            "status": "clean",
            "path_fingerprint": "sha256:" + "a" * 64,
            "_ordered_bar_ranges": [
                {"open_time_ms": index * 60_000, "high": 101.0, "low": 99.0}
                for index in range(500)
            ],
        },
        "provenance": {
            "source": "shadow_entry_result_and_okx_native_1m_path",
            "observation_window": {
                "start": "2026-10-22T01:00:00+00:00",
                "end": "2026-10-22T01:30:00+00:00",
            },
            "sample_count": 1,
            "effective_sample_size": 1.0,
            "generated_at": "2026-10-22T01:30:01+00:00",
            "strategy_version": MARKET_FACT_CONTRACT_VERSION,
            "fallback_reason": "",
            "data_fingerprint": "b" * 64,
            "scheduled_label_at": "2026-10-22T01:29:25+00:00",
            "result_source_available_at": "2026-10-22T01:30:00+00:00",
            "label_maturity_lag_seconds": 35.0,
            "availability_basis": "one_minute_candle_close_lower_bound",
            "delivery_latency_observed": False,
        },
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
    assert features["market_fact"]["prices"] == {
        key: value
        for key, value in original["market_fact"]["prices"].items()
        if value is not None
    }
    assert features["market_fact"]["liquidity"] == original["market_fact"]["liquidity"]
    assert features["market_fact"]["fact_id"] == original["market_fact"]["fact_id"]
    assert features["market_fact"]["storage_contract"]["full_fact_fingerprint"] == (
        original["market_fact"]["fact_id"]
    )
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
            if key == "market_fact":
                assert features[key] == compact_market_fact(original[key])
            else:
                assert features[key] == original[key]
    assert "mark_price" in features and features["mark_price"] is None
    assert not {"news_articles", "social_posts", "recent_headlines"} & features.keys()
    transport = _training_transport_features(features)
    assert "feature_availability_contract" not in transport
    assert transport["current_price"] == original["current_price"]
    assert training_feature_storage_bytes(features) <= TRAINING_FEATURE_SNAPSHOT_MAX_BYTES


def test_required_market_contract_is_compacted_and_survives_optional_budget() -> None:
    sample = _trainable_shadow_sample()
    features = sample["features"]
    contract = _large_native_market_fact_contract()
    features["training_market_fact_contract"] = contract
    features.update(
        {
            f"optional_diagnostic_{index:03d}": "x" * 512
            for index in range(80)
        }
    )

    assert training_feature_storage_bytes(contract) > 16 * 1024
    compacted = _compact_training_feature_snapshot(features)
    compact_contract = compacted["training_market_fact_contract"]
    expected_contract = compact_market_fact_contract(contract)
    expected_contract["observation_window"] = json.loads(
        expected_contract["observation_window"]
    )

    assert compacted.get("training_feature_compaction_error") is None
    assert compact_contract == expected_contract
    assert training_feature_storage_bytes(compact_contract) <= 16 * 1024
    assert _compact_training_feature_snapshot(compacted) == compacted
    assert market_fact_contract_reasons(compact_contract) == []
    assert compacted["feature_availability_contract"]["input_fingerprint"] == (
        features["feature_availability_contract"]["input_fingerprint"]
    )
    assert "training_market_fact_contract" not in FROZEN_TRAINING_FEATURE_KEYS
    assert "optional_diagnostic_079" not in compacted
    sample["features"] = compacted
    assert assess_shadow_sample(sample).status == "included"


def test_explicit_market_path_status_takes_precedence_over_nested_status() -> None:
    contract = _large_native_market_fact_contract()
    contract["path_status"] = "clean"
    contract["price_path"]["status"] = "historical_ohlcv_only"

    compacted = compact_market_fact_contract(contract)

    assert compacted["path_status"] == "clean"


def test_required_market_contract_does_not_change_frozen_input_fingerprint() -> None:
    features = _trainable_shadow_sample()["features"]
    original_fingerprint = features["feature_availability_contract"]["input_fingerprint"]
    first_contract = _large_native_market_fact_contract()
    changed_contract = deepcopy(first_contract)
    changed_contract["provenance"]["data_fingerprint"] = "c" * 64

    first = {
        **features,
        "training_market_fact_contract": first_contract,
    }
    second = {
        **features,
        "training_market_fact_contract": changed_contract,
    }

    assert compact_input_fingerprint(first, original_fingerprint) == compact_input_fingerprint(
        second,
        original_fingerprint,
    )


def test_missing_market_contract_still_fails_closed_after_compaction() -> None:
    sample = _trainable_shadow_sample()
    sample["features"].pop("training_market_fact_contract")
    sample["features"] = _compact_training_feature_snapshot(sample["features"])

    assessment = assess_shadow_sample(sample)

    assert assessment.exclude_from_training is True
    assert "market_fact_contract:market_fact_contract_missing_or_stale" in assessment.reasons


def test_market_fact_storage_projection_preserves_quality_and_native_inputs() -> None:
    fact = _full_feature_vector_sample()["features"]["market_fact"]
    fact["source_consistency"] = {
        "version": "2026-07-17.okx-source-consistency.v4",
        "status": "clean",
        "reasons": [],
        "assertions": {
            "native_identity_verified": True,
            "executable_quotes_verified": True,
            "tick_alignment_verified": True,
            "reference_prices_verified": True,
            "market_continuity_verified": True,
        },
        "orderbook_fact": {"raw_depth": [100.0] * 200},
        "reference_prices": {"raw_mark": {"book": [101.0] * 200}},
        "provenance": {
            "source": "okx_native_rest_ws_book_mark_index_1m",
            "strategy_version": "2026-07-17.okx-source-consistency.v4",
            "data_fingerprint": "source-consistency-digest",
            "fallback_reason": "x" * 512,
        },
    }
    fact["quality"] = {"status": "clean", "violation_count": 0, "reasons": []}

    compact = compact_market_fact(fact)

    assert compact["prices"] == {
        key: value for key, value in fact["prices"].items() if value is not None
    }
    assert compact["liquidity"] == fact["liquidity"]
    assert compact["native_identity"]["contract_spec_version"] == (
        fact["native_identity"]["contract_spec_version"]
    )
    assert compact["source_consistency"]["assertions"] == (
        fact["source_consistency"]["assertions"]
    )
    assert compact["storage_contract"]["full_fact_fingerprint"] == fact["fact_id"]
    assert compact["storage_contract"]["full_source_consistency_fingerprint"] == (
        "source-consistency-digest"
    )
    assert set(compact["source_consistency"]["provenance"]) == {
        "source", "strategy_version", "data_fingerprint", "fallback_reason",
    }
    assert "orderbook_fact" not in compact["source_consistency"]
    assert "reference_prices" not in compact["source_consistency"]
    assert len(compact["source_consistency"]["provenance"]["fallback_reason"]) == 160
    assert market_fact_reasons(compact) == market_fact_reasons(fact)


def test_verbose_market_consistency_fits_frozen_budget_without_losing_quality_gate() -> None:
    sample = _full_feature_vector_sample()
    features = sample["features"]
    fact = features["market_fact"]
    fact["source_consistency"] = {
        "version": "2026-07-17.okx-source-consistency.v4",
        "status": "clean",
        "reasons": [],
        "reference_warnings": ["optional_index_path_not_collected"],
        "assertions": {
            "native_identity_verified": True,
            "executable_quotes_verified": True,
            "tick_alignment_verified": True,
            "reference_prices_verified": True,
            "market_continuity_verified": True,
        },
        "native_identity": {"specification": ["x" * 400] * 20},
        "quote_reconciliation": {"diagnostic_samples": ["x" * 400] * 20},
        "path": {"missing_open_times_ms": list(range(2_000))},
        "reference_observations": {"details": ["x" * 400] * 20},
        "orderbook_fact": {"raw_depth": [100.0] * 200},
        "reference_prices": {"raw_mark": {"book": [101.0] * 200}},
        "ticker_sources": [
            {
                "fact_id": f"sha256:{index:064d}",
                "source_interface": "rest",
                "source_timestamp_ms": 1_798_000_000_000 + index,
            }
            for index in range(8)
        ],
        "executable_intervals": [
            {"source": "rest", "bid": 99.9, "ask": 100.1}
        ],
        "provenance": {
            "data_fingerprint": "source-consistency-digest",
            "fallback_reason": "",
        },
    }
    features.pop("feature_availability_contract")
    seal_feature_availability(features, frozen_at=FROZEN_AT)

    assert frozen_feature_payload_size(features) <= FROZEN_TRAINING_FEATURE_MAX_BYTES
    compacted = _compact_training_feature_snapshot(features)
    sample["features"] = compacted

    assert "training_feature_compaction_error" not in compacted
    assert compacted["market_fact"]["prices"] == {
        key: value for key, value in fact["prices"].items() if value is not None
    }
    assert compacted["market_fact"]["liquidity"] == fact["liquidity"]
    assert compacted["market_fact"]["source_consistency"]["assertions"] == (
        fact["source_consistency"]["assertions"]
    )
    assert "orderbook_fact" not in compacted["market_fact"]["source_consistency"]


def test_market_fact_with_old_storage_contract_is_recompacted() -> None:
    sample = _full_feature_vector_sample()
    fact = sample["features"]["market_fact"]
    fact["source_consistency"].setdefault("path", {})
    fact["source_consistency"]["path"]["missing_open_times_ms"] = list(range(2_000))
    fact["storage_contract"] = {
        "version": "2026-11-12.compact-market-fact.v2",
        "full_fact_fingerprint": fact["fact_id"],
        "full_source_consistency_fingerprint": "old-source-consistency",
    }

    compact = compact_market_fact(fact)

    assert compact["storage_contract"]["full_fact_fingerprint"] == fact["fact_id"]
    assert "missing_open_times_ms" not in compact["source_consistency"]["path"]
    assert "missing_open_times_count" in compact["source_consistency"]["path"]
    assert training_feature_storage_bytes(compact) < training_feature_storage_bytes(fact)
    assert training_feature_storage_bytes({"market_fact": fact}) > (
        FROZEN_TRAINING_FEATURE_MAX_BYTES
    )
    assert frozen_feature_payload_size({"market_fact": compact}) < (
        FROZEN_TRAINING_FEATURE_MAX_BYTES
    )


def test_frozen_budget_audit_uses_production_compacted_projection() -> None:
    from scripts.audit_online_frozen_feature_budget import REMOTE_SCRIPT

    features = _full_feature_vector_sample()["features"]
    features["market_fact"]["source_consistency"] = {
        "version": "2026-07-17.okx-source-consistency.v4",
        "status": "clean",
        "assertions": {"executable_quotes_verified": True},
        "path": {"missing_open_times_ms": list(range(2_000))},
        "reference_prices": {"raw_mark": {"book": [101.0] * 200}},
    }

    compacted = compact_frozen_feature_payload(features)

    assert "training_feature_storage_bytes(frozen)" not in REMOTE_SCRIPT
    assert "frozen_feature_payload_size(features)" in REMOTE_SCRIPT
    assert (
        training_feature_storage_bytes(features["market_fact"])
        > FROZEN_TRAINING_FEATURE_MAX_BYTES
    )
    assert frozen_feature_payload_size(features) <= FROZEN_TRAINING_FEATURE_MAX_BYTES
    assert compacted["market_fact"]["source_consistency"]["assertions"] == {
        "executable_quotes_verified": True
    }
    assert (
        "missing_open_times_ms"
        not in compacted["market_fact"]["source_consistency"]["path"]
    )


def test_compaction_metadata_binds_original_fingerprint_without_storing_it_twice() -> None:
    features = _full_feature_vector_sample()["features"]

    compacted = compact_frozen_feature_payload(features)
    storage_contract = compacted["feature_storage_compaction"]

    assert storage_contract["version"] == FEATURE_STORAGE_COMPACTION_VERSION
    assert "full_input_fingerprint" not in storage_contract
    assert storage_contract["compact_input_fingerprint"] == compact_input_fingerprint(
        compacted,
        features["feature_availability_contract"]["input_fingerprint"],
    )


def test_v2_compaction_fits_payload_that_legacy_metadata_put_over_budget() -> None:
    template = _full_feature_vector_sample()["features"]
    template.pop("feature_availability_contract")
    template.pop("feature_snapshot_frozen_at")
    selected = None
    for length in range(4_400, 4_700):
        features = deepcopy(template)
        features["sequence_quality_warning"] = "x" * length
        seal_feature_availability(features, frozen_at=FROZEN_AT)
        compacted = compact_frozen_feature_payload(features)
        legacy = deepcopy(compacted)
        compact_input = {
            key: compacted[key]
            for key in FROZEN_FEATURE_FIELDS
            if key in compacted
        }
        legacy["feature_storage_compaction"] = {
            "version": LEGACY_FEATURE_STORAGE_COMPACTION_VERSION,
            "full_input_fingerprint": (
                features["feature_availability_contract"]["input_fingerprint"]
            ),
            "compact_input_fingerprint": storage_feature_fingerprint(compact_input),
        }
        legacy_size = training_feature_storage_bytes(legacy)
        compacted_size = training_feature_storage_bytes(compacted)
        if (
            legacy_size > FROZEN_TRAINING_FEATURE_MAX_BYTES
            and compacted_size <= FROZEN_TRAINING_FEATURE_MAX_BYTES
        ):
            selected = features
            break

    assert selected is not None
    assert selected["feature_availability_contract"].get("seal_error") is None

    oversized = deepcopy(selected)
    oversized["sequence_quality_warning"] += "x" * 512
    oversized["feature_availability_contract"]["input_fingerprint"] = (
        availability_fingerprint(
            {
                key: oversized[key]
                for key in FROZEN_FEATURE_FIELDS
                if key in oversized
            }
        )
    )
    oversized = compact_frozen_feature_payload(oversized)
    assert frozen_feature_payload_size(oversized) > FROZEN_TRAINING_FEATURE_MAX_BYTES
    reasons = feature_availability_reasons(
        oversized,
        decision_at=oversized["original_decision_timestamp"],
    )
    assert "feature_availability_frozen_input_budget_exceeded" in reasons


def test_legacy_v1_training_compaction_is_read_and_upgraded() -> None:
    features = _full_feature_vector_sample()["features"]
    legacy = compact_frozen_feature_payload(features)
    compact_input = {
        key: legacy[key]
        for key in FROZEN_FEATURE_FIELDS
        if key in legacy
    }
    legacy["feature_storage_compaction"] = {
        "version": LEGACY_FEATURE_STORAGE_COMPACTION_VERSION,
        "full_input_fingerprint": (
            features["feature_availability_contract"]["input_fingerprint"]
        ),
        "compact_input_fingerprint": storage_feature_fingerprint(compact_input),
    }

    upgraded = compact_training_feature_snapshot(legacy)

    assert upgraded["feature_storage_compaction"]["version"] == (
        FEATURE_STORAGE_COMPACTION_VERSION
    )
    assert "full_input_fingerprint" not in upgraded["feature_storage_compaction"]
    assert feature_availability_reasons(
        legacy,
        decision_at=legacy["original_decision_timestamp"],
    ) == feature_availability_reasons(
        compact_frozen_feature_payload(features),
        decision_at=features["original_decision_timestamp"],
    )


def test_compacted_market_fact_input_tampering_breaks_storage_fingerprint() -> None:
    features = _full_feature_vector_sample()["features"]
    compacted = _compact_training_feature_snapshot(features)
    compacted["market_fact"]["prices"]["last"] += 1

    assert "feature_inputs_changed_after_freeze" in feature_availability_reasons(
        compacted,
        decision_at=compacted["original_decision_timestamp"],
    )


def test_verbose_market_fact_recompaction_keeps_storage_fingerprint() -> None:
    features = _full_feature_vector_sample()["features"]
    features["market_fact"]["source_consistency"] = {
        "version": "2026-07-17.okx-source-consistency.v4",
        "status": "clean",
        "assertions": {"executable_quotes_verified": True},
        "path": {"missing_open_times_ms": list(range(2000))},
        "reference_prices": {"raw_mark": {"book": [101.0] * 200}},
        "provenance": {"data_fingerprint": "source-consistency-digest"},
    }
    features.pop("feature_availability_contract")
    features.pop("feature_snapshot_frozen_at")
    seal_feature_availability(features, frozen_at=FROZEN_AT)
    compacted = _compact_training_feature_snapshot(features)

    assert "training_feature_compaction_error" not in compacted
    assert _compact_training_feature_snapshot(compacted) == compacted


def _specialist_shadow_payload() -> dict[str, object]:
    return {
        "market_fact": _trainable_shadow_sample()["features"]["market_fact"],
        "local_ai_tools_shadow": {
            "status": "ok",
            "captured_at": "2026-11-09T13:00:00+00:00",
            "time_series_prediction": {
                "available": True,
                "actual_inference": True,
                "best_side": "long",
                "expected_return_pct": 1.25,
                "professional_model_shadow": {
                    "kind": "timeseries",
                    "actual_inference": True,
                    "shadow_result_collected": True,
                    "shadow_result": {
                        "model": "google/timesfm-2.5-200m-pytorch",
                        "available": True,
                        "actual_inference": True,
                        "expected_return_pct": 1.25,
                        "best_side": "long",
                        "sequence_length": 240,
                        "status": "completed",
                    },
                },
                "debug_payload": "x" * 40000,
            },
        },
        "local_ml_shadow": {
            "version": "2026-11-09.local-ml-shadow.v1",
            "available": True,
            "actual_inference": True,
            "model_version": "model-v2",
            "primary_horizon_minutes": 240,
            "predictions": [
                {
                    "horizon_minutes": 240,
                    "best_side": "long",
                    "long": {"objective_net_return_pct": 1.1},
                    "short": {"objective_net_return_pct": -0.4},
                    "debug_payload": "y" * 40000,
                }
            ],
        },
    }


def test_specialist_shadow_survives_compaction_without_full_payload() -> None:
    features = _specialist_shadow_payload()

    compacted = _compact_training_feature_snapshot(features)

    assert "training_feature_compaction_error" not in compacted
    assert training_feature_storage_bytes(compacted) <= TRAINING_FEATURE_SNAPSHOT_MAX_BYTES
    tools = compacted["local_ai_tools_shadow"]
    result = tools["time_series_prediction"]["professional_model_shadow"]["shadow_result"]
    assert result["actual_inference"] is True
    assert result["expected_return_pct"] == 1.25
    assert "debug_payload" not in tools["time_series_prediction"]
    ml = compacted["local_ml_shadow"]
    assert ml["predictions"][0]["long"]["objective_net_return_pct"] == 1.1
    assert "debug_payload" not in ml["predictions"][0]
    assert _compact_training_feature_snapshot(features) == compacted


def test_sparse_specialist_shadow_preserves_only_observed_fields() -> None:
    features = {
        "local_ml_shadow": {
            "available": True,
            "raw_payload": "x" * 5000,
            "predictions": [
                {},
                {"horizon_minutes": 240, "long": {}, "short": None},
            ],
        },
        "local_ai_tools_shadow": {
            "raw_payload": "x" * 5000,
            "profit_prediction": {"available": False},
            "time_series_prediction": {"professional_model_shadow": {}},
            "sentiment_analysis": {
                "professional_model_shadow": {"actual_inference": False},
            },
        },
    }

    compacted = compact_training_feature_snapshot(features)

    assert compacted == {
        "local_ml_shadow": {
            "available": True,
            "predictions": [{"horizon_minutes": 240}],
        },
        "local_ai_tools_shadow": {
            "profit_prediction": {"available": False},
            "sentiment_analysis": {
                "professional_model_shadow": {"actual_inference": False},
            },
        },
    }
    assert compact_training_feature_snapshot(compacted) == compacted


def test_postgres_compaction_retains_specialist_shadow_projection() -> None:
    functions = postgres_training_feature_snapshot_functions()
    shadow_sql = next(
        sql for sql in functions if "bb_compact_specialist_shadow" in sql
    )
    compact_sql = functions[-1]

    assert "local_ai_tools_shadow" in shadow_sql
    assert "local_ml_shadow" in shadow_sql
    assert "'professional_model_shadow'" in shadow_sql
    assert "bb_compact_specialist_shadow(raw)" in compact_sql


def test_optional_diagnostics_are_dropped_when_they_exceed_snapshot_budget() -> None:
    features = _trainable_shadow_sample()["features"]
    features.update(
        {
            f"optional_diagnostic_{index:03d}": "x" * 512
            for index in range(220)
        }
    )

    compacted = _compact_training_feature_snapshot(features)

    assert "training_feature_compaction_error" not in compacted
    assert frozen_feature_payload_size(compacted) <= FROZEN_TRAINING_FEATURE_MAX_BYTES
    assert training_feature_storage_bytes(compacted) <= TRAINING_FEATURE_SNAPSHOT_MAX_BYTES
    assert "optional_diagnostic_000" in compacted
    assert "optional_diagnostic_219" not in compacted
    assert _compact_training_feature_snapshot(features) == compacted


def test_shadow_exit_cost_observation_survives_optional_budget_without_entering_seal() -> None:
    features = _full_feature_vector_sample()["features"]
    original_seal = deepcopy(features["feature_availability_contract"])
    observation = {
        "version": "2026-11-11.observed-shadow-exit-cost.v2",
        "used_as_decision_time_features": False,
        "market_snapshot": {"bid": 101.0, "ask": 101.1, "taker_fee_rate": 0.0005},
        "entry_price": 100.0,
        "exit_price": 101.05,
    }
    features["shadow_settlement_cost_observation"] = observation
    features.update({f"optional_diagnostic_{index:03d}": "x" * 512 for index in range(220)})

    compacted = _compact_training_feature_snapshot(features)

    assert compacted["shadow_settlement_cost_observation"] == observation
    assert compacted["feature_availability_contract"] == original_seal
    assert "optional_diagnostic_219" not in compacted
    assert "shadow_settlement_cost_observation" not in FROZEN_FEATURE_FIELDS
    assert _compact_training_feature_snapshot(compacted) == compacted


def test_optional_metadata_cannot_overwrite_normalized_market_contract() -> None:
    features = _full_feature_vector_sample()["features"]
    window = {
        "start": "2026-11-11T00:00:00+00:00",
        "end": "2026-11-11T00:05:00+00:00",
    }
    features["training_market_fact_contract"] = {
        "version": MARKET_FACT_CONTRACT_VERSION,
        "status": "clean",
        "observation_window": json.dumps(window),
    }

    compacted = _compact_training_feature_snapshot(features)

    assert compacted["training_market_fact_contract"]["observation_window"] == window
    assert _compact_training_feature_snapshot(compacted) == compacted
    assert isinstance(features["training_market_fact_contract"]["observation_window"], str)


def test_specialist_evidence_is_preserved_one_group_at_a_time(monkeypatch) -> None:
    features = {
        "local_ml_shadow": {"available": True},
        "local_ai_tools_shadow": {"available": True},
    }
    monkeypatch.setattr(
        "core.training_feature_snapshot.TRAINING_FEATURE_SNAPSHOT_MAX_BYTES",
        training_feature_storage_bytes({"local_ml_shadow": {"available": True}}),
    )

    compacted = _compact_training_feature_snapshot(features)

    assert compacted == {"local_ml_shadow": {"available": True}}


def _mixed_market_fact_contract() -> dict:
    return {
        "version": MARKET_FACT_CONTRACT_VERSION,
        "status": "historical_ohlcv_only",
        "price_path": {"status": "historical_ohlcv_only"},
        "path_status": "clean",
        "data_fingerprint": "d" * 64,
        "path_fingerprint": "p" * 64,
        "native_instrument_identity_verified": True,
        "executable_market_fact_verified": False,
        "source": HISTORICAL_SHADOW_SOURCE,
        "scheduled_label_at": "2026-11-11T00:05:25+00:00",
        "result_source_available_at": "2026-11-11T00:06:00+00:00",
        "label_maturity_lag_seconds": 35.0,
        "delivery_latency_observed": False,
        "observation_window": json.dumps({"start": "entry", "end": "exit"}),
        "entry_fact": {"fact_id": "entry-fact"},
        "entry_inst_id": "BTC-USDT-SWAP",
        "result_fact_id": "result-fact",
        "violation_reason_codes": "historical_only",
    }


def test_mixed_market_contract_preserves_flat_provenance() -> None:
    contract = _mixed_market_fact_contract()

    compacted = _compact_training_feature_snapshot({
        "training_market_fact_contract": contract,
    })["training_market_fact_contract"]

    for key, value in contract.items():
        if key not in {"price_path", "entry_fact", "observation_window"}:
            assert compacted[key] == value
    assert compacted["entry_fact_id"] == "entry-fact"
    assert compacted["observation_window"] == {"start": "entry", "end": "exit"}


def test_mixed_market_contract_does_not_override_nested_rejection() -> None:
    contract = _mixed_market_fact_contract()
    contract["assertions"] = {"native_instrument_identity_verified": False}
    contract["provenance"] = {
        "data_fingerprint": None,
        "label_maturity_lag_seconds": 0,
        "delivery_latency_observed": None,
        "result_source_available_at": None,
        "observation_window": None,
    }

    compacted = _compact_training_feature_snapshot({
        "training_market_fact_contract": contract,
    })["training_market_fact_contract"]

    assert compacted["native_instrument_identity_verified"] is False
    assert compacted["data_fingerprint"] is None
    assert compacted["label_maturity_lag_seconds"] == 0
    assert compacted["delivery_latency_observed"] is None
    assert compacted["result_source_available_at"] is None
    assert compacted["observation_window"] is None
    assert compacted["scheduled_label_at"] == contract["scheduled_label_at"]


def test_mixed_market_contract_uses_flat_window_when_nested_value_is_missing() -> None:
    contract = _mixed_market_fact_contract()
    contract["provenance"] = {"data_fingerprint": "d" * 64}

    compacted = _compact_training_feature_snapshot({
        "training_market_fact_contract": contract,
    })["training_market_fact_contract"]

    assert compacted["observation_window"] == {"start": "entry", "end": "exit"}


def test_postgres_optional_compaction_uses_non_conflicting_loop_record() -> None:
    functions = postgres_training_feature_snapshot_functions()
    compact_sql = functions[-1]
    market_contract_sql = next(
        sql
        for sql in functions
        if "CREATE OR REPLACE FUNCTION bb_compact_training_market_fact_contract" in sql
    )

    assert "optional_item RECORD" in compact_sql
    assert "FOR optional_item IN" in compact_sql
    assert "jsonb_build_object(\n                    optional_item.key" in compact_sql
    assert "bb_compact_training_market_fact_contract(" in compact_sql
    assert "'training_market_fact_contract', required_contract" in compact_sql
    assert "FOR item IN" not in compact_sql
    assert "'scheduled_label_at'," in market_contract_sql
    assert "'result_source_available_at'," in market_contract_sql
    assert "'availability_basis'," in market_contract_sql
    assert "'path_status', COALESCE(source->'path_status', path->'status')" in market_contract_sql


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
    assert sample["features"] == {
        "training_feature_compaction_error": "frozen_input_storage_mismatch"
    }


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
async def test_postgres_and_python_snapshot_compaction_match_in_temporary_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncpg

    connection = await asyncpg.connect(os.environ["BB_TEST_POSTGRES_URL"])
    transaction = connection.transaction()
    await transaction.start()
    try:
        await connection.execute("SET LOCAL statement_timeout = '15s'")
        await connection.execute("SET LOCAL lock_timeout = '5s'")
        await connection.execute("CREATE TEMP TABLE bb_compaction_test_scope (id INTEGER)")
        await connection.execute("SET LOCAL search_path TO pg_temp, public")
        function_names = (
            "bb_training_feature_storage_bytes",
            "bb_training_feature_normalize_jsonb",
            "bb_compact_market_fact",
            "bb_compact_training_market_fact_contract",
            "bb_compact_training_feature_snapshot",
            "bb_project_specialist_shadow_evidence",
            "bb_compact_specialist_shadow",
        )
        for sql in postgres_training_feature_snapshot_functions():
            for function_name in function_names:
                sql = sql.replace(f"{function_name}(", f"pg_temp.{function_name}(")
            await connection.execute(sql)
        original = _full_feature_vector_sample()["features"]
        oversized = deepcopy(original)
        oversized["close_sequence"] = [100.0] * 1000
        oversized["feature_availability_contract"]["input_fingerprint"] = (
            availability_fingerprint({
                key: oversized[key]
                for key in FROZEN_FEATURE_FIELDS
                if key in oversized
            })
        )
        with_contract = deepcopy(original)
        with_contract["training_market_fact_contract"] = (
            _large_native_market_fact_contract()
        )
        with_contract["shadow_settlement_cost_observation"] = {
            "version": "2026-11-11.observed-shadow-exit-cost.v2",
            "used_as_decision_time_features": False,
            "market_snapshot": {"bid": 101.0, "ask": 101.1, "taker_fee_rate": 0.0005},
        }
        with_contract.update(
            {
                f"optional_diagnostic_{index:03d}": "x" * 512
                for index in range(80)
            }
        )
        with_historical_contract = deepcopy(original)
        with_historical_contract["training_market_fact_contract"] = {
            "version": MARKET_FACT_CONTRACT_VERSION,
            "status": "historical_ohlcv_only",
            "violation_count": 0,
            "violation_reason_codes": "",
            "native_instrument_identity_verified": True,
            "same_contract_price_path_verified": True,
            "executable_market_fact_verified": False,
            "source": HISTORICAL_SHADOW_SOURCE,
            "observation_window": json.dumps(
                {
                    "start": "2026-09-01T00:00:00+00:00",
                    "end": "2026-09-01T00:06:00+00:00",
                }
            ),
            "sample_count": 6,
            "effective_sample_size": 1.0,
            "generated_at": "2026-09-01T00:06:01+00:00",
            "strategy_version": "historical-test-v1",
            "fallback_reason": "historical_ohlcv_has_no_historical_orderbook",
            "data_fingerprint": "a" * 64,
            "entry_fact_id": None,
            "result_fact_id": None,
            "result_source_timestamp_ms": None,
            "result_received_at": None,
            "scheduled_label_at": "2026-09-01T00:05:25+00:00",
            "result_source_available_at": "2026-09-01T00:06:00+00:00",
            "label_maturity_lag_seconds": 35.0,
            "availability_basis": "one_minute_candle_close_lower_bound",
            "delivery_latency_observed": False,
            "entry_inst_id": None,
            "result_inst_id": None,
            "entry_contract_spec_version": None,
            "result_contract_spec_version": None,
            "path_status": "clean",
            "path_fingerprint": "b" * 64,
        }
        with_small_contract = deepcopy(original)
        with_small_contract["training_market_fact_contract"] = {
            "version": MARKET_FACT_CONTRACT_VERSION,
            "status": "clean",
            "observation_window": json.dumps({
                "start": "2026-11-11T00:00:00+00:00",
                "end": "2026-11-11T00:05:00+00:00",
            }),
        }
        with_mixed_contract = {"training_market_fact_contract": _mixed_market_fact_contract()}
        with_nested_rejection = deepcopy(with_mixed_contract)
        with_nested_rejection["training_market_fact_contract"].update({
            "assertions": {"native_instrument_identity_verified": False},
            "provenance": {
                "data_fingerprint": None,
                "label_maturity_lag_seconds": 0,
                "delivery_latency_observed": None,
                "result_source_available_at": None,
                "observation_window": None,
            },
        })
        with_missing_nested_window = deepcopy(with_mixed_contract)
        with_missing_nested_window["training_market_fact_contract"]["provenance"] = {
            "data_fingerprint": "d" * 64,
        }
        with_sparse_specialists = {
            "local_ml_shadow": {
                "available": True,
                "raw_payload": "x" * 5000,
                "predictions": [
                    {},
                    {"horizon_minutes": 240, "long": {}, "short": None},
                ],
            },
            "local_ai_tools_shadow": {
                "raw_payload": "x" * 5000,
                "profit_prediction": {"available": False},
                "time_series_prediction": {"professional_model_shadow": {}},
                "sentiment_analysis": {
                    "professional_model_shadow": {"actual_inference": False},
                },
            },
        }
        with_empty_specialist_result = {
            "local_ai_tools_shadow": {
                "raw_payload": "x" * 5000,
                "profit_prediction": {"professional_model_shadow": {"shadow_result": {}}},
            },
        }
        with_long_specialist_strings = {
            "local_ml_shadow": {
                "available": True,
                "raw_payload": "x" * 5000,
                "model_version": "m" * 1000,
                "predictions": [{"best_side": "s" * 1000}],
            },
            "local_ai_tools_shadow": {
                "raw_payload": "x" * 5000,
                "status": "s" * 1000,
                "profit_prediction": {
                    "fallback_reason": "r" * 1000,
                    "professional_model_shadow": {
                        "shadow_result": {"reason": "r" * 1000},
                    },
                },
            },
        }
        for features in (
            original,
            oversized,
            with_contract,
            with_historical_contract,
            with_small_contract,
            with_mixed_contract,
            with_nested_rejection,
            with_missing_nested_window,
            with_sparse_specialists,
            with_empty_specialist_result,
            with_long_specialist_strings,
            {"empty": {}, "array": [], "text": "x" * 513},
        ):
            encoded = json.dumps(features, ensure_ascii=False)
            pg_compacted = json.loads(await connection.fetchval(
                "SELECT pg_temp.bb_compact_training_feature_snapshot($1::JSONB)", encoded,
            ))
            server_recomputed_fingerprint = await connection.fetchval(
                """
                WITH compacted AS (
                    SELECT pg_temp.bb_compact_training_feature_snapshot($1::JSONB) AS snapshot
                ),
                projection AS (
                    SELECT COALESCE(jsonb_object_agg(item.key, item.value), '{}'::JSONB) AS value
                    FROM compacted
                    CROSS JOIN LATERAL jsonb_each(compacted.snapshot) AS item(key, value)
                    WHERE item.key = ANY($2::TEXT[])
                )
                SELECT md5(
                    pg_temp.bb_training_feature_normalize_jsonb(
                        jsonb_strip_nulls(projection.value)
                    )::TEXT
                )
                FROM projection
                """,
                encoded,
                list(FROZEN_FEATURE_FIELDS),
            )
            manually_rebuilt_fingerprint = await connection.fetchval(
                """
                WITH raw AS (
                    SELECT $1::JSONB AS value
                ),
                frozen AS (
                    SELECT COALESCE(jsonb_object_agg(item.key, item.value), '{}'::JSONB) AS value
                    FROM raw
                    CROSS JOIN LATERAL jsonb_each(raw.value) AS item(key, value)
                    WHERE item.key = ANY($2::TEXT[])
                      AND item.key <> 'feature_storage_compaction'
                ),
                compacted AS (
                    SELECT CASE
                        WHEN frozen.value ? 'market_fact' THEN jsonb_set(
                            frozen.value,
                            '{market_fact}',
                            pg_temp.bb_compact_market_fact(frozen.value->'market_fact'),
                            true
                        )
                        ELSE frozen.value
                    END AS value
                    FROM frozen
                ),
                projection AS (
                    SELECT COALESCE(jsonb_object_agg(item.key, item.value), '{}'::JSONB) AS value
                    FROM compacted
                    CROSS JOIN LATERAL jsonb_each(compacted.value) AS item(key, value)
                    WHERE item.key = ANY($3::TEXT[])
                )
                SELECT md5(
                    pg_temp.bb_training_feature_normalize_jsonb(
                        jsonb_strip_nulls(
                            projection.value || jsonb_build_object(
                                '__feature_availability_input_fingerprint__',
                                raw.value->'feature_availability_contract'
                                    ->>'input_fingerprint'
                            )
                        )
                    )::TEXT
                )
                FROM projection
                CROSS JOIN raw
                """,
                encoded,
                list(FROZEN_TRAINING_FEATURE_KEYS),
                list(FROZEN_FEATURE_FIELDS),
            )
            raw_market_fingerprint = await connection.fetchval(
                """
                WITH raw AS (
                    SELECT $1::JSONB AS value
                ),
                frozen AS (
                    SELECT COALESCE(jsonb_object_agg(item.key, item.value), '{}'::JSONB) AS value
                    FROM raw
                    CROSS JOIN LATERAL jsonb_each(raw.value) AS item(key, value)
                    WHERE item.key = ANY($2::TEXT[])
                      AND item.key <> 'feature_storage_compaction'
                ),
                projection AS (
                    SELECT COALESCE(jsonb_object_agg(item.key, item.value), '{}'::JSONB) AS value
                    FROM frozen
                    CROSS JOIN LATERAL jsonb_each(frozen.value) AS item(key, value)
                    WHERE item.key = ANY($3::TEXT[])
                )
                SELECT md5(
                    pg_temp.bb_training_feature_normalize_jsonb(
                        jsonb_strip_nulls(projection.value)
                    )::TEXT
                )
                FROM projection
                """,
                encoded,
                list(FROZEN_TRAINING_FEATURE_KEYS),
                list(FROZEN_FEATURE_FIELDS),
            )
            expected_compacted = _compact_training_feature_snapshot(features)
            differences = _json_differences(pg_compacted, expected_compacted)
            if "training_feature_compaction_error" in expected_compacted:
                assert pg_compacted == expected_compacted, differences
                assert await connection.fetchval(
                    "SELECT pg_temp.bb_training_feature_storage_bytes($1::JSONB)", encoded,
                ) == training_feature_storage_bytes(features)
                continue
            storage_contract = pg_compacted.get("feature_storage_compaction")
            if "feature_availability_contract" in features:
                assert isinstance(storage_contract, dict), pg_compacted
                assert (
                    manually_rebuilt_fingerprint
                    == storage_contract.get("compact_input_fingerprint")
                ), (
                    manually_rebuilt_fingerprint,
                    raw_market_fingerprint,
                    storage_contract.get("compact_input_fingerprint"),
                )
            else:
                assert storage_contract is None
            pg_input = {
                key: pg_compacted[key]
                for key in FROZEN_FEATURE_FIELDS
                if key in pg_compacted
            }
            py_input = {
                key: expected_compacted[key]
                for key in FROZEN_FEATURE_FIELDS
                if key in expected_compacted
            }
            pg_input_text = await connection.fetchval(
                "SELECT pg_temp.bb_training_feature_normalize_jsonb("
                "jsonb_strip_nulls($1::JSONB))::TEXT",
                json.dumps(pg_input, ensure_ascii=False),
            )
            py_input_text = _jsonb_text(_storage_fingerprint_value(py_input))
            if pg_input_text != py_input_text:
                mismatch_at = next(
                    (
                        index
                        for index, (pg_char, py_char) in enumerate(
                            zip(pg_input_text, py_input_text, strict=False)
                        )
                        if pg_char != py_char
                    ),
                    min(len(pg_input_text), len(py_input_text)),
                )
                text_difference = (
                    "jsonb text differs at "
                    f"{mismatch_at}: postgres={pg_input_text[max(0, mismatch_at - 40):mismatch_at + 80]!r}, "
                    f"python={py_input_text[max(0, mismatch_at - 40):mismatch_at + 80]!r}"
                )
                assert pg_input_text == py_input_text, text_difference
            pg_input_fingerprint = await connection.fetchval(
                "SELECT md5(pg_temp.bb_training_feature_normalize_jsonb("
                "jsonb_strip_nulls($1::JSONB))::TEXT)",
                json.dumps(pg_input, ensure_ascii=False),
            )
            assert pg_input_fingerprint == storage_feature_fingerprint(py_input)
            if isinstance(storage_contract, dict):
                assert server_recomputed_fingerprint == pg_input_fingerprint
                assert (
                    storage_contract.get("compact_input_fingerprint")
                    == manually_rebuilt_fingerprint
                )
            assert not differences, differences
            pg_recompacted = json.loads(await connection.fetchval(
                "SELECT pg_temp.bb_compact_training_feature_snapshot($1::JSONB)",
                json.dumps(pg_compacted, ensure_ascii=False),
            ))
            assert pg_recompacted == pg_compacted, _json_differences(
                pg_recompacted, pg_compacted
            )
            assert compact_training_feature_snapshot(expected_compacted) == expected_compacted
            assert await connection.fetchval(
                "SELECT pg_temp.bb_training_feature_storage_bytes($1::JSONB)", encoded,
            ) == training_feature_storage_bytes(features)
            assert await connection.fetchval(
                "SELECT pg_column_size($1::JSONB)", json.dumps(pg_compacted),
            ) <= TRAINING_FEATURE_SNAPSHOT_MAX_BYTES
        partial_shadow = {
            "local_ml_shadow": {"available": True},
            "local_ai_tools_shadow": {"available": True},
        }
        monkeypatch.setattr(
            "core.training_feature_snapshot.TRAINING_FEATURE_SNAPSHOT_MAX_BYTES",
            training_feature_storage_bytes({"local_ml_shadow": {"available": True}}),
        )
        for sql in postgres_training_feature_snapshot_functions():
            for function_name in function_names:
                sql = sql.replace(f"{function_name}(", f"pg_temp.{function_name}(")
            await connection.execute(sql)
        pg_partial = json.loads(await connection.fetchval(
            "SELECT pg_temp.bb_compact_training_feature_snapshot($1::JSONB)",
            json.dumps(partial_shadow),
        ))
        assert pg_partial == _compact_training_feature_snapshot(partial_shadow)
        assert pg_partial == {"local_ml_shadow": {"available": True}}
    finally:
        await transaction.rollback()
        await connection.close()
